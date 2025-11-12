from email import generator
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader

import os
from models.GAN_mri_package.model_3D import StyledGenerator, Discriminator
import argparse
from models.GAN_mri_package.dataset import MriFileFolderDataset
import pdb
import SimpleITK as sitk
import torch.nn.functional as F
import pickle
import ants
##
## w_coarse: tensor or np array: (1, code_dim)
import time
import sys

from models.e4e.psp import pSp
import matplotlib.pyplot as plt
import csv

def compute_loss(img1, img2, loss_type, device):

    if loss_type == 'L1':

        loss = F.l1_loss(img1, img2)

    elif loss_type == 'FFT':
        fftn1 = torch.fft.fftn(img1)
        fftn2 = torch.fft.fftn(img2)
        
        dis = abs(fftn1 - fftn2)
        #pdb.set_trace()

        loss = F.l1_loss(dis, torch.zeros_like(dis, dtype=torch.float, device=device))
    
    return loss


def get_bl_file(img_dir):
    
    bl_file_name = []
    bl_file_paths = []
    
    for subject in os.listdir(img_dir):
        subject_folder = os.path.join(img_dir, subject)
        dates = sorted(os.listdir(subject_folder))
        if len(dates) == 1:
            pass
        else:
            bl_date = dates[0]
            bl_file_name.append(os.listdir(os.path.join(subject_folder, bl_date))[0])
            bl_file_paths.append(os.path.join(subject_folder, bl_date, bl_file_name[-1]))
    
    return bl_file_name, bl_file_paths

def get_bl_date(img_dir):
    
    bl_subject_date = {}
    
    for subject in os.listdir(img_dir):
        subject_folder = os.path.join(img_dir, subject)
        dates = sorted(os.listdir(subject_folder))
        if len(dates) == 1:
            pass
        else:
            bl_date = dates[0]
            bl_subject_date[subject] = bl_date
    
    return bl_subject_date

def get_bl_mean_from_csv(csv_pth):

    subejct_bl_mean = {}
    with open(csv_pth, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            subject_id = row['subject ID']
            bl_mean = row['bl_mean']

            subejct_bl_mean[subject_id] = float(bl_mean)
    
    return subejct_bl_mean




def check_grad(model, search_noise):
    
    w_grad_norm = torch.norm(model.w.grad).item()
    if search_noise:
        noise_grad_norm = [torch.norm(noise.grad).item() for noise in model.noise]
    else:
        noise_grad_norm = [0 for noise in model.noise]

    return w_grad_norm, noise_grad_norm




class optimise_w(nn.Module):

    def __init__(self, w_coarse, noise_coarse, search_noise=False, w_star=True, n_latent=12, device='cuda', train_noise_idx=None):
        
        super(optimise_w, self).__init__()

        if not torch.is_tensor(w_coarse):
            w_coarse = torch.tensor(w_coarse)

        if not w_star:
            self.w0 = nn.Parameter(w_coarse)
            #w = w0.repeat(n_latent, 1) ## (n_latent, code_dim)
        else:
            if w_coarse.shape[0] == 1:
                w = w_coarse.repeat(n_latent, 1)
            else:
                w = w_coarse
            w = nn.Parameter(w)
            self.w = w
        

        self.step = 5
        self.batch = 1
        self.search_noise = search_noise
        self.w_star = w_star
        self.n_latent = n_latent

        if noise_coarse is None:
            noise = []
            for i in range(self.step + 1):
                size_x = 6 * 2 ** i
                size_y = 7 * 2 ** i 
                size_z = 6 * 2 ** i 
                noise.append(nn.Parameter(torch.randn((self.batch, 1, size_x, size_y, size_z))))
        else:
            noise = [nn.Parameter(noise_) for noise_ in noise_coarse]
        
        if search_noise:
            for i in range(self.step + 1):
                if i in train_noise_idx:
                    noise[i].requires_grad = True
                else:
                    noise[i].requires_grad = False
            self.noise = nn.ParameterList(noise)
        
        else:
            for i in range(self.step + 1):
                noise[i].requires_grad = False
            self.noise = noise

    
    def set_w_from_w0(self):
        self.w = self.w0.repeat(self.n_latent, 1)
        
    def forward(self, generator, step=5):
 
        if self.w_star:
            rec = generator(input=self.w.unsqueeze(0), noise=self.noise, step=step, input_is_latent=True)
        else:
            rec = generator(input=self.w0.repeat(self.n_latent, 1).unsqueeze(0), noise=self.noise, step=step, input_is_latent=True) 

        return rec

    
def optimise_in_image_domain(original_image, generator, max_epochs, loss_type, threshold,
                                search_noise=False,  w_star=True, n_latent=12, downsample_rate=[1, 8], 
                                device='cuda', optim_method='Adam', lr=1e-3, save_interval=10, exp_dir=None, write_image=True, 
                                train_noise_idx=None, w_coarse=None, log_loss=False, noise_coarse=None):

    if write_image:
        print('encoder reconstruction image writing to {}'.format(exp_dir)) 
    if  not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    
    
    if log_loss:
        n_noise = int(n_latent/2)
        w_grad_trace = []
        noise_grad_trace = []
        for i in range(n_noise):
            noise_grad_trace.append([])
    
    print('')
    print('encoding {}, {} begin!'.format(exp_dir.split('/')[-2], exp_dir.split('/')[-1]))


    if original_image.dim() == 3:
        original_image.unsqueeze(0).unsqueeze(0)
    elif original_image.dim() == 4:
        original_image.unsqueeze(0)
    
    #original_image[original_image<(35/256)*torch.max(original_image)] = 0
    original_image = original_image.float()

    if w_coarse is None:
        mean_latent = generator.module.mean_latent(1000)
        w_coarse = mean_latent

    w_net = optimise_w(w_coarse=w_coarse, noise_coarse=noise_coarse,
                       search_noise=search_noise, w_star=w_star, n_latent=n_latent, device=device, train_noise_idx=train_noise_idx)
    w_net.to(device)

    
    #if optim_method == 'SGD':
    #    optimizer = torch.optim.SGD(w_net.parameters(), momentum=0, weight_decay=1e-3, lr=lr)
    #elif optim_method == 'Adam':
        #pdb.set_trace()
    #    optimizer = torch.optim.Adam(w_net.parameters(), lr=lr, betas=(0.9, 0.99))

    if write_image:
        ori_ = F.avg_pool3d(original_image, (downsample_rate[0]))
        ori_ = ori_.cpu().squeeze()
        ori = sitk.GetImageFromArray(ori_)
        sitk.WriteImage(ori, os.path.join(exp_dir, 'ori.nii.gz'))

    loss_ = {}
    optimizer = torch.optim.RAdam(w_net.parameters(), lr=lr, betas=(0.9, 0.999))

    if type(downsample_rate) == int:
        downsample_rate = [downsample_rate]
    for downsample_rate_ in downsample_rate:
        loss_[downsample_rate_] = []
    
    
    else:
        for epoch in range(max_epochs):

            loss = .0
            rec= w_net(generator)

            #th = torch.min(rec) + (torch.max(rec) - torch.min(rec))*(35/256)
            #mask = rec > th 
            #rec = rec * mask


            for downsample_rate_ in downsample_rate:
            
                rec_ = F.avg_pool3d(rec, (downsample_rate_))
                ori_ = F.avg_pool3d(original_image, (downsample_rate_))
                
                loss1 = compute_loss(rec_, ori_, loss_type, device)
                loss += loss1
       
                loss_[downsample_rate_].append(float(loss1))

            if float(loss_[downsample_rate[0]][-1]<threshold) and loss_type == 'FFT':

                #loss_type = 'L1'
                #threshold = 0.005
                pass


            if float(loss_[downsample_rate[0]][-1]<threshold):
                
                rec_ = F.avg_pool3d(rec, (downsample_rate[0]))
                img = rec_.squeeze().detach().cpu().numpy()
                
                
                img = ants.from_numpy(img)
                masked_img = ants.mask_image(img, ants.get_mask(img)).numpy()
                img = sitk.GetImageFromArray(masked_img)
                sitk.WriteImage(img, os.path.join(exp_dir, 'rec_epoch={}_loss={:.5f}.nii.gz'.format(epoch+1, loss_[downsample_rate[0]][-1])))

                break
            else:
                optimizer.zero_grad()
                loss.backward()

                if log_loss:

                    w_grad_norm, noise_grad_norm = check_grad(w_net, search_noise=search_noise)
                    w_grad_trace.append(w_grad_norm)
                    for i in range(n_noise):
                        noise_grad_trace[i].append(noise_grad_norm[i])         
                    
                optimizer.step()

                    
            if (epoch+1) % save_interval == 0 or (epoch+1)==1:
            
                if write_image:
                    rec_ = F.avg_pool3d(rec, (downsample_rate[0]))
                    img = rec_.squeeze().detach().cpu().numpy()
                    img = sitk.GetImageFromArray(img)
                    sitk.WriteImage(img, os.path.join(exp_dir, 'rec_epoch={}_loss={:.5f}.nii.gz'.format(epoch+1, loss_[downsample_rate[0]][-1])))
        
            if (epoch+1) % 1 == 0:
                print('epoch {}, loss {:.6f}'.format(epoch, loss_[downsample_rate[0]][-1]))
            

        
        
        if log_loss:
            
            plt.figure()
            plt.plot(loss_[downsample_rate[0]][500:])
            plt.title('l1 loss')
            plt.xlabel('steps')
            plt.savefig(os.path.join(exp_dir, 'loss.jpg'))
            plt.close()

            plt.figure()
            plt.plot(w_grad_trace[500:])

            plt.title('w grad norm')
            plt.xlabel('steps')
            plt.savefig(os.path.join(exp_dir, 'w grad.jpg'))
            plt.close()

            plt.figure()
            for i in range(n_noise):
                plt.subplot(2, 3, i+1)
                plt.plot(noise_grad_trace[i][500:])
                plt.title('noise {}'.format(i))
            plt.savefig(os.path.join(exp_dir, 'noise grad.jpg'))
            plt.close()
        
        print('encoding {} finished! total epochs: {} loss: {:.6f}'.format(os.path.split(exp_dir)[-1], epoch, loss_[downsample_rate[0]][-1]))
    
    
    if not w_star:
        w_net.set_w_from_w0()
    
    return loss_, w_coarse, w_net.w.detach(), [noise.detach() for noise in w_net.noise]




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train encoder for styleGAN model')
    
    parser.add_argument('--gan_pth', default=None, description='pretrained gan model', type=str)
    parser.add_argument('--encoder_pth', default=None, type=str, description='pretrained encoder model')
    parser.add_argument('--encoder', action='store_true', default=False)

    parser.add_argument('--code_dim', default=128)
    parser.add_argument('--n_latent', default=12)
    parser.add_argument('--not_search_noise', action='store_true', default=False)
    parser.add_argument('--not_w_star', action='store_true', default=False)
    
    parser.add_argument('--cuda_id', default=7, type=int)

    parser.add_argument('--img_dir', default=None, type=str, description='image directory for encoding')
    parser.add_argument('--exp_dir', default='./ImageEncoder/exp', type=str, description='experiment directory')
    parser.add_argument('--bl_csv', default='./ImageEncoder/ADNI/T1/excel/bl_mean.csv', type=str, description='path to bl csv file for image normalization')


    parser.add_argument('--optim_method', default='Adam')
    parser.add_argument('--lr', default=2e-3, type=float)
    parser.add_argument('--loss', default='l1')

    parser.add_argument('--max_epoch', default=10000, type=int)
    parser.add_argument('--save_interval', default=1000, type=int)
    parser.add_argument('--fft_loss_threshold', default=10, type=float)
    parser.add_argument('--l1_loss_threshold', default=0.005, type=float)
    parser.add_argument('--downsample_rate', default=[1], nargs='+', type=int)
    parser.add_argument('--train_noise_idx', default=[0,1,2,3,4,5], nargs='+', type=int)
    parser.add_argument('--encode_bl', action='store_true', default=False)
    #parser.add_argument('--train_noise_idx', default=[4,5], nargs='+', type=int)

    


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_id)
    #pdb.set_trace()
    device = torch.device('cuda:0') if not args.cuda_id == None else torch.device('cpu')
    
    for k, v in vars(args).items():
        print('{} : {}'.format(k, v))
    
    if args.encoder:
        
        args.encoder_type = 'E4E_3d'
        args.checkpoint_path = args.encoder_pth
        args.stylegan_weights = args.gan_pth
        args.start_from_latent_avg = False
        args.device = device
        args.stylegan_size = 256
        args.update_param_list = []
        args.noise_encoder=True

        psp = pSp(args).to(device)

        for params in psp.parameters():
            params.requires_grad = False

    generator = StyledGenerator(args.code_dim, init_size=(6, 7, 6)).to(device)
    generator = nn.DataParallel(generator, device_ids=None)
    
    ckpt = torch.load(args.gan_pth, map_location='cuda:0')
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    
    del ckpt

    for params in generator.parameters():
        params.requires_grad = False
    
    
    img_root = args.img_dir

    
    print('')
    print('exp dir: {}'.format(args.exp_dir))
    print('encoder for images in {}'.format(img_root))
    
    bl_file_names, bl_file_paths = get_bl_file(args.img_dir)
    bl_subject_date = get_bl_date(args.img_dir)
    subject_bl_mean = get_bl_mean_from_csv(csv_pth=args.bl_csv)
        
    training_dataset = MriFileFolderDataset(root=img_root, crop=True, crop_size=(192, 224, 192), return_pth=True, walk=True)
    training_data_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    for batch_idx, batch in enumerate(training_data_loader):

        image, path = batch
        image = image.to(device)
        
        path = path[0]
        subject, date, name = tuple(path.split('/')[-3:])
        dst_folder = os.path.join(args.exp_dir, subject, date)
        codes = None
        noise_list = None
        
        if os.path.exists(dst_folder):
            continue
        
        previous_pkl = os.path.join(args.previous_dir, subject, date, 'w_and_noise.pkl')
        if os.path.exists(previous_pkl) and args.encode_bl:
            
            f_read = open(previous_pkl, 'rb+')
            mri_dict = pickle.load(f_read)

            w = mri_dict['w']
            noise = mri_dict['noise']
            
            if not torch.is_tensor(noise[0]):
                noise_list = [torch.tensor(noise_layer, device=device).unsqueeze(0).unsqueeze(0) for noise_layer in noise]
            else:
                noise_list = [noise_layer.unsqueeze(0).unsqueeze(0).to(device) for noise_layer in noise]

            codes = torch.tensor(w, device=device, dtype=torch.float32).view(12, -1)

        
        if args.encode_bl:
            #loss_type = 'FFT'
            #th = args.fft_loss_threshold
            loss_type = 'L1'
            th = args.l1_loss_threshold
            if path not in bl_file_paths:
                #pdb.set_trace()
                continue
            else:
                #pdb.set_trace()
                if args.encoder and codes is None:
                    with torch.no_grad():
                        codes = psp.encoder(image).squeeze()
                        noise_list = list(psp.noise_encoder(image).values)
                        noise_list.append(torch.randn((image.shape[0], 1, 192, 224, 192), device=device))
                   #noise_list = None
        else:
            #pdb.set_trace()
            loss_type = 'L1'
            th = args.l1_loss_threshold
            #loss_type = 'FFT'
            #th = args.fft_loss_threshold
            if subject not in bl_subject_date.keys() or subject not in subject_bl_mean.keys():
                continue
            bl_date = bl_subject_date[subject]
            bl_mean = subject_bl_mean[subject]
            
            image = image/torch.mean(image)*torch.tensor(bl_mean)


            bl_pkl_pth = os.path.join(args.exp_dir, subject, bl_date, 'w_and_noise.pkl')
            #previous = os.path.join(args.exp_dir, subject, bl_date, 'w_and_noise.pkl')
            
            if os.path.exists(bl_pkl_pth) and len(os.listdir(os.path.join(args.exp_dir, subject, bl_date)))==2:
                f_read = open(bl_pkl_pth, 'rb+')
                mri_dict = pickle.load(f_read)

                w = mri_dict['w']
                noise = mri_dict['noise']
                
                if not torch.is_tensor(noise[0]):
                    noise_list = [torch.tensor(noise_layer, device=device).unsqueeze(0).unsqueeze(0) for noise_layer in noise]
                else:
                    noise_list = [noise_layer.unsqueeze(0).unsqueeze(0).to(device) for noise_layer in noise]
    
                codes = torch.tensor(w, device=device, dtype=torch.float32).view(12, -1)

            else:
                continue
            
        begin_time = time.time()
        loss_, w_coarse, w_net, noise = optimise_in_image_domain(image, generator, args.max_epoch, loss_type, th,
                                search_noise=not args.not_search_noise,  w_star=not args.not_w_star, n_latent=12, downsample_rate=args.downsample_rate,
                                device=device, optim_method=args.optim_method, lr=args.lr, save_interval=args.save_interval, 
                                exp_dir=dst_folder, write_image=False, train_noise_idx=args.train_noise_idx,
                                 w_coarse=codes, log_loss=False, noise_coarse=noise_list)

        w_and_noise = {'w':w_net.squeeze().cpu().numpy(),
                       'noise':[noise_.squeeze().cpu().numpy() for noise_ in noise]}
        f_save = open(os.path.join(dst_folder, 'w_and_noise.pkl'), 'wb')
        pickle.dump(w_and_noise, f_save)
        f_save.close()
        end_time = time.time()
        duration = end_time - begin_time
        print('time: {} min {} s'.format(duration//60, duration%60))
        

    

    




