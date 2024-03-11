import torch
import os
import SimpleITK as sitk
from GAN_mri_package.dataset import MriFileFolderDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from GAN_mri_package.model_3D import StyledGenerator, Discriminator
import numpy as np
import pickle
import pdb
import ants

def get_generator_from_ckpt(gan_pth, gpu_id, eval=True, code_size=128, init_size=(6, 7, 6)):

    device = torch.device('cuda:{}'.format(gpu_id))
    generator = StyledGenerator(code_dim=code_size, init_size=init_size).to(device)
    generator = torch.nn.DataParallel(generator, device_ids=[gpu_id])
    
    ckpt = torch.load(gan_pth, map_location=device)
    generator.load_state_dict(ckpt['generator'])
    if eval:
        generator.eval()
    del ckpt
    return generator

def get_threshold(img, interval=0.01, threshold_ratio=0.275):

    if torch.is_tensor(img):
        #img_list = img.view(-1, )
        for i in np.arange(0, 1, interval):
            if torch.sum(img>i) < threshold_ratio*256*224*224:
                threshold = i
                break
        #threshold = torch.sort(img_list)[int(threshold_ratio*256*224*224)]
    
    elif type(img) == list:
        #threshold = np.sort(img_list)[int(threshold_ratio*256*224*224)]
        img = np.array(img)
        for i in np.arange(0, 1, interval):
            if np.sum(img>i) < threshold_ratio*256*224*224:
                threshold = i
                break

    else:
        for i in np.arange(0, 1, interval):
            if np.sum(img>i) < threshold_ratio*256*224*224:
                threshold = i
                break
    

    return threshold



def normalize_batch_img_3D(img_3D):

    flag = False

    if img_3D.dim() == 5:
        img_3D = img_3D[:, 0]
        flag = True
    
    if img_3D.dim() == 4:
        for i in range(img_3D.shape[0]):
            img_3D[i] = img_3D[i]-(torch.min(img_3D[i]))
            img_3D[i] = img_3D[i] / torch.max(img_3D[i])
    else:
        img_3D = img_3D - (torch.min(img_3D))
        img_3D = img_3D / torch.max(img_3D)
    
    if flag:
        img_3D = img_3D.unsqueeze(1)

    
    return img_3D


def generate_2D_slice_from_mri(mri_pth, goal_img_folder, batch_size, crop_size=None, 
                                slice=['x', 'y', 'z'], device='cuda', print_progress=False):

    #pdb.set_trace()
    if crop_size is None:   
        dataset = MriFileFolderDataset(root=mri_pth, return_latent=False, return_pth=True, walk=True)
    else:
        dataset = MriFileFolderDataset(root=mri_pth, return_latent=False, return_pth=True, walk=True, crop=True, crop_size=crop_size)
    
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    if type(slice) == str:
        slice = [slice]
    
    for slice_ in slice:
        folder = os.path.join(goal_img_folder, 'slice_{}'.format(slice_))    
        os.makedirs(folder, exist_ok=True)
        
    for idx, batch in enumerate(dataloader):
        
        image, path = batch
        image = image.to(device)

        shape = image[0].squeeze().shape
        
        if type(path) == str:
            path = [path]

        for i in range(len(path)):         
            image_i = image[i].squeeze().cpu().numpy()
            
            for slice_ in slice:

                folder = os.path.join(goal_img_folder, 'slice_{}'.format(slice_)) 


                if slice_ == 'x': 
                    slice_i = np.flip(image_i[shape[0]//2], axis=0)
                elif slice_ == 'y':
                    slice_i = image_i[::-1, 120]
                elif slice_ == 'z':
                    slice_i = image_i[::-1, :, shape[2]//2]
                
                        
                plt.figure()
                plt.imshow(slice_i, cmap=plt.get_cmap('gray'))
                path_i = path[i]
                _, name = os.path.split(path_i)
                img_idx = name.split('.nii')[0]
                image_name = '{}_{}.jpg'.format(img_idx, slice_)
                plt.axis('off')
                #pdb.set_trace()
                plt.savefig(os.path.join(folder, image_name),  bbox_inches='tight',pad_inches = 0)
                plt.close()

                if print_progress:
                    print('{}/{} , {},  finished!'.format(idx*batch_size+i, len(dataset), slice_))

def generate_2D_slice_from_gan(gan_pth, goal_img_folder, batch_size, 
                                total_number=1800, slice=['x', 'y', 'z'], gpu_id=None, print_progress=False):

    code_size = 128
    #pdb.set_trace()
    generator = get_generator_from_ckpt(gan_pth=gan_pth, gpu_id=gpu_id)
    device = torch.device('cuda:{}'.format(gpu_id)) if not gpu_id == None else torch.device('cpu')

    if goal_img_folder == None:
        epoch_ = gan_pth.split('/')[-1].split('.')[0]
        goal_img_folder = '/home1/yujiali/cf_mri_2/StyleGAN/fake_img/candidate/'+epoch_

    if type(slice) == str:
        slice = [slice]
    
    for slice_ in slice:
        folder = os.path.join(goal_img_folder, 'slice_{}'.format(slice_))    
        
        if not os.path.exists(folder):
            os.makedirs(folder)
    

    finished_number = 0
    
    while finished_number < total_number:

        num = min(batch_size, total_number - finished_number)
        #pdb.set_trace()
        random_noise =  torch.randn(num, code_size, device=device)
                
        with torch.no_grad():
        
            img_3D = generator(
                            random_noise, step=5, input_is_latent=False
                        ).data.detach()
        
            img_3D = img_3D.squeeze()
            image = normalize_batch_img_3D(img_3D)
            del img_3D

        shape = image[0].squeeze().shape
        
        
        filename = ['{:0>4}'.format(index) for index in range(finished_number, min(finished_number+batch_size, total_number))]
        assert len(filename) == num, 'something wrong!'

        for i in range(len(filename)):         
            
            #pdb.set_trace()
            image_i = image[i].squeeze().detach().cpu().numpy()
            mask = ants.get_mask(ants.from_numpy(image_i))
            image_i = image_i*mask.numpy()
           # masked_img = ants_img.new_image_like(masked_img)
            #pdb.set_trace()

            for slice_ in slice:

                folder = os.path.join(goal_img_folder, 'slice_{}'.format(slice_)) 

                if slice_ == 'x': 
                    slice_i = np.flip(image_i[shape[0]//2], axis=0)
                elif slice_ == 'y':
                    slice_i = image_i[::-1, 120]
                elif slice_ == 'z':
                    slice_i = image_i[::-1, :, shape[2]//2]
                
                slice_i = (np.clip(slice_i.copy(), a_min=0.2, a_max=0.9)-0.2)/0.7

                        
                plt.figure()
                plt.imshow(slice_i, cmap=plt.get_cmap('gray'))
                image_name = '{}_{}.jpg'.format(filename[i], slice_)
                plt.axis('off')
                plt.savefig(os.path.join(folder, image_name),  bbox_inches='tight',pad_inches = 0)
                plt.close()

                if print_progress:
                    print('{}/{} , {}, finished!'.format(finished_number+i, total_number, slice_))

            del mask
            del image_i
        finished_number += num
    
def reconstruct_image(w, nii_save_pth=None, generator=None, device='cuda', noise=None, threshold=True):
    
    if not torch.is_tensor(w):
        w = torch.tensor(w)
    
    w = w.to(device=device)

    while w.dim() < 3:
        w = w.unsqueeze(0)

    generator.to(device=device)

    #pdb.set_trace()

    if not (noise is None):
        
        if not torch.is_tensor(noise[0]):
            noise = [torch.tensor(noise_, device=device) for noise_ in noise]
        else:
            noise = [noise_.to(device=device) for noise_ in noise]
        
        while (noise[0].dim()<5):
            noise = [noise_.unsqueeze(0) for noise_ in noise]


    if type(nii_save_pth)==str:
        nii_save_pth = [nii_save_pth]
    #pdb.set_trace()
    
    #assert w.shape[0] == len(nii_save_pth), 'no coherent!'
    
    with torch.no_grad():

        #print('correct w:', w)
        #print('correct noise:', noise)
        
        if noise is None:
            img_3D = generator(w, step=5, input_is_latent=True).detach()
        else:
            img_3D = generator(w, step=5, input_is_latent=True, noise=noise).detach()
        
        #adv = discriminator(img_3D, step=5)
        #img_3D = img_3D.squeeze(1)
    img_3D = normalize_batch_img_3D(img_3D)
        
    for i in range(w.shape[0]):

      

        img = img_3D[i].squeeze().cpu().numpy().astype('float32')

        
        if threshold:
            th = get_threshold(img)
            img = img*(img>th)
            
        if not nii_save_pth is None:
            out = sitk.GetImageFromArray(img)
            sitk.WriteImage(out, nii_save_pth[i])
            
    return w, img_3D.squeeze(1).cpu().numpy().astype('float32')

def reconstruct_image_from_pkl_file(pkl_file, nii_save_pth=None, generator=None, device='cuda', threshold=True):

    if os.path.exists(pkl_file):
        f_read = open(pkl_file, 'rb+')
        dict2 = pickle.load(f_read)
        w = dict2['w']
        noise  = dict2['noise']
    else:
        print('no such pickle file! {}'.format(pkl_file))
    
    return reconstruct_image(w=w, nii_save_pth=nii_save_pth, generator=generator, device=device, noise=noise, threshold=threshold)

def denoise_mask(ants_img):

    mask = ants.get_mask(ants_img)

    masked_img = ants_img.numpy()*mask.numpy()
    masked_img = ants_img.new_image_like(masked_img)

    return masked_img

def generate_slice_collection(generator, device, output_dir):
    

    images = []

    gen_i, gen_j = (4, 2)
    generator.eval()
    code_size=128

    fig = plt.figure()
    with torch.no_grad():
        for _ in range(gen_i):
            img_3D = generator(
                            torch.randn(gen_j, code_size).to(device), step=5, alpha=0
                        ).data.cpu()
                    #pdb.set_trace()
            img_3D = normalize_batch_img_3D(img_3D)
            #pdb.set_trace()
            images.append(img_3D[:, 0, img_3D.shape[2]//2])
            #pdb.set_trace()
            
    images = torch.cat(images, 0)
            #images = normalize_batch_img(images)
            
    for img_idx in range(images.shape[0]):
        ax1 = fig.add_subplot(gen_j, gen_i, img_idx+1)
        ax1.imshow(images[img_idx].numpy().T, cmap=plt.get_cmap('gray'))

    fig.savefig(output_dir+'/sample.jpg')
    plt.close(fig)




if __name__ == '__main__':
    
    device = torch.device('cuda')
    generator_ = get_generator_from_ckpt(gan_pth='/home1/yujiali/GAN_mri/exp/exp1/checkpoint/296000.model', device=device)
    
    reconstruct_image_from_pkl_file(pkl_file='/home1/yujiali/GAN_mri/exp/exp9_finetune/real_mri/I115093_mni/w_and_noise.pkl', 
    nii_save_pth='./test.nii.gz', generator=generator_, device=device, threshold=False)




