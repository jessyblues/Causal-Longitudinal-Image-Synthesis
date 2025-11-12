from collections import namedtuple
import torch
import torch.nn.functional as F
from torch.nn import ReLU, Sigmoid, Sequential, Module
from torch.nn import Conv3d, BatchNorm3d, MaxPool3d, ConvTranspose3d, Upsample
from models.GAN_mri_package.dataset import MriFileFolderDataset
import pdb
import torch.nn as nn
from models.GAN_mri_package.model_3D import StyledGenerator
import sys

from ImageEnocder.models.psp import pSp

from ImageEnocder.models.encoders.my_modules import noise_encoder
import numpy as np


if __name__ == '__main__':

    from torch.utils.data import DataLoader
    import argparse
    import os


    parser = argparse.ArgumentParser(description='Generate styleGAN mri image')
    parser.add_argument('--gan_pth', default='/home1/yujiali/cf_mri_2/StyleGAN/exp_1/checkpoint/204400.model')
    parser.add_argument('--encoder_pth', default='/home1/yujiali/cf_mri_2/Encoder_GAN/e4e_exp2/checkpoints/iteration_157000.pt', type=str)
    parser.add_argument('--encoder', action='store_true', default=False)

    parser.add_argument('--code_dim', default=128)
    parser.add_argument('--n_latent', default=12)

    parser.add_argument('--img_dir', default='/home1/yujiali/dataset/brain_MRI/ADNI/T1/brain_MNI')
    parser.add_argument('--cuda_id', default='7', type=str)
    parser.add_argument('--exp_dir', default='/home1/yujiali/cf_mri_2/Encoder_GAN/e4e_exp4')
    parser.add_argument('--batch_size', default=1, type=int)


    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--max_epoch', default=10, type=int)
    parser.add_argument('--save_every', default=200, type=int)
    parser.add_argument('--print_every', default=1, type=int)



    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    device = torch.device('cuda:0')

    os.makedirs(os.path.join(args.exp_dir, 'checkpoint'), exist_ok=True)

    args.encoder_type = 'E4E_3d'
    args.checkpoint_path = args.encoder_pth
    args.stylegan_weights = args.gan_pth
    args.start_from_latent_avg = False
    args.device = device
    args.stylegan_size = 256
    args.noise_encoder = False
    args.update_param_list = []

    psp = pSp(args)
    

    for params in psp.parameters():
        params.requires_grad = False


    generator = StyledGenerator(args.code_dim, init_size=(6, 7, 6)).to(device)
    generator = nn.DataParallel(generator, device_ids=np.arange(torch.cuda.device_count()).tolist())
    
    ckpt = torch.load(args.gan_pth, map_location='cuda:0')
    generator.load_state_dict(ckpt['generator'])
    generator.eval()
    
    del ckpt

    for params in generator.parameters():
        params.requires_grad = False

    parser = argparse.ArgumentParser(description='training noise encoder')
    
    noise_encoder1 = noise_encoder(n_channels=1, n_classes=1).to(device)
    noise_encoder1 = nn.DataParallel(noise_encoder1, device_ids=np.arange(torch.cuda.device_count()).tolist())
    
    training_dataset = MriFileFolderDataset(root=args.img_dir, crop=True, crop_size=(192, 224, 192), return_pth=True, walk=True)
    #training_data_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    optimizer = torch.optim.Adam(noise_encoder1.parameters(), lr=1e-3)
    loss = nn.L1Loss().to(device)

    epoch = 0 
    step = 0
    #max_epoch = 10
    step_loss = []
    
    while epoch < args.max_epoch:
        
        training_data_loader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
        
        for batch_idx, batch in enumerate(training_data_loader):
            
            image, path = batch
            image = image.to(device)

            noise_list = list(noise_encoder1(image))
            noise_list.append(torch.randn((image.shape[0], 1, 192, 224, 192), device=device))

            
            codes = psp.encoder(image)
            rec = generator(input=[codes], noise=noise_list, step=5, input_is_latent=True)
            loss_l1 = F.l1_loss(image, rec)
            
            
            optimizer.zero_grad()
            loss_l1.backward()
            optimizer.step()

            step_loss.append(loss_l1.item())
        
            
            if (step+1) % args.print_every == 0:
                print('epoch {}, batch {}/{}, total step {}, l1 loss {}'.format(epoch, \
                            batch_idx, len(training_data_loader), step, np.mean(step_loss)))
                step_loss = []
            
            if (step+1) % args.save_every == 0:
                net_dict = {
                    'w_encoder': psp.encoder.state_dict(),
                    'noise_encoder': noise_encoder1.state_dict()
                    }
                torch.save(net_dict, os.path.join(args.exp_dir, 'checkpoint', 'stpe={}.model'.format(step)))
            
            step += 1

            
            
        
        




        #pdb.set_trace()
