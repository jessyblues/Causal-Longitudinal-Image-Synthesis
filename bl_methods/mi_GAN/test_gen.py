import argparse
import random
import math

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad


import os
from torch.backends import cudnn
import pdb
import matplotlib.pyplot as plt
import SimpleITK as sitk


import sys
from bl_methods.mi_GAN.data_utils import patch_paired_inertval_dataset, get_paired_patches_dataset
from bl_methods.mi_GAN.model import UNet3D, discriminator3D
from torch.utils.data import DataLoader
import ants

def main_work(args):
    device = torch.device('cuda:0')
    dataset = get_paired_patches_dataset(dataset_type=args.dataset, interval=args.interval, is_train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchSize, shuffle=False)
    
    print('sample numbers:{}, patches number:{}'.format(
        len(dataset.image_paths1), len(dataset)))
    
    unet = UNet3D(in_channels=1, out_channels=1, info_dim=args.info_dim)
    unet.to(device)
    unet = nn.DataParallel(unet, device_ids=[0])
    unet.eval()
    
    ckpt = torch.load(args.ckpt_path, map_location=device)
    unet.load_state_dict(ckpt['unet'])
    
    single_slices = 0
    rec_imgs_list = []
    dst_patch_list = []
    
    for idx, data in enumerate(dataloader):
            
        path1, path2, info, patches1, patches2 = data
        patches1, patches2, info = patches1.to(device), patches2.to(device), info.to(device)
        
        with torch.no_grad():
            rec_imgs = unet(x=patches1, mi=info)
        
        path1 = path1[0]
        path2 = path2[0]
        
        subject = path1.split('/')[-3]
        date1 = path1.split('/')[-2]
        date2 = path2.split('/')[-2]
        single_slices += patches1.shape[0]
        rec_imgs_list.append(rec_imgs.squeeze())
        dst_patch_list.append(patches2.squeeze())
        
        if single_slices == 80:
            
            imgs_flag = np.ones((160, 192, 160))
            imgs_ = np.zeros((160, 192, 160))
            rec_imgs = torch.cat(rec_imgs_list, dim=0).squeeze()
            dst_patchs = torch.cat(dst_patch_list, dim=0).squeeze()
            
            for patch_idx in range(rec_imgs.shape[0]):
                
                patch = rec_imgs[patch_idx].cpu().numpy().squeeze()
                dst_patch = dst_patchs[patch_idx].cpu().numpy().squeeze()
                patch = patch / np.sum(patch) * np.sum(dst_patch)
                
                i = patch_idx // 20
                j = (patch_idx - i*20) // 4
                k = patch_idx % 4
                        
                i_1 = i*32
                j_1 = j*32
                k_1 = k*32
                
                #pdb.set_trace()
                try:
                    imgs_[i_1:i_1+64, j_1:j_1+64, k_1:k_1+64] += patch
                    imgs_flag[i_1:i_1+64, j_1:j_1+64, k_1:k_1+64] += 1
                except Exception as e:
                    pdb.set_trace()
        
            imgs_ = imgs_ / imgs_flag
            mask = ants.get_mask(ants.from_numpy(imgs_)).numpy().astype('float')
            imgs_ = imgs_ * mask
            
            single_folder = os.path.join(args.output_dir_path, subject, date2)
            os.makedirs(single_folder, exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(imgs_), os.path.join(single_folder,
                                                                        'rec.nii.gz'))
            print(single_folder)
                #pdb.set_trace()
            
            rec_imgs_list = []
            dst_patch_list = []
            single_slices = 0

        

            

   

def main():
        
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    ## dataset setting

    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--interval', default=4,type=int)
    parser.add_argument('--dataset', default='ADNI',type=str)
    parser.add_argument('--batchSize', default=10,type=int)
    parser.add_argument('--info_dim', default=4,type=int)
    
    ## learning setting

    parser.add_argument(
        '--ckpt_path', default='/home1/yujiali/cf_mri_2/bl_methods/mi_GAN/interval=1year/ckpt/iter=376200.model', 
        type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--output_dir_path', default='/home1/yujiali/cf_mri_2/bl_methods/mi_GAN/predict_imgs', type=str
    )
    ## DP setting
    parser.add_argument('--gpu_id', type=str, default='6')


    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    args.output_dir_path = os.path.join(args.output_dir_path, 
                                        args.dataset,
                                        'interval={}year'.format(args.interval))
    os.makedirs(args.output_dir_path, exist_ok=True)
    main_work(args)


if __name__ == '__main__':
    
    main()
