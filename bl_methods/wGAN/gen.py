import argparse
import random
import math

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="0" 

import torch
from torch.backends import cudnn
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt
import copy
import SimpleITK as sitk

import sys
from models.wGAN_and_VAE_GAN.Model_VAEGAN import Generator
import ants
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize

latent_dim = 1000
import time


if __name__ == '__main__':
    

    img_number = 500
    batch_size = 1
    ckpt_path = '/home1/yujiali/cf_mri_2/bl_methods/wGAN/exp/ckpt/G_W_iter17000.pth'
    ckpt = torch.load(ckpt_path, map_location=torch.device('cuda:0'))
    finished_number = 0
    img_dir = '/home1/yujiali/cf_mri_2/bl_methods/wGAN/exp/output_img'
    
    G = Generator(noise = latent_dim)
    G.cuda()
    G.load_state_dict(ckpt)
    time_list = []

    while finished_number < img_number:
        
        num = min(batch_size, img_number - finished_number)
        z_rand = torch.randn((num, latent_dim), dtype = torch.float).cuda()
        
        
        with torch.no_grad():
            time0=time.perf_counter()
            img_3D = G(z_rand)
            img_3D = SpatialPad(spatial_size=(1, 192, 224, 192))(img_3D)

            
            img_3D = img_3D.squeeze()
            time1=time.perf_counter()
            time_list.append(time1-time0)
            finished_number+=1


        
            for j in range(num):
            
                index = finished_number+1
                img_name = img_dir+f'/3D_images/{str(index).zfill(4)}.nii.gz'

                img = img_3D[j].squeeze()
                img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
                img = img.cpu().numpy()
                back_ground = float(img[0, 0, 0])
                img = np.clip(img-back_ground, a_min=0, a_max=None)
                img = img / (1 - back_ground)


                mask = ants.get_mask(ants.from_numpy(img))
                img = img*(mask.numpy())
                    
                out = sitk.GetImageFromArray(img)
                sitk.WriteImage(out, img_name)
                finished_number += 1

            print('{} has been written!'.format(index))

    print(np.mean(time_list), np.std(time_list))
