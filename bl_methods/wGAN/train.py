import argparse
import random
import math

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="5" 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad

from models.GAN_mri_package.dataset import MriFileFolderDataset
from torch.backends import cudnn
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt
import copy
import SimpleITK as sitk


from models.wGAN_and_VAE_GAN.Model_WGAN import Generator, Discriminator

def inf_train_gen(data_loader):
    while True:
        for _,images in enumerate(data_loader):
            yield images
            
def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    print('{:.2f} Mb'.format(total_params*4/1024/1024))
    #total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_params*4/1024/1024
 


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA):    
    alpha = torch.rand(real_data.size(0),1,1,1,1)
    alpha = alpha.expand(real_data.size())
    
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

BATCH_SIZE=4
gpu = True
workers = 4

reg = 5e-10

gamma = 20
beta = 10


#setting latent variable sizes
latent_dim = 1000

exp_dir = '/home1/yujiali/cf_mri_2/bl_methods/wGAN/exp'
ckpt_dir = os.path.join(exp_dir, 'ckpt')
visual_dir = os.path.join(exp_dir, 'visual')

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

if __name__ == '__main__':
    
    dataset_root = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/bl_mri'
    dataset = MriFileFolderDataset(root=dataset_root, crop=True, crop_size=(160, 192, 160), walk=True)
    
    print('sample number: {}'.format(len(dataset)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=workers)
    
    
    D = Discriminator()
    G = Generator(noise = latent_dim)

    #G.cuda()
    #D.cuda()
    
    D_param_num, D_param_size = cnn_paras_count(D)
    G_param_num, G_param_size = cnn_paras_count(G)
    
    print(D_param_size+G_param_size)
    
    pdb.set_trace()
    
    g_optimizer = optim.Adam(G.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001)
    
    N_EPOCH = 100
    
    real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
    fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
    loss_f = nn.BCELoss()

    d_real_losses = list()
    d_fake_losses = list()
    d_losses = list()
    g_losses = list()
    divergences = list()
    
    
    TOTAL_ITER = 200000
    gen_load = inf_train_gen(train_loader)
    for iteration in range(TOTAL_ITER):
    ###############################################
    # Train D 
    ###############################################
        for p in D.parameters():  
            p.requires_grad = True 
        for p in G.parameters():  
            p.requires_grad = False 

        real_images = gen_load.__next__()
        D.zero_grad()
        
        real_images = Variable(real_images).cuda()
        _batch_size = real_images.size(0)

        y_real_pred = D(real_images)
        d_real_loss = y_real_pred.mean()
        
        noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1, 1)),volatile=True).cuda()
        with torch.no_grad():
            fake_images = G(noise)
        y_fake_pred = D(fake_images.detach())

        d_fake_loss = y_fake_pred.mean()

        gradient_penalty = calc_gradient_penalty(D,real_images.data, fake_images.data, 10)
    
        d_loss = - d_real_loss + d_fake_loss +gradient_penalty
        d_loss.backward()
        Wasserstein_D = d_real_loss - d_fake_loss

        d_optimizer.step()

        ###############################################
        # Train G 
        ###############################################
        for p in D.parameters():
            p.requires_grad = False
        for p in G.parameters():
            p.requires_grad = True
            
        for iters in range(5):
            G.zero_grad()
            noise = Variable(torch.randn((_batch_size, latent_dim, 1, 1 ,1)).cuda())
            fake_image =G(noise)
            y_fake_g = D(fake_image)

            g_loss = -y_fake_g.mean()

            g_loss.backward()
            g_optimizer.step()

    ###############################################
    # Visualization
    ###############################################
        if iteration%10 == 0:
            d_real_losses.append(torch.mean(d_real_loss).item())
            d_fake_losses.append(torch.mean(d_fake_loss).item())
            d_losses.append(torch.mean(d_loss).item())
            g_losses.append(torch.mean(g_loss).item())

            print('[{}/{}]'.format(iteration,TOTAL_ITER),
                'D: {:<8.3}'.format(d_losses[-1]), 
                'D_real: {:<8.3}'.format(d_real_losses[-1]),
                'D_fake: {:<8.3}'.format(d_fake_losses[-1]), 
                'G: {:<8.3}'.format(g_losses[-1]))


            
        if (iteration+1)%500 ==0:
            torch.save(G.state_dict(),os.path.join(ckpt_dir, 'G_W_iter'+str(iteration+1)+'.pth'))
            torch.save(D.state_dict(),os.path.join(ckpt_dir, 'D_W_iter'+str(iteration+1)+'.pth'))
        
        if (iteration+1)%20 ==0:        
            x_fake_np = fake_image[0].detach().cpu().squeeze().numpy()
            sitk.WriteImage(sitk.GetImageFromArray(x_fake_np), 
                            os.path.join(visual_dir, 'fake_iters={}.nii.gz'.format(iteration)))
            

    
    
    