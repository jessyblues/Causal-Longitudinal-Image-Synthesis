import argparse
import random
import math

import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] ="6" 

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad

from GAN_mri_package.dataset import MriFileFolderDataset
import pdb
import SimpleITK as sitk


from models.wGAN_and_VAE_GAN.Model_VAEGAN import Generator, Discriminator, Encoder

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag
        
def cnn_paras_count(net):
    """cnn参数量统计, 使用方式cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    print('{:.2f} Mb'.format(total_params*4/1024/1024))
    #total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_params*4/1024/1024

BATCH_SIZE=2
max_epoch = 100
gpu = True
workers = 4

reg = 5e-10

gamma = 20
beta = 10


#setting latent variable sizes
latent_dim = 1000

exp_dir = '/home1/yujiali/cf_mri_2/bl_methods/VAE_GAN/exp'
ckpt_dir = os.path.join(exp_dir, 'ckpt')
visual_dir = os.path.join(exp_dir, 'visual')

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(visual_dir, exist_ok=True)

if __name__ == '__main__':
    
    dataset_root = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/bl_mri'
    dataset = MriFileFolderDataset(root=dataset_root, crop=True, crop_size=(160, 192, 160), walk=True)
    
    print('sample number: {}'.format(len(dataset)))
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=workers, drop_last=True)
    
    
    G = Generator(noise = latent_dim)
    D = Discriminator()
    E = Encoder()
    
    print(cnn_paras_count(G)[1]+cnn_paras_count(D)[1]+cnn_paras_count(E)[1])
    pdb.set_trace()

    G.cuda()
    D.cuda()
    E.cuda()
    
    g_optimizer = optim.Adam(G.parameters(), lr=0.0001)
    d_optimizer = optim.Adam(D.parameters(), lr=0.0001)
    e_optimizer = optim.Adam(E.parameters(), lr = 0.0001)
    
    N_EPOCH = 100

    real_y = Variable(torch.ones((BATCH_SIZE, 1)).cuda())
    fake_y = Variable(torch.zeros((BATCH_SIZE, 1)).cuda())
    criterion_bce = nn.BCELoss()
    criterion_l1 = nn.L1Loss()
    
    for epoch in range(N_EPOCH):
        for step, real_images in enumerate(train_loader):
            _batch_size = real_images.size(0)
            real_images = Variable(real_images,requires_grad=False).cuda()
            z_rand = Variable(torch.randn((_batch_size, latent_dim)),requires_grad=False).cuda()
            #pdb.set_trace()
            ###############################################
            # Train D 
            ###############################################
            d_optimizer.zero_grad()
            
            requires_grad(E, False)
            requires_grad(G, False)
            requires_grad(D, True)
            with torch.no_grad():
                mean,logvar,code = E(real_images)
                x_rec = G(code)
                x_rand = G(z_rand)
        
            d_real_loss = criterion_bce(D(real_images),real_y[:_batch_size])
            d_recon_loss = criterion_bce(D(x_rec), fake_y[:_batch_size])
            d_fake_loss = criterion_bce(D(x_rand), fake_y[:_batch_size])
            
            dis_loss = d_recon_loss+d_real_loss + d_fake_loss
            dis_loss.backward(retain_graph=True)
            
            d_optimizer.step()
            
            ###############################################
            # Train G
            ###############################################
            
            requires_grad(E, False)
            requires_grad(G, True)
            requires_grad(D, False)
            g_optimizer.zero_grad()
            
            with torch.no_grad():
                mean,logvar,code = E(real_images)
            
            x_rec = G(code)
            x_rand = G(z_rand)
            
            output = D(real_images)
            d_real_loss = criterion_bce(output,real_y[:_batch_size])
            output = D(x_rec)
            d_recon_loss = criterion_bce(output,fake_y[:_batch_size])
            output = D(x_rand)
            d_fake_loss = criterion_bce(output,fake_y[:_batch_size])
            
            d_img_loss = d_real_loss + d_recon_loss+ d_fake_loss
            gen_img_loss = -d_img_loss
            
            rec_loss = ((x_rec - real_images)**2).mean()
            
            err_dec = gamma* rec_loss + gen_img_loss
            
            err_dec.backward()
            g_optimizer.step()
            ###############################################
            # Train E
            ###############################################
            
            requires_grad(E, True)
            requires_grad(G, False)
            requires_grad(D, False)
            
            mean,logvar,code = E(real_images)
            x_rec = G(code)
            x_rand = G(z_rand)
            
            prior_loss = 1+logvar-mean.pow(2) - logvar.exp()
            prior_loss = (-0.5*torch.sum(prior_loss))/torch.numel(mean.data)
            rec_loss = ((x_rec - real_images)**2).mean()
            err_enc = prior_loss + beta*rec_loss
            
            e_optimizer.zero_grad()
            err_enc.backward()
            e_optimizer.step()
            ###############################################
            # Visualization
            ###############################################
    
            if step % 10 == 0:
                print('[epoch {}/{} step {}/{}]'.format(epoch,N_EPOCH, step, len(train_loader)),
                    'D: {:<8.3}'.format(torch.mean(dis_loss).item()), 
                    'En: {:<8.3}'.format(torch.mean(err_enc).item()),
                    'De: {:<8.3}'.format(torch.mean(err_dec).item()) 
                    )
                
                
                
        x_real_np = real_images[0].detach().cpu().squeeze().numpy()
        x_rec_np = x_rec[0].detach().cpu().squeeze().numpy()
        x_rand_np = x_rand[0].detach().cpu().squeeze().numpy()
        
        if (epoch) % 1 == 0:
            
            sitk.WriteImage(sitk.GetImageFromArray(x_real_np), os.path.join(visual_dir, 'epoch={}_real.nii.gz'.format(epoch)))
            sitk.WriteImage(sitk.GetImageFromArray(x_rec_np), os.path.join(visual_dir, 'epoch={}_rec.nii.gz'.format(epoch)))
            sitk.WriteImage(sitk.GetImageFromArray(x_rand_np), os.path.join(visual_dir, 'epoch={}_rand.nii.gz'.format(epoch)))

            torch.save(G.state_dict(),os.path.join(ckpt_dir, 'G_VG_ep_'+str(epoch+1)+'.pth'))
            torch.save(D.state_dict(),os.path.join(ckpt_dir,'D_VG_ep_'+str(epoch+1)+'.pth'))
            torch.save(E.state_dict(),os.path.join(ckpt_dir,'E_VG_ep_'+str(epoch+1)+'.pth'))
    
    
    