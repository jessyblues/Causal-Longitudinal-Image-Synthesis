import argparse

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad


import os
import SimpleITK as sitk



from bl_methods.mi_GAN.data_utils import patch_paired_inertval_dataset, get_paired_patches_dataset
from bl_methods.mi_GAN.model import UNet3D, discriminator3D
from torch.utils.data import DataLoader

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def compute_diff(a):
    
    all_a = a*6
    diff_a = torch.zeros_like(a, device=a.device)
    
    diff_a[1:, :, :] = a[:-1, :, :]
    all_a = all_a - diff_a
    
    diff_a = torch.zeros_like(a, device=a.device)
    diff_a[:-1, :, :] = a[1:, :, :]
    all_a = all_a - diff_a
    
    diff_a[:, 1:, :] = a[:, :-1, :]
    all_a = all_a - diff_a
    
    diff_a = torch.zeros_like(a, device=a.device)
    diff_a[:, :-1, :] = a[:, 1:, :]
    all_a = all_a - diff_a
    
    diff_a = torch.zeros_like(a, device=a.device)
    diff_a[:, :, :-1] = a[:, :, 1:]
    all_a = all_a - diff_a
    
    diff_a = torch.zeros_like(a, device=a.device)
    diff_a[:, :, :-1] = a[:, :, 1:]
    all_a = all_a - diff_a
    
    
    
    
    
    
    

def main_work(args):
    
    device = torch.device('cuda:0')
    gpu_number = torch.cuda.device_count()
    
    
    dataset = get_paired_patches_dataset(dataset_type=args.dataset, interval=args.interval)
    dataloader = DataLoader(dataset=dataset, batch_size=args.batchSize, shuffle=True)
    print('sample numbers:{}, patches number:{}'.format(
        len(dataset.image_paths1), len(dataset)))
    
    begin_iter = args.begin_batch_num
    iters = begin_iter
    
    unet = UNet3D(in_channels=1, out_channels=1, info_dim=args.info_dim)
    discriminator = discriminator3D(in_channels=1)
    
    unet.to(device)
    discriminator.to(device)
    
    
    unet = nn.DataParallel(unet, device_ids=np.arange(gpu_number).tolist())
    discriminator = nn.DataParallel(discriminator, device_ids=np.arange(gpu_number).tolist())
    

    
    g_optimizer = optim.Adam(
        unet.parameters(), lr=args.lr, betas=(0.9, 0.99)
        )
    d_optimizer = optim.Adam(
        discriminator.parameters(), lr=args.lr, betas=(0.9, 0.99)
        )

    if args.ckpt_path is not None:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        unet.load_state_dict(ckpt['unet'])
        discriminator.load_state_dict(ckpt['discriminator'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
    
    for i in range(args.total_epoch):
        for idx, data in enumerate(dataloader):
            
            path1, path2, info, patches1, patches2 = data
            patches1, patches2, info = patches1.to(device), patches2.to(device), info.to(device)
            rec_imgs = unet(x=patches1, mi=info)
            
            requires_grad(unet, False)
            requires_grad(discriminator, True)
            
            ## train discriminator
            fake_predict = discriminator(rec_imgs.detach())
            real_predict = discriminator(patches2)
            
            fake_predict = fake_predict.mean()
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            
            d_optimizer.zero_grad()
            fake_predict.backward()
            (-real_predict).backward()
            d_optimizer.step()
            
            ## train generator
            requires_grad(unet, True)
            requires_grad(discriminator, False)
            
            rec_imgs = unet(x=patches1, mi=info)
            fake_predict = discriminator(rec_imgs)
            
            rec_loss = F.mse_loss(rec_imgs, patches2)
            fftn1 = torch.fft.fftn(rec_imgs)
            fftn2 = torch.fft.fftn(patches2)
            
            dis = abs(fftn1 - fftn2)**2
            fft_loss = F.l1_loss(dis, torch.zeros_like(dis, dtype=torch.float, device=device))
            
            grad_loss = F.mse_loss(fftn1.real, fftn2.real)
            
           # pdb.set_trace()
            g_loss = -fake_predict.mean() + 300 * (rec_loss)+ 0.0005*fft_loss + 0.0005 * (grad_loss)
            
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            
            
            if iters % args.save_every == 0:
                save_dict = {
                    'unet':unet.state_dict(),
                    'discriminator':discriminator.state_dict(), 
                    'g_optimizer':g_optimizer.state_dict(),
                    'd_optimizer':d_optimizer.state_dict()
                }
                save_path = os.path.join(args.exp_dir, 'ckpt/iter={}.model'.format(iters))
                torch.save(save_dict, save_path)
                
            if iters % args.print_every == 0:
                print('iter:{} fake predict:{} real predict:{} g loss:{} rec mse loss:{}'.format(
                    iters, fake_predict.mean().item(), real_predict.mean().item(), g_loss.item(), rec_loss.item()
                ))
            
            if iters % args.sample_every == 0:
                
                patches1 = patches1[0].cpu().numpy().squeeze()
                patches2 = patches2[0].cpu().numpy().squeeze()
                rec_imgs = rec_imgs[0].detach().cpu().numpy().squeeze()
                
                sitk.WriteImage(sitk.GetImageFromArray(patches1),
                                os.path.join(args.exp_dir, 'visualise/iter={}_bl.nii.gz'.format(iters)))
                sitk.WriteImage(sitk.GetImageFromArray(patches2),
                                os.path.join(args.exp_dir, 'visualise/iter={}_target.nii.gz'.format(iters)))
                sitk.WriteImage(sitk.GetImageFromArray(rec_imgs),
                                os.path.join(args.exp_dir, 'visualise/iter={}_rec.nii.gz'.format(iters)))
            
            iters+=args.batchSize
    

def main():
        
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    ## dataset setting

    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--begin_batch_num', default=0,type=int)
    parser.add_argument('--interval', default=2,type=int)
    parser.add_argument('--dataset', default='ADNI',type=str)
    parser.add_argument('--batchSize', default=72,type=int)
    parser.add_argument('--info_dim', default=4,type=int)
    
    ## learning setting

    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', default=True, help='use lr scheduling')
    parser.add_argument('--total_epoch', default=100, help='total batch number for all phases')


    ## save and sample setting
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--exp_dir', type=str, help='exp dir', default='/home1/yujiali/cf_mri_2/bl_methods/mi_GAN')
    parser.add_argument('--ckpt_path', type=str, help='ckpt dir', default=None)
    
    ## DP setting
    parser.add_argument('--gpu_id', type=str, default='0')


    args = parser.parse_args()
    #pdb.set_trace()
    args.exp_dir = os.path.join(args.exp_dir, 'interval={}year'.format(args.interval))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(os.path.join(args.exp_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'visualise'), exist_ok=True)
    main_work(args)


if __name__ == '__main__':
    
    main()