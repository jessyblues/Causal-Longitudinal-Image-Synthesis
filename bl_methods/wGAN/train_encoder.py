import torch
import pdb
import torch
from torch import nn
from functools import partial

import torch.nn.functional as F
from torch.autograd import Variable

from models.ResNet import resnet50
import os
from models.wGAN_and_VAE_GAN.Model_VAEGAN import Generator
import argparse
from models.GAN_mri_package.dataset import MriFileFolderDataset
import numpy as np
import SimpleITK as sitk

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



def main_work(args):
    
    encoder = resnet50().cuda()
    encoder.add_linear(output_channel=args.latent_dim)
    encoder = encoder.cuda()

    G = Generator(noise=args.latent_dim).cuda()
    ckpt=torch.load(args.ckpt, map_location='cuda')
    G.load_state_dict(ckpt)
    G.eval()
    
    gpu_number = torch.cuda.device_count()
    
    encoder = nn.DataParallel(encoder, device_ids=np.arange(gpu_number).tolist())
    G = nn.DataParallel(G, device_ids=np.arange(gpu_number).tolist())
    
    requires_grad(G, False)
    
    dataset = MriFileFolderDataset(root=args.img_dir, crop=True, crop_size=(160, 192, 160), walk=True)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,  
                                                    batch_size=args.batchSize, 
                                                    shuffle = False,
                                                    drop_last=True,
                                                    num_workers=1)
    
    
    optimiser = torch.optim.Adam(params=encoder.parameters(), lr=1e-3)
    
    for epoch in range(args.begin_epoch_num, args.total_epoch):
        
        for idx, image in enumerate(dataloader):
            
            image = image.to(torch.device('cuda:0'))
            encoder_latent = encoder(image)
            encoder_latent = encoder.module.linear(encoder_latent)
            #pdb.set_trace()
            
            
            rec_image = G(encoder_latent)
            image_l1_loss = F.l1_loss(image, rec_image)
            
            optimiser.zero_grad()
            image_l1_loss.backward()
            optimiser.step()

            if idx% 10 == 0:
                print('epoch {}/{} batch {}/{} loss {:.5f}'.format(epoch, args.total_epoch, idx, len(dataloader), image_l1_loss.item()))
            
            if epoch % args.sample_every == 0 and idx == 0:
            
                ori_image_np = image[0].detach().squeeze().cpu().numpy()
                rec_image_np = rec_image[0].detach().squeeze().cpu().numpy()
                
                sitk.WriteImage(sitk.GetImageFromArray(ori_image_np), os.path.join(args.exp_dir, 'visualise', 'epoch={}_ori.nii.gz'.format(epoch)))
                sitk.WriteImage(sitk.GetImageFromArray(rec_image_np), os.path.join(args.exp_dir, 'visualise', 'epoch={}_rec.nii.gz'.format(epoch)))
            


        if epoch % args.save_every == 0:
            
            state_dict = encoder.state_dict()
            torch.save({'encoder':state_dict}, os.path.join(args.exp_dir, 'ckpt', 'epoch={}.model'.format(epoch)))
        
    


def main():
        
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    ## dataset setting

    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--begin_epoch_num', default=0,type=int)
    parser.add_argument('--interval', default=2,type=int)
    parser.add_argument('--img_dir', default='/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI',type=str)
    parser.add_argument('--batchSize', default=8,type=int)
    parser.add_argument('--info_dim', default=4,type=int)
    
    ## learning setting

    parser.add_argument(
        '--ckpt', default='/home1/yujiali/cf_mri_2/bl_methods/wGAN/exp/ckpt/G_W_iter17500.pth', 
        type=str, help='load from previous checkpoints'
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', default=True, help='use lr scheduling')
    parser.add_argument('--total_epoch', default=100, help='total batch number for all phases')


    ## save and sample setting
    parser.add_argument('--sample_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--exp_dir', type=str, help='exp dir', default='/home1/yujiali/cf_mri_2/bl_methods/wGAN/encoder_exp')

    
    ## DP setting
    parser.add_argument('--gpu_id', type=str, default='5, 6')
    parser.add_argument('--latent-dim', default=1000, type=int,
                    help='size of the input latent variable')


    args = parser.parse_args()
    #pdb.set_trace()
    args.exp_dir = os.path.join(args.exp_dir)
    #args.ckpt_path = os.path.join(args.exp_dir, 'ckpt')
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(os.path.join(args.exp_dir, 'ckpt'), exist_ok=True)
    os.makedirs(os.path.join(args.exp_dir, 'visualise'), exist_ok=True)
    main_work(args)


if __name__ == '__main__':
    
    main()
