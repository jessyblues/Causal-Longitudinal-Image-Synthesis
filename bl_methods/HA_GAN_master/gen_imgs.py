import numpy as np
import torch
import os
import json
import argparse
import pdb

from torch import nn
from .utils import trim_state_dict_name

import ants
import time


torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch HA-GAN gen')
parser.add_argument('--batch-size', default=2, type=int,
                    help='mini-batch size (default: 4), this is the total '
                         'batch size of all GPUs')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 8)')
parser.add_argument('--img-size', default=256, type=int,
                    help='size of training images (default: 256, can be 128 or 256)')

parser.add_argument('--continue-iter', default=165_000, type=int,
                    help='continue from a ckeckpoint that has run for n iteration  (0 if a new run)')
parser.add_argument('--latent-dim', default=1024, type=int,
                    help='size of the input latent variable')

parser.add_argument('--data-dir', type=str, default=None,
                    help='path to the preprocessed data folder')
parser.add_argument('--exp-name', default='HA_GAN_256', type=str,
                    help='name of the experiment')
parser.add_argument('--fold', default=1, type=int,
                    help='fold number for cross validation')
parser.add_argument('--gpu_ids', default="0", type=str,
                    help='the gpu to use')

# configs for conditional generation
parser.add_argument('--num-class', default=0, type=int,
                    help='number of class for auxiliary classifier (0 if unconditional)')
parser.add_argument('--output_dir', default=None)
parser.add_argument('--gen_number', default=500, type=int)

def main():
    # Configuration
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    print('gpu number:', torch.cuda.device_count())


    
    if args.img_size == 256:
        from bl_methods.HA_GAN_master.models.Model_HA_GAN_256 import Discriminator, Generator, Encoder, Sub_Encoder
    elif args.img_size == 128:
        from bl_methods.HA_GAN_master.models.Model_HA_GAN_128 import Discriminator, Generator, Encoder, Sub_Encoder
    else:
        #raise NotImplmentedError
        print('No such image size', args.img_size)
        exit()
        
    G = Generator(mode='eval', latent_dim=args.latent_dim, num_class=args.num_class).cuda()



    # Resume from a previous checkpoint
    if args.continue_iter != 0:
        ckpt_path = '/home1/yujiali/cf_mri_2/bl_methods/HA_GAN_master/checkpoint/'+args.exp_name+'/G_iter'+str(args.continue_iter)+'.pth'
        ckpt = torch.load(ckpt_path, map_location='cuda')
        ckpt['model'] = trim_state_dict_name(ckpt['model'])
        G.load_state_dict(ckpt['model'])
    else:
        print('please specify the ckpt!')
        
    G = nn.DataParallel(G)
    G.eval()

    time_list = []
    for i in range(args.gen_number): 
        noise = torch.randn((1, args.latent_dim)).cuda()
        time0 = time.perf_counter()
        with torch.no_grad():
            fake_images = G(noise, crop_idx=None, class_label=None)
        time1 = time.perf_counter()
        time_list.append(time1-time0)
        
        for j in range(2):
            
            fake_image = fake_images[j].squeeze()
            featmask = np.swapaxes((0.5*fake_image+0.5).data.cpu().numpy(), 0, 2)[32:-32, 16:-16, 32:-32]
            featmask = ants.from_numpy((featmask - np.min(featmask))/(np.max(featmask)-np.min(featmask)))
            featmask = ants.mask_image(featmask, ants.get_mask(featmask))
            featmask.to_file(os.path.join(args.output_dir, 'test_{:0>4d}.nii.gz'.format(i+4*j)))
            print('{}/{} finished!'.format(i, args.gen_number))
            #pdb.set_trace()
    print(np.mean(time_list), np.std(time_list))
            

if __name__ == '__main__':
    main()
