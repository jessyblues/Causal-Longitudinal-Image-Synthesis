## use traditioanl gan to produce image first and then train the w(style vector) to predict the attribute(smiling)

import argparse
import random
import math

import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad

from models.GAN_mri_package.model_3D import StyledGenerator, Discriminator, MinibatchDiscrimination
import os
from models.GAN_mri_package.dataset import MriFileFolderDataset
from torch.backends import cudnn
import torch.distributed as dist
import pdb
import matplotlib.pyplot as plt
import copy
import SimpleITK as sitk


def cnn_paras_count(net):
    """cnn_paras_count(net)"""
    # Find total parameters and trainable parameters
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    print('{:.2f} Mb'.format(total_params*4/1024/1024))
    #total_trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    #print(f'{total_trainable_params:,} training parameters.')
    return total_params, total_params*4/1024/1024



def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(dataset, batch_size, resolution=8, use_DDP=True, num_workers=4):
    dataset.resolution = resolution
    if use_DDP:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        loader = torch.utils.data.DataLoader(dataset=dataset,  
                                                    batch_size=batch_size, 
                                                    sampler=train_sampler,
                                                    drop_last=True,
                                                    num_workers=num_workers)
    else:
        loader = torch.utils.data.DataLoader(dataset=dataset,  
                                                    batch_size=batch_size, 
                                                    shuffle = True,
                                                    drop_last=True,
                                                    num_workers=num_workers)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult


def main_work(args):
    
    exp_logdir = args.root_dir+'/exp_{}'.format(args.version) ## experiments dir, save ckpt, samples, training logs
    
    if not os.path.exists(exp_logdir):
        os.makedirs(os.path.join(exp_logdir, 'sample'))
        os.makedirs(os.path.join(exp_logdir, 'checkpoint'))

    cudnn.benchmark = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    
    gpu_number = torch.cuda.device_count()
    
    ## set model and DP
    
    code_size = args.style_dim
    device = torch.device("cuda:0")

    generator = StyledGenerator(code_size, init_size=(6, 7, 6))
    discriminator = Discriminator(from_mri_activate=not args.from_mri_activate, init_size=(6, 7, 6), set_mini_batch_discriminator=False)
    
    print(cnn_paras_count(generator.generator)[1]+cnn_paras_count(generator.style)[1]+cnn_paras_count(discriminator)[1])
    pdb.set_trace()
    generator = nn.DataParallel(generator, device_ids=np.arange(gpu_number).tolist())
    discriminator = nn.DataParallel(discriminator, device_ids=np.arange(gpu_number).tolist())

    if args.use_mini_batch_discriminator:
        mini_batch_discriminator = MinibatchDiscrimination(in_features=256, out_features=128, kernel_dims=3)
        mini_batch_discriminator = nn.DataParallel(mini_batch_discriminator, device_ids=np.arange(gpu_number).tolist())
    else:
        mini_batch_discriminator = None


    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
        )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
        )
    d_optimizer = optim.Adam(discriminator.module.parameters(), lr=args.lr, betas=(0.0, 0.99))
    
    if mini_batch_discriminator is not None:
        d_optimizer.add_param_group(
        {
            'params': mini_batch_discriminator.parameters(),
            'lr': args.lr,
            'betas': (0.0, 0.99)
        }
        )


    ## load ckpt

    if args.ckpt is not None:
        
        ckpt = torch.load(args.ckpt, map_location=device)
        #pdb.set_trace()
        if 'g_optimizer' in ckpt.keys():

            generator.load_state_dict(ckpt['generator'])
            discriminator.load_state_dict(ckpt['discriminator'])

            g_optimizer.load_state_dict(ckpt['g_optimizer'])
            d_optimizer.load_state_dict(ckpt['d_optimizer'])

        else:
            generator.load_state_dict(ckpt['generator'])
            discriminator.load_state_dict(ckpt['discriminator'])
        
        #pdb.set_trace()
        if mini_batch_discriminator is not None:
            if 'mini_batch_discriminator' in ckpt.keys():
                mini_batch_discriminator.load_state_dict(ckpt['mini_batch_discriminator'])
            #else:
            #    mini_batch_discriminator.load_state_dict(ckpt['discriminator'], strict=False)
        
        del ckpt


    ## sched setting
    if args.sched:
        args.lr = {64:0.001, 128: 0.0015, 256: 0.002}
        args.batch = {16: gpu_number*256, 32: gpu_number*128, 64: gpu_number*32, 128: gpu_number*8, 256: gpu_number*3}
        args.num_workers = {16:8, 32:4, 64:2, 128:1, 256:1}

    else:
        args.lr = {}
        args.batch = {}
    args.gen_sample = {16: (8, 4), 32: (8, 4), 64:(8, 4), 128:(8,4), 256:(6, 6)}
    args.batch_default = 8
    
    ## set initial step and resolution, and corresponding lr, dataloader

    step = int(math.log2(args.init_size)) - 3
    resolution = 8 * 2 ** step
    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    ## dataset setting
    dataset = MriFileFolderDataset(root=os.listdir(args.root_dir), crop=True, crop_size=tuple(args.crop_size), walk=True)
    print('sample number: {}'.format(len(dataset)))
    loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution, use_DDP=False, num_workers=args.num_workers.get(resolution, 4)
            )
    
    data_loader = iter(loader)
    requires_grad(generator, False)
    requires_grad(discriminator, True)
    if mini_batch_discriminator is not None:
        requires_grad(mini_batch_discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = args.used_sample

    max_step = int(math.log2(args.max_size)) - 3
    final_progress = True
    
    print('args:')
    print(args)


    for i in range(args.begin_batch_num, args.total_batch_number):
        discriminator.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        ## end one phase and begin next
        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 8 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution, use_DDP=False, num_workers=args.num_workers.get(resolution, 4)
            )
            data_loader = iter(loader)

            save_dict = {'generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict()} if mini_batch_discriminator is None else \
                    {'generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'mini_batch_discriminator':mini_batch_discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict()} 
            
            torch.save(
                save_dict,
                exp_logdir+f'/checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

        try:
            real_image = next(data_loader)


        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image = next(data_loader)
        
        ## used_sample added! 
        used_sample += real_image.shape[0]


        b_size = real_image.size(0)
        real_image = real_image.to(device)
        #pdb.set_trace()
        ## train discriminator
        ## real image for discriminator training


        if args.loss == 'wgan-gp':
            

            real_predict, feature = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            #pdb.set_trace()
            if mini_batch_discriminator is not None:
                real_mini_batch_loss = 0.01*mini_batch_discriminator(feature).mean()
                (-real_predict-real_mini_batch_loss).backward()
            else:
                (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
        
        ## fake image for discriminator training
        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device=device
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device=device).chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)


        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict, feature = discriminator(fake_image, step=step, alpha=alpha)


        if args.loss == 'wgan-gp':

            fake_predict = fake_predict.mean()
            if mini_batch_discriminator is not None:
                fake_mini_batch_loss = 0.01*mini_batch_discriminator(feature).mean()
                (fake_predict+fake_mini_batch_loss).backward()
            else:
                fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1, 1).to(device)
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            #pdb.set_trace()
            hat_predict, feature = discriminator(x_hat, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            
            if i%10 == 0:
                grad_loss_val = grad_penalty.item()
                disc_loss_val = (-real_predict + fake_predict).item()
                disc_mini_batch_loss_val = 0 if mini_batch_discriminator is None else (-real_mini_batch_loss+fake_mini_batch_loss).item()

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            if i%10 == 0:
                disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()



        ## train generator
        if (i + 1) % args.n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            if mini_batch_discriminator is not None:
                requires_grad(mini_batch_discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)

            predict, feature = discriminator(fake_image, step=step, alpha=alpha)
            

            mini_batch_loss = -0.01*mini_batch_discriminator(feature).mean() \
                if mini_batch_discriminator is not None else torch.tensor(0)

            mini_batch_loss_val = mini_batch_loss.item()


            if args.loss == 'wgan-gp':
                loss = -predict.mean()
                loss_ = mini_batch_loss+loss

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            if i%10 == 0:
                gen_loss_val = loss.item()

            loss_.backward()
            g_optimizer.step()
            #accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)
            if mini_batch_discriminator is not None:
                requires_grad(mini_batch_discriminator, True)


        ## sample 
        if (i + 1) % args.sample_every == 0 or i == args.begin_batch_num:
            images = []

            gen_i, gen_j = args.gen_sample.get(resolution, (10, 5))
            #generator_eval = copy.deepcopy(generator.module).cpu()
            generator.eval()


            #random_noise =  torch.randn(8, code_size).to(device)
            img_3D = generator(
                            torch.randn(6, code_size).to(device), step=step, alpha=alpha
                        ).data.cpu().squeeze()
            
            for j in range(img_3D.shape[0]):
            
                img_path = exp_logdir+f'/sample_3d/{i+1}/{j}.nii.gz'
                dir_, img_name = os.path.split(img_path)
                os.makedirs(dir_, exist_ok=True)
                

                #img = torch.clamp(img_3D[j].squeeze(), min=0, max=1)
                img = img_3D[j].squeeze()
                img = (img - torch.min(img)) / (torch.max(img) - torch.min(img))
                
                img = img.cpu().numpy()
                back_ground = float(img[0, 0, 0])
                img = np.clip(img-back_ground, a_min=0, a_max=None)
                img = img / (1 - back_ground)
                    
                out = sitk.GetImageFromArray(img)
                sitk.WriteImage(out, img_path)

            

            fig = plt.figure()
            with torch.no_grad():
                for _ in range(gen_i):
                #pdb.set_trace()
                    
                    img_3D = generator(
                            torch.randn(gen_j, code_size), step=step, alpha=alpha
                        ).data.cpu()
                    #pdb.set_trace()
                    #img_3D = normalize_batch_img_3D(img_3D)
                    images.append(img_3D[:, 0, img_3D.shape[2]//2])
            #pdb.set_trace()
            
            images = torch.cat(images, 0)

            for img_idx in range(images.shape[0]):
                ax1 = fig.add_subplot(gen_j, gen_i, img_idx+1)
                ax1.imshow(np.flip(images[img_idx].numpy(), axis=0), cmap=plt.get_cmap('gray'))

            fig.savefig(exp_logdir+f'/sample/{str(i + 1).zfill(6)}.jpg')
            plt.close(fig)

            generator.train()


        
        if (i + 1) % args.save_every == 0:
            
            save_dict = {'generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict()} if mini_batch_discriminator is None else \
                    {'generator':generator.state_dict(),
                    'discriminator':discriminator.state_dict(),
                    'mini_batch_discriminator':mini_batch_discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict()} 
            torch.save(save_dict, 
                exp_logdir+f'/checkpoint/{str(i + 1).zfill(6)}.model'
            )

        if (i + 1) % args.print_every == 0:
            print('step', step, used_sample, 'samples out of', args.phase * 2, 
            'grad_loss: ', round(grad_loss_val, 4), 'disc_loss:', round(disc_loss_val, 4), 'mini_disc_loss:', round(disc_mini_batch_loss_val, 4),
            'gen_loss:', round(gen_loss_val, 4), 'mini_gen_loss:', round(mini_batch_loss_val, 4), 
            'batch numebr: ', i+1,
            'alpha: ', alpha)

def main():
        
    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    ## dataset setting
    parser.add_argument('--init_size', default=16, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument('--img_size', default=256, type=int, help='standard image size')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')
    parser.add_argument('--begin_batch_num', default=0,type=int)
    parser.add_argument('--used_sample', default=0, type=int)
    
    ## learning setting
    parser.add_argument(
        '--phase',
        type=int,
        default=1000_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', default=True, help='use lr scheduling')
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )
    parser.add_argument('--total_batch_number', default=3_000_000, help='total batch number for all phases')
    parser.add_argument('--n_critic', type=int, default=1, help='the period of generator training')


    ## model setting 
    parser.add_argument('--from_mri_activate', default=True, help='use activate in from_rgb (original implementation)')
    parser.add_argument('--mixing', default=True, help='use mixing regularization')
    parser.add_argument('--style_dim', type=int, default=128,
                        help='Style code dimension')
    parser.add_argument('--use_mini_batch_discriminator', action='store_true', default=False)

    ## save and sample setting
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--save_every', type=int, default=200)
    parser.add_argument('--print_every', type=int, default=50)
    parser.add_argument('--version', type=str)
    parser.add_argument('--root_dir', type=str, help='exp dir', default='./dataset')
    parser.add_argument('--crop_size', nargs=3, type=int, default=[192, 224, 192])
    
    ## DP setting
    parser.add_argument('--gpu_id', type=str, default='1')


    args = parser.parse_args()
    #pdb.set_trace()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    main_work(args)


if __name__ == '__main__':
    
    main()





