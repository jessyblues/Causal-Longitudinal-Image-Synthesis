import torch
import math
import time
from bl_methods.TR_GAN_main.options.train_options import TrainOptions
from bl_methods.TR_GAN_main.data import create_dataset
from bl_methods.TR_GAN_main.models import create_model
from bl_methods.TR_GAN_main.util.visualizer import Visualizer
import datetime
from bl_methods.TR_GAN_main.util.test_and_inference import test_data_loss
from bl_methods.TR_GAN_main.util.GPUManager import GPUManager
import numpy as np
import random
import ants
import os
from bl_methods.utils import RNN_dataset
from torch.utils.data import DataLoader
import pdb

class my_visuliazer():
    
    def __init__(self, exp_dir):
        self.exp_dir = exp_dir
        self.img_dir = os.path.join(exp_dir, 'visualise_img')
        self.ckpt_dir = os.path.join(exp_dir, 'ckpt')

        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save_imgs(self, model_visuals, epoch, iter):

        for label, images_batch in model_visuals.items():            
            image = images_batch[0, :, :, :, :].detach().cpu().numpy().squeeze()
            image = image-np.min(image)
            ants.from_numpy(image).to_file(os.path.join(self.img_dir, 'epoch={}_iter={}_{}.nii.gz'.format(epoch, iter, label)))

    def print_current_losses(self, epoch, total_iters, losses, t_data):

        print('epoch {} iter {} time {}'.format(epoch, total_iters, t_data))
        message = ''
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        print(message)  # print the message

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.enabled = False


if __name__ == '__main__':

    set_seed(2020)
    opt = TrainOptions().parse()  # get training options
    

    opt.phase = 'train'
    opt.suffix = opt.model + opt.netG + '_exp_T_' + str(opt.T_length) + '_' + datetime.datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S")
    # opt.name = opt.suffix

    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    
    dataset = RNN_dataset(dataset_session_length=opt.T_length+1, dataset_type=opt.dataset, single_interval=30*opt.interval,
                          crop=True, crop_size=(193, 229, 193), resize=True, resize_size=(128, 128, 128), is_train=(opt.phase=='train'))
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=True)
    
    
    dataset_size = len(dataset)  # get the number of images in the dataset.
    
    print('The number of training images = %d' % dataset_size)
    
    visualizer = my_visuliazer(exp_dir=opt.checkpoints_dir)
    #visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    opt.display_env = 'train' + opt.suffix
    best_l1_loss = 10000
    best_epoch = None
    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        #tq = tqdm.tqdm(total=math.ceil(dataset_size / opt.batch_size))
        #tq.set_description('Epoch {}'.format(epoch))
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        #visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        model.update_learning_rate()  # update learning rates in the beginning of every epoch.
        model.log_epoch(epoch, opt.n_epochs, opt.n_epochs_decay)  # log epoch to use changed loss
        
        for i, data in enumerate(dataloader):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            
            #data = {'paths_list':img_paths, 'images':image.to(torch.device('cuda:0'))}

            model.set_input(data, epoch)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()  # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:  # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.save_imgs(model.get_current_visuals(), epoch, total_iters)

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                #pdb.set_trace()
                visualizer.print_current_losses(epoch=epoch, total_iters=total_iters, losses=losses, t_data=t_data)
            
            if total_iters == 0:
                train_l1_loss = test_data_loss(model, dataloader, epoch, opt, phase=opt.phase, save_results=False, save_freq=50,
                                       all_metric=False, is_train=True)
                
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size

            #tq.update(1)
            iter_data_time = time.time()
        
        if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks(epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' % (
            epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        eval_start_time = time.time()
        # test on train dataset
        train_l1_loss = test_data_loss(model, dataloader, epoch, opt, phase=opt.phase, save_results=False, save_freq=50,
                                       all_metric=False, is_train=True)
        print('Eval {} dataset, Time Taken: {} sec'.format(opt.phase, time.time() - eval_start_time))
        if train_l1_loss < best_l1_loss:
            best_l1_loss = train_l1_loss
            model.save_networks(epoch, delete_epoch=best_epoch)
            best_epoch = epoch
            print('Find better L1 loss and saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
