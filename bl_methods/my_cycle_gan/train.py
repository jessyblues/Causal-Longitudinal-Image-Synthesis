
import os
import time
from bl_methods.pix2pix_and_CycleGAN.options.train_options import TrainOptions
#from data.data_loader import CreateDataLoader
from bl_methods.pix2pix_and_CycleGAN.models.models import create_model
from bl_methods.my_cycle_gan.utils import save_for_visaul, print_current_errors
from bl_methods.utils import get_paired_dataset
from torch.utils.data import DataLoader
import pdb

opt = TrainOptions().parse()

dataset = get_paired_dataset(interval=opt.interval, dataset_type=opt.dataset_type, 
                             resize=True, resize_size=(96, 112, 96))
dataset_size = len(dataset)
dataloader = DataLoader(dataset=dataset, batch_size=opt.batchSize, shuffle=True)


print('#training images = %d' % dataset_size)

model = create_model(opt)
#visualizer = Visualizer(opt)
#pdb.set_trace()
total_steps = 0
start_time = time.time()

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0
    for i, (img_path1, img_path2, img1, img2) in enumerate(dataloader):
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        input = {'A':img1, 'B':img2, 'A_paths':img_path1, 'B_paths':img_path2}
        model.set_input(input)
        model.optimize_parameters()

        if total_steps % opt.display_freq == 0:
            #visualizer.display_current_results(model.get_current_visuals(), epoch)
            save_for_visaul(exp_dir=opt.exp_dir, epoch=epoch, iter=epoch_iter, current_visuals=model.get_current_visuals())
            #pass

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - start_time)
            start_time = time.time()
            #visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            #if opt.display_id > 0:
            #    visualizer.plot_current_errors(epoch, float(epoch_iter)/dataset_size, opt, errors)
            print_current_errors(epoch, epoch_iter, errors, t)


        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)
        
    if epoch > opt.niter:
        model.update_learning_rate()