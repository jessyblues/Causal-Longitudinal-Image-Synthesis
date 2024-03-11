import os

from  bl_methods.TR_GAN_main.options.test_options import TestOptions
from  bl_methods.TR_GAN_main.models import create_model

from bl_methods.utils import RNN_dataset
from torch.utils.data import DataLoader
import ants
import numpy as np
import pdb
def save_predict_imgs(img_dir, model_visuals, interval):
    
    need_label = 'fake_ses_M{}'.format(interval)
    #pdb.set_trace()
    for label, images_batch in model_visuals.items():            
        if need_label != label:
            continue
        image = images_batch[0, :, :, :, :].detach().cpu().numpy().squeeze()
        image = image-np.min(image)
        
        ants.from_numpy(image).to_file(os.path.join(img_dir, '{}.nii.gz'.format(need_label)))


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True  # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1  # no visdom display; the test code saves the results to a HTML file.
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # create a website
    

    year = opt.T_test
        
    dataset = RNN_dataset(dataset_session_length=2, dataset_type=opt.dataset, single_interval=30*opt.interval*year,
                          crop=True, crop_size=(193, 229, 193), resize=True, resize_size=(128, 128, 128), is_train=False)
    dataloader = DataLoader(dataset=dataset, batch_size=opt.batch_size, shuffle=False)

    # test with eval mode. This only affects layers like batchnorm and dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataloader):
        #if i >= opt.num_test:  # only apply our model to opt.num_test images.
        #    break
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results
        #img_path = model.get_image_paths()  # get image paths
        #pdb.set_trace()
        final_img_path = data['paths_list'][-1][0]
        
        subject = final_img_path.split('/')[-3]
        date = final_img_path.split('/')[-2]
        
        single_dir = os.path.join(opt.results_dir, opt.dataset, 'interval={}year'.format(year), subject, date)
        os.makedirs(single_dir, exist_ok=True)
        save_predict_imgs(single_dir, visuals, interval=year*12)