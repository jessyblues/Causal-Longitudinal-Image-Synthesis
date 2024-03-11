import os
import sys
import time
from bl_methods.pix2pix_and_CycleGAN.options.test_options import TestOptions
#from data.data_loader import CreateDataLoader
from bl_methods.pix2pix_and_CycleGAN.models.models import create_model
from bl_methods.my_pixel_to_pixel.utils import save_for_visaul, print_current_errors
from bl_methods.utils import get_paired_dataset
from torch.utils.data import DataLoader
import pdb
import SimpleITK as sitk
from scipy import ndimage
import ants
import numpy as np
import torch

opt = TestOptions().parse()
#data_loader = CreateDataLoader(opt)
#dataset = data_loader.load_data()
#dataset_size = len(data_loader)

dataset = get_paired_dataset(interval=opt.interval, dataset_type=opt.dataset_type, 
                             is_train=False, resize=True, resize_size=(96, 112, 96))
dataset_size = len(dataset)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)



print('#test images = %d' % dataset_size)
#pdb.set_trace()
model = create_model(opt)
#visualizer = Visualizer(opt)
#pdb.set_trace()
total_steps = 0
start_time = time.time()
output_dir = os.path.join(opt.results_dir,opt.dataset_type,'interval={}_year'.format(opt.interval))
os.makedirs(output_dir, exist_ok=True)



for i, (img_path1, img_path2, img1, img2) in enumerate(dataloader):

    

    input = {'A':img1, 'B':img2, 'A_paths':img_path1, 'B_paths':img_path2}

    subject = img_path2[0].split('/')[-3]
    date = img_path2[0].split('/')[-2]
    single_folder = os.path.join(output_dir, subject, date)
    os.makedirs(single_folder, exist_ok=True)
    
    model.set_input(input)
    model.test()
    current_visuals = model.get_current_visuals()
    
    fake_B = current_visuals['fake_B'].squeeze()

    assert fake_B.ndim == 3, print('please set bacthsize to 1')
    
    if opt.interval == 1:
        pass
    else:
        for i in range(opt.interval//2):
            fake_B = torch.tensor(fake_B, device=torch.device('cuda:0')).unsqueeze(0).unsqueeze(0)
            input = {'A':fake_B, 'B':img2, 'A_paths':img_path1, 'B_paths':img_path2}
            model.set_input(input)
            model.test()
            current_visuals = model.get_current_visuals()
            fake_B = current_visuals['fake_B'].squeeze()
            
    fake_B = ndimage.zoom(fake_B, (2, 2, 2)) 
    fake_B = (fake_B - np.min(fake_B)) / (np.max(fake_B) - np.min(fake_B))
    back_ground = float(fake_B[0, 0, 0])
    fake_B = np.clip(fake_B-back_ground, a_min=0, a_max=None)
    fake_B = fake_B / (1 - back_ground)

    mask = ants.get_mask(ants.from_numpy(fake_B))
    fake_B = fake_B*(mask.numpy()) 
    sitk.WriteImage(sitk.GetImageFromArray(fake_B), os.path.join(single_folder, 'fakeB.nii.gz'))
    
    print(single_folder, 'finished!')





