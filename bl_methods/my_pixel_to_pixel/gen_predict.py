
import os
import time
from bl_methods.pix2pix_and_CycleGAN.options.test_options import TestOptions
from bl_methods.pix2pix_and_CycleGAN.models.models import create_model
from bl_methods.utils import get_paired_dataset
from torch.utils.data import DataLoader
import SimpleITK as sitk

opt = TestOptions().parse()

dataset = get_paired_dataset(interval=opt.interval, dataset_type=opt.dataset_type, is_train=False)
dataset_size = len(dataset)
dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)


print('#test images = %d' % dataset_size)
model = create_model(opt)
total_steps = 0
start_time = time.time()
output_dir = os.path.join(opt.results_dir,opt.dataset_type,'interval={}_year'.format(opt.interval))
os.makedirs(output_dir, exist_ok=True)



for i, (img_path1, img_path2, img1, img2) in enumerate(dataloader):


    input = {'A':img1, 'B':img2, 'A_paths':img_path1, 'B_paths':img_path2}
    #pdb.set_trace()
    subject = img_path2[0].split('/')[-3]
    date = img_path2[0].split('/')[-2]
    single_folder = os.path.join(output_dir, subject, date)
    os.makedirs(single_folder, exist_ok=True)
    
    model.set_input(input)
    model.test()
    current_visuals = model.get_current_visuals()
    
    real_A = current_visuals['real_A'].squeeze()
    fake_B = current_visuals['fake_B'].squeeze()
    real_B = current_visuals['real_B'].squeeze()

    assert fake_B.ndim == 3, print('please set bacthsize to 1')
    #pdb.set_trace()
    sitk.WriteImage(sitk.GetImageFromArray(fake_B), os.path.join(single_folder, 'fakeB.nii.gz'))
    
    print(single_folder, 'finished!')





