import torch
from fft_conv_pytorch import fft_conv, FFTConv3d
from GAN_mri_package.dataset import info_Dataset, MriFileFolderDataset
import os
import SimpleITK as sitk
import pdb


if __name__ == '__main__':
    
    sys_image_folder = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'
    subjects = os.listdir(sys_image_folder)
    root_folder= sys_image_folder


    for subject in sorted(subjects):
        subject_folder = os.path.join(root_folder, subject)
        dates = sorted(os.listdir(subject_folder))

        for date in sorted(dates):
            
            date_folder = os.path.join(subject_folder, date)
            sys_file = os.listdir(date_folder)[0]
            
            
            sys_image = sitk.ReadImage(os.path.join(date_folder, sys_file))
            sys_image = torch.tensor(sitk.GetArrayFromImage(sys_image), dtype=torch.float)

            fftn = torch.fft.fftn(sys_image)

            #pdb.set_trace()




