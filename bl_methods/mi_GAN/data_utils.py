import torch
import numpy as np
import csv
import os
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize

from bl_methods.utils import get_A_B_image_paths
from causal.attributes_model.utils import get_mean_and_std

def get_subject_info(dataset, mi_infos):
    
    mean_and_std = get_mean_and_std(pkl_path='/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/ADNI_mean_and_std.pkl')
    if dataset == 'ADNI':
        info_csv = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/test_add_mmse.csv'
    else:
        info_csv = '/home1/yujiali/dataset/brain_MRI/NACC/NACC1.csv'
        

    info_dicts = {}
    with open(info_csv, "r", encoding="utf-8") as f:

        reader = csv.DictReader(f)
        for row in reader:
            subject = row['Subject']
            date = row['Acq Date']
            mi_info_line = []
            
            for k, v in row.items():
                if k in mi_infos:
                    try:
                        v = float(v) if v not in mean_and_std.keys() else (float(v)-mean_and_std[k][0])/mean_and_std[k][1]        
                    except Exception as e:
                        if k == 'Age':
                            v = 70
                        elif k == 'PTEDUCAT':
                            v = 14
                        else:
                            v = 0
                    mi_info_line.append(v)
            info_dicts[(subject, date)] = mi_info_line   
    
    return info_dicts

class patch_paired_inertval_dataset(Dataset):

    def __init__(self, interval, image_paths1, image_paths2, info_dicts,
                 crop=True, crop_size=(160, 192, 160), resize=False, 
                 resize_size=None):
        
        super().__init__()

        self.interval = interval
        self.crop = crop
        self.crop_size = crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        self.info_dicts = info_dicts
        
    
    def _cut_patches(self, img):
        
        patches = []
        for i in range(4):
            for j in range(5):
                for k in range(4):
                    
                    i_1 = i*32
                    j_1 = j*32
                    k_1 = k*32
                    
                    patches.append(img[:, i_1:i_1+64, j_1:j_1+64, k_1:k_1+64])
                    
        return patches
        
    
    def _preprocess_img(self, img):
        
        img = torch.tensor(img)
        #pdb.set_trace()
        if self.crop:
            img = SpatialPad(spatial_size=self.crop_size)(img.unsqueeze(0)).squeeze()
            img = CenterSpatialCrop(roi_size=self.crop_size)(img.unsqueeze(0)).squeeze()
        if self.resize:
            img = Resize(spatial_size=self.resize_size)(img.unsqueeze(0)).squeeze()
            
        img = img/torch.max(img)
        img = img.unsqueeze(0)
        patches = self._cut_patches(img)

        return patches
    
    def __getitem__(self, index):
        
        img_index = index // (4*5*4)
        patch_index = index - img_index*(4*5*4)
        
        folder1 = self.image_paths1[img_index]
        folder2 = self.image_paths2[img_index]

        img_file1 = os.listdir(folder1)[0]
        img_file2 = os.listdir(folder2)[0]

        img_path1 = os.path.join(folder1, img_file1)
        img_path2 = os.path.join(folder2, img_file2)
        subject = folder1.split('/')[-2]
        date = folder1.split('/')[-1]
        #pdb.set_trace()
        
        info = torch.tensor(self.info_dicts[(subject, date)], dtype=torch.float)
        
        img1 = sitk.GetArrayFromImage(sitk.ReadImage(img_path1))
        img2 = sitk.GetArrayFromImage(sitk.ReadImage(img_path2))
        img_patches1 = self._preprocess_img(img1)
        img_patches2 = self._preprocess_img(img2)
        
        return img_path1, img_path2, info, img_patches1[patch_index], img_patches2[patch_index]

    def __len__(self):
        return len(self.image_paths1)*4*5*4
    

def get_paired_patches_dataset(dataset_type, interval=1, is_train=True,
                               infos = ['Age', 'Sex', 'PTEDUCAT', 'APOE4']):

    if dataset_type == 'ADNI':
        img_dir = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'
        ori_csv = '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/training.csv' if is_train else \
                    '/home1/yujiali/dataset/brain_MRI/ADNI/T1/excel/test.csv'
    elif dataset_type == 'OASIS':
        img_dir = '/home1/yujiali/dataset/brain_MRI/OASIS/oasis3/align_mni'
        ori_csv = '/home1/yujiali/dataset/brain_MRI/OASIS/oasis3/excel/oasis3.csv'
    elif dataset_type == 'NACC':
        img_dir = '/home1/yujiali/dataset/brain_MRI/NACC/aligned_mni'
        ori_csv = '/home1/yujiali/dataset/brain_MRI/NACC/NACC1.csv'

    folders1, folders2 = get_A_B_image_paths(ori_csv=ori_csv, interval=interval, img_dir=img_dir)
    info_dicts = get_subject_info(dataset=dataset_type, mi_infos=infos)
    dataset = patch_paired_inertval_dataset(interval=interval, image_paths1=folders1, image_paths2=folders2, info_dicts=info_dicts)
    
    return dataset