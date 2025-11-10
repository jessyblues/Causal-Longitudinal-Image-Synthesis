
import os
import pdb

import numpy as np

import torch
from torch.utils import data
import SimpleITK as sitk
import pandas as pd
import torch.nn.functional as F

from models.GAN_mri_package.transforms import Compose, RandomFlip_LR, RandomFlip_UD
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize
from glob import glob


class MriFileFolderDataset(data.Dataset):

    def __init__(self, root, 
                w_file='/home1/yujiali/GAN_mri/fake_dataset/latent.npy', 
                return_latent=False, return_pth=False, 
                crop=False, crop_size=None, 
                resize=False, resize_size=None,
                resolution=256, 
                walk=False):
        
        super().__init__()

        self.paths = []
        if type(root) == str:
            root = [root]
        for root_ in root:
            self.paths += glob(os.path.join(root_, '**', '*.nii.gz'), recursive=walk)
        #pdb.set_trace()

        self.return_pth = return_pth
        self.crop = crop
        self.crop_size = crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.resolution = resolution

        if return_latent:
            self.latents = torch.tensor(np.load(w_file))
        else:
            self.latents = None


    def __len__(self):
        return len(self.paths)
	
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

        downsample_scale = int(256/self.resolution)
        img = F.avg_pool3d(img, (downsample_scale))
    
        return img

    def __getitem__(self, index):

        from_path = self.paths[index]

        img = sitk.ReadImage(from_path)
        img = sitk.GetArrayFromImage(img) 
        img = self._preprocess_img(img)


        if self.latents is None:
            if self.return_pth:
                return img, self.paths[index]
            else:
                return img
        else:
            if self.return_file_name:
                return img, self.latents[index], self.paths[index]
            else:
                return img, self.latents[index]
            




class info_Dataset(data.Dataset):
    
    def __init__(self, csv_path, eps:float=1e-4, apply_log=False):
        
        super().__init__()
        self.csv_path = csv_path
        self.apply_log = apply_log


        csv = pd.read_csv(csv_path)

        #csv['duration'] = csv['duration'].fillna(0.) + eps
        csv['score'] = csv['MMSE'].fillna(0.)
        csv['age'] = csv['Age'].fillna(0.)

        csv['sex'] = csv['Sex'].map({'M': 0., 'F': 1.})
        #csv['group'] = csv['Group'].map({'CN': 0., 'MCI': 1., 'AD': 2.})
        csv['ventricle_volume'] = csv['VentricleVolume'].astype(np.float32)
        csv['brain_volume'] = csv['BrainVolume'].astype(np.float32)
        csv['GM_volume'] = csv['GreyMatterVolume'].astype(np.float32)
        csv['CSF_volume'] = csv['CSFVolume'].astype(np.float32)
        csv['WM_volume'] = csv['WhiteMatterVolume'].astype(np.float32)
        
        csv['ventricle_ratio'] = csv['Ventricle_ratio'].astype(np.float32)
        csv['GM_ratio'] = csv['GM_ratio'].astype(np.float32)
        csv['CSF_ratio'] = csv['CSF_ratio'].astype(np.float32)
        csv['WM_ratio'] = csv['WM_ratio'].astype(np.float32)

        
        if csv.isnull().values.any():
            #pdb.set_trace()
            #print(np.argwhere(csv.isnull().values==True))
            raise ValueError(
                'There is either an empty space, nan, or otherwise something wrong in the csv {}'.format(csv_path),
                'location: {}'.format(np.argwhere(csv.isnull().values==True))
            )
            
        
        self.csv = csv

        #pdb.set_trace()


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        item = self.csv.loc[index]
        item = item.to_dict()
        item = self._prepare_item(item)

        return item

    def _prepare_item(self, item):

        item['age'] = torch.as_tensor(item['age'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['age'], dtype=torch.float32))
        item['sex'] = torch.as_tensor(item['sex'], dtype=torch.float32)
        #item['group'] = torch.as_tensor(item['group'], dtype=torch.float32)
        
        #item['duration'] = torch.as_tensor(item['duration'], dtype=torch.float32)
       
        item['brain_volume'] = torch.as_tensor(item['brain_volume'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['brain_volume'], dtype=torch.float32))
        item['ventricle_volume'] = torch.as_tensor(item['ventricle_volume'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['ventricle_volume'], dtype=torch.float32))
        item['GM_volume'] = torch.as_tensor(item['GM_volume'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['GM_volume'], dtype=torch.float32))
        
        item['CSF_volume'] = torch.as_tensor(item['CSF_volume'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['CSF_volume'], dtype=torch.float32))
        item['WM_volume'] = torch.as_tensor(item['WM_volume'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['WM_volume'], dtype=torch.float32))
        
        item['score'] = torch.as_tensor(item['score'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['score'], dtype=torch.float32))
        
        item['GM_ratio'] = torch.as_tensor(item['GM_ratio'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['GM_ratio'], dtype=torch.float32))
        item['ventricle_ratio'] = torch.as_tensor(item['ventricle_ratio'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['ventricle_ratio'], dtype=torch.float32))
        item['WM_ratio'] = torch.as_tensor(item['WM_ratio'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['WM_ratio'], dtype=torch.float32))
        item['CSF_ratio'] = torch.as_tensor(item['CSF_ratio'], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item['CSF_ratio'], dtype=torch.float32))


        key_list = list(item.keys())
        for key in key_list:
            if not key in ['ImageID', 'SubjectID', 'age', 'sex',
                           'brain_volume', 'ventricle_volume',
                           'GM_volume', 'score', 'GM_ratio', 'ventricle_ratio',
                           'WM_volume', 'WM_ratio', 'CSF_volume', 'CSF_ratio']:
                item.pop(key)
            
        
        return item

class SegFakeMriDataset(data.Dataset):

    def __init__(self, root, img_crop=True, return_file_name=False, crop_size=(160, 160, 160)):
        
        super().__init__()

        self.paths = []
        if type(root) == str:
            root = [root]
        for root_ in root:
            self.paths += sorted(make_dataset(root_,  data_type='mri_3d',walk=True))
        #pdb.set_trace()

        self.return_file_name = return_file_name
        self.img_crop = img_crop
        self.crop_size = crop_size


    def __len__(self):
        return len(self.paths)
	
    def _preprocess_img(self, img):
        
       
        img = img.swapaxes(1, 2) ## (224, 224, 256) z, x, y
        
        #if self.img_crop:
        #s    img = img[32:-32, 32:-32, 48:-48]
        #if self.img_crop:
        #    img = SpatialPad(spatial_size=self.crop_size)(img.unsqueeze(0)).squeeze()
        #    img = CenterSpatialCrop(roi_size=self.crop_size)(img.unsqueeze(0)).squeeze()


        img = torch.tensor(img)
        img = img/torch.max(img)
        img = img.unsqueeze(0)

        if self.img_crop:
            img = SpatialPad(spatial_size=self.crop_size)(img)
            img = CenterSpatialCrop(roi_size=self.crop_size)(img)

        return img
    

    def __getitem__(self, index):
        #pdb.set_trace()
        from_path = self.paths[index]
        #pdb.set_trace()
        img = sitk.ReadImage(from_path)
        img = sitk.GetArrayFromImage(img) ## (241, 286, 241), z, y, x
        #pdb.set_trace()
        img = self._preprocess_img(img)

        subject, date, file_name = tuple(self.paths[index].split('/')[-3:])

        if self.return_file_name:
            return img, subject, date, file_name
        else:
            return img

class SegMriFileFolderDataset(data.Dataset):

    def __init__(self, root, img_crop=True, mask_crop=True, img_aug=True, return_file_name=False):
        
        super().__init__()

        self.paths = []
        if type(root) == str:
            root = [root]
        for root_ in root:
            self.paths += sorted(make_dataset(root_,  data_type='mri_3d'))
        #pdb.set_trace()

        self.return_file_name = return_file_name
        self.img_crop = img_crop
        self.mask_crop = mask_crop
        self.transforms = Compose([
                #RandomCrop(crop_size),
                RandomFlip_LR(prob=0.5),
                RandomFlip_UD(prob=0.5),
            ])
        self.img_aug = img_aug


    def __len__(self):
        return len(self.paths)
	
    def _preprocess_img(self, img):
        
       
        img = img.swapaxes(1, 2) ## (241, 241, 286) z, x, y
        
        if self.img_crop:
            img = img[41:-40, 41:-40, 63:-63]
            #img = img[41:-40, 41:-40, 55:-55]
            #img = img[49:-48, 49:-48, 71:-71]

        img = torch.tensor(img)

        img = img/torch.max(img)
        img = img.unsqueeze(0)
        #pdb.set_trace()
        return img
    
    def _preprocess_mask(self, mask):
        
       
        mask = mask.swapaxes(1, 2) ## (241, 241, 286) z, x, y
        
        if self.mask_crop:
            mask = (mask[41:-40, 41:-40, 63:-63]>0).astype(np.float32)
        
        mask = torch.tensor(mask)
        mask = mask.unsqueeze(0)

        return mask


    def __getitem__(self, index):
        #pdb.set_trace()
        from_path = self.paths[index]
        #pdb.set_trace()
        img = sitk.ReadImage(from_path)
        img = sitk.GetArrayFromImage(img) ## (241, 286, 241), z, y, x
        img = self._preprocess_img(img)
        #print(img.shape)

        file_name = os.path.split(self.paths[index])[-1]

        if file_name[0] == 'I': # ADNI 1 data
            mask_dir = '/home1/yujiali/dataset/brain_MRI/ADNI1/output/ventricle_mask' 
        elif file_name[0] == '0': # oasis 2 data
            mask_dir = '/home1/yujiali/dataset/brain_MRI/oasis/oasis2/output/ventricle_mask' 
        elif file_name[0] == 'O': # oasis 3 data
            mask_dir = '/home1/yujiali/dataset/brain_MRI/oasis/oasis3/output/ventricle_mask'

        mask = sitk.ReadImage(os.path.join(mask_dir, file_name))
        mask = sitk.GetArrayFromImage(mask) ## (241, 286, 241), z, y, x
        mask = self._preprocess_mask(mask)  

        if self.img_aug:
            self.transforms(img, mask)      

        if self.return_file_name:
            return img, mask, file_name
        else:
            return img, mask

class info_Dataset2(data.Dataset):
    
    def __init__(self, csv_path, float_value=['Age', 'Years_bl', 'PTEDUCAT', 'APOE4', 'FDG', 
                                               'CDRSB', 'ADAS11', 'ADAS13', 'MMSE', 
                                               'Ventricles', 'Hippocampus', 'WholeBrain', 'Entorhinal', 'Fusiform', 'MidTemp',
                                               'Grey Matter', 'White Matter'],
                       apply_log=False):
        
        super().__init__()
        self.csv_path = csv_path
        self.float_value = float_value
        self.apply_log = apply_log
        csv = pd.read_csv(csv_path)


        csv['Sex'] = csv['Sex'].map({'M': 0., 'F': 1.})
        for k in float_value:
            csv[k] = csv[k].astype(np.float32)
        self.float_value.append('Sex')
        
        #if csv.isnull().values.any():
            #pdb.set_trace()
            #print(np.argwhere(csv.isnull().values==True))
        #    raise ValueError(
        #        'There is either an empty space, nan, or otherwise something wrong in the csv {}'.format(csv_path),
        #        'location: {}'.format(np.argwhere(csv.isnull().values==True))
        #    )
            
        
        self.csv = csv

        #pdb.set_trace()


    def __len__(self):
        return len(self.csv)

    def __getitem__(self, index):
        item = self.csv.loc[index]
        item = item.to_dict()
        item = self._prepare_item(item)

        return item

    def _prepare_item(self, item):

        for k in self.float_value:
            item[k] = torch.as_tensor(item[k], dtype=torch.float32) if not self.apply_log else torch.log(torch.as_tensor(item[k], dtype=torch.float32))
        
        return item
    



if __name__ == '__main__':

    dataset = MriFileFolderDataset(root='/home1/yujiali/dataset/test/brain_mni', crop=True, crop_size=(192, 224, 192), resolution=256, walk=True)
    img = dataset[0]