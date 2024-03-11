import torch
import numpy as np
import csv
import os
import pdb
from torch.utils.data import Dataset
import SimpleITK as sitk
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize
import torch.nn.functional as F
from datetime import datetime



class paired_inertval_dataset(Dataset):

    def __init__(self, interval, image_paths1, image_paths2, 
                 crop=True, crop_size=(192, 224, 192), resize=False, resize_size=None, resolution=256):
        
        super().__init__()

        self.interval = interval
        self.crop = crop
        self.crop_size = crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.resolution = resolution
        self.image_paths1 = image_paths1
        self.image_paths2 = image_paths2
        
    
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
        
        folder1 = self.image_paths1[index]
        folder2 = self.image_paths2[index]

        img_file1 = os.listdir(folder1)[0]
        img_file2 = os.listdir(folder2)[0]
        if not img_file1.endswith('.nii.gz') or not img_file2.endswith('.nii.gz'):
            pdb.set_trace()
        
        img_path1 = os.path.join(folder1, img_file1)
        img_path2 = os.path.join(folder2, img_file2)
        img1 = sitk.GetArrayFromImage(sitk.ReadImage(img_path1))
        img2 = sitk.GetArrayFromImage(sitk.ReadImage(img_path2))
        img1 = self._preprocess_img(img1)
        img2 = self._preprocess_img(img2)
       
        return img_path1, img_path2, img1, img2

    def __len__(self):
        return len(self.image_paths1)
    

def divide_by_intervals(ori_csv, interval=1, img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'):

    subject_dates = {}
    subject_rows = {}
    with open(ori_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row['Subject']
            date = row['Acq Date']
            if not os.path.exists(os.path.join(img_dir, subject, date)) or date == '':
                continue
            else:
                if '-' in date:
                    subject_dates[subject] = [datetime.strptime(date, '%Y-%m-%d')] if subject not in subject_dates.keys() else subject_dates[subject] + [datetime.strptime(date, '%Y-%m-%d')]
                elif 'd' in date:
                    subject_dates[subject] = [date] if subject not in subject_dates.keys() else subject_dates[subject] + [date]
                subject_rows[subject]  = [row] if subject not in subject_rows.keys() else subject_rows[subject] + [row]
    
    rows1 = []
    rows2 = []
    for subject, dates in subject_dates.items():
        
        if len(dates) == 1:
            continue

        for idx, date in enumerate(dates[:-1]):
            #diff_day = []
            for i in range(idx+1, len(dates)):
                if not type(date) == str:
                    diff_day = (dates[i] - date).days
                else:
                    diff_day = int(dates[i][1:]) - int(date[1:])
                if diff_day > (interval*365 - 60) and diff_day < (interval*365 + 60):
                    rows1.append(subject_rows[subject][idx])
                    rows2.append(subject_rows[subject][i])

    return rows1, rows2


def get_A_B_image_paths(ori_csv, interval=1, 
                        img_dir='/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI'):

    rows1, rows2 = divide_by_intervals(ori_csv=ori_csv, interval=interval, img_dir=img_dir)
    
    folders1 = []
    folders2 = []
    
    for idx, row in enumerate(rows1):
        subject = row['Subject']
        date1 = row['Acq Date']

        date2 = rows2[idx]['Acq Date']
        folders1.append(os.path.join(img_dir, subject, date1))
        folders2.append(os.path.join(img_dir, subject, date2))

    return folders1, folders2

def get_paired_dataset(dataset_type, interval=1, is_train=True,
                       crop=True, crop_size=(192, 224, 192), resize=False, resize_size=None):

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
    dataset = paired_inertval_dataset(interval=interval, image_paths1=folders1, image_paths2=folders2,
                                      crop=crop, crop_size=crop_size, resize=resize, resize_size=resize_size)

    return dataset

def get_rnn_paths(dataset_type, single_interval, dataset_session_length=3, is_train=True):
    
    ## get csv and img dir
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

    ## get all subject-dates 
    subject_dates = {}
    subject_rows = {}
    with open(ori_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row['Subject']
            date = row['Acq Date']
            if not os.path.exists(os.path.join(img_dir, subject, date)) or date == '':
                continue
            else:
                if '-' in date:
                    subject_dates[subject] = [datetime.strptime(date, '%Y-%m-%d')] if subject not in subject_dates.keys() else subject_dates[subject] + [datetime.strptime(date, '%Y-%m-%d')]
                elif 'd' in date:
                    subject_dates[subject] = [date] if subject not in subject_dates.keys() else subject_dates[subject] + [date]
                subject_rows[subject]  = [row] if subject not in subject_rows.keys() else subject_rows[subject] + [row]
    
    valid_subject_dates = []
    
    ## get valids dataset_session_length data
    for subject, dates in subject_dates.items():
        
        if len(dates) < dataset_session_length:
            continue

        for idx, date in enumerate(dates[:-dataset_session_length+1]):
            
            if not type(date) == str:
                diff_days = [(dates[i] - date).days for i in range(idx+1, len(dates))]
                days_type = 'datetime'
            else:
                diff_days = [int(dates[i][1:]) - int(date[1:]) for i in range(idx+1, len(dates))]
                days_type = 'day'
                
            j_dates = [date]
            for j in range(dataset_session_length-1):
                inertval_j = (j+1)*single_interval
                j_date_idx = np.argmin([np.abs(diff_day - inertval_j) for diff_day in diff_days])
                error_days = [np.abs(diff_day - inertval_j) for diff_day in diff_days][j_date_idx]
                if error_days < (single_interval-30):
                    j_dates.append(dates[idx+1:][j_date_idx])
                else:
                    j_dates.append(False)
            
            if not False in j_dates:
                if days_type == 'datetime':
                    j_dates = [j_date.strftime('%Y-%m-%d') for j_date in j_dates]
                else:
                    j_dates = ['d{:.0>4d}'.format(j_date) for j_date in j_dates]
                
                if len(j_dates)==len(set(j_dates)):
                    valid_subject_dates.append((subject, j_dates))
                #print(subject, *j_dates)
    
    ## create datapaths
    
    img_paths = []
    for (subject, dates) in valid_subject_dates:
        #subject = list(subject__dates.keys())[0]
        #dates = subject__dates[subject]
        path_ = [os.path.join(img_dir, subject, date) for date in dates]
        img_paths.append(path_)
    
    return img_paths
        
    

class RNN_dataset(Dataset):
    
    def __init__(self, dataset_session_length, dataset_type, single_interval=180,
                 crop=True, crop_size=(192, 224, 192), resize=True, resize_size=(128, 128, 128), is_train=True):
        
        super().__init__()

        self.dataset_session_length = dataset_session_length
        self.crop = crop
        self.crop_size = crop_size
        self.resize = resize
        self.resize_size = resize_size
        self.dataset_type = dataset_type
        self.single_interval = single_interval
        self.image_paths = get_rnn_paths(dataset_type, 
                                         single_interval = single_interval, 
                                         dataset_session_length=dataset_session_length,
                                         is_train=is_train)
        
    
    def _preprocess_img(self, img):
        
        img = torch.tensor(img)

        if self.crop:
            img = SpatialPad(spatial_size=self.crop_size)(img.unsqueeze(0)).squeeze()
            img = CenterSpatialCrop(roi_size=self.crop_size)(img.unsqueeze(0)).squeeze()
        if self.resize:
            img = Resize(spatial_size=self.resize_size)(img.unsqueeze(0)).squeeze()
            
        img = (img/torch.max(img) - 0.5)*2
        img = img.unsqueeze(0)

        return img
    
    def __getitem__(self, index):
        
        folders = self.image_paths[index]
        imgs = []
        img_paths = []
        #print(folders)
        for folder in folders:
            img_file = os.listdir(folder)[0]
            if not img_file.endswith('.nii.gz'):
                pdb.set_trace()
            img_path = os.path.join(folder, img_file)
            img = sitk.GetArrayFromImage(sitk.ReadImage(img_path))
            img = self._preprocess_img(img)
            imgs.append(img)
            img_paths.append(img_path)
                
        
        imgs = torch.cat(imgs, dim=0).unsqueeze(1)
       
        return {'paths_list':img_paths, 'images':imgs.to(torch.device('cuda:0'))}

    def __len__(self):
        return len(self.image_paths)
    

if __name__ == '__main__':

    
    dataset = RNN_dataset(dataset_session_length=2, dataset_type='ADNI', single_interval=360,
                 crop=True, crop_size=(193, 229, 193), resize=True, resize_size=(128, 128, 128), is_train=False)
    dataset = RNN_dataset(dataset_session_length=2, dataset_type='ADNI', single_interval=360,
                 crop=True, crop_size=(193, 229, 193), resize=True, resize_size=(128, 128, 128), is_train=True)
    print(len(dataset))



    
        
    