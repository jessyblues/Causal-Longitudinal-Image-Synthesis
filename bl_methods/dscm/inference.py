import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import sys
sys.path.append('/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dscm')
from vae_3D import VAE3D, vae_loss
import os
import logging
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import random
import pandas as pd
from typing import Dict
from torch import Tensor
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    RandSpatialCrop,
    CenterSpatialCrop,
    Resize,
    SpatialPad,
)


class ADNI_3d_Dataset_inference(Dataset):
    def __init__(
        self,
        interval: int,
        dataset_type: str,
        parents = ['VIT', 'GreyMatter', 'Ventricles'],
        transform=None,
        min_and_max:dict = None,
        concat_pa=True
    ):
        super().__init__()
        self.interval = interval
        self.dataset_type = dataset_type
        self.parents = parents

        self.csv_path = f'/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dataset_config/{dataset_type}/interval={interval}year_gt.csv'
        self.img_dir = f'/home/yujiali/dataset/ADNI/T1/aligned_brain_MNI' if dataset_type == 'ADNI' else f'/home/yujiali/dataset/NACC/T1/brain_mni'
        self.concat_pa = concat_pa  # return concatenated parents
        self.transform = transform

        print(f"\nLoading subject csv data: {self.csv_path}")
        
        self.df = pd.read_csv(self.csv_path)
        self.data_list = []
        for i in range(len(self.df)):
            subject = self.df.iloc[i]['Subject']
            date1 = self.df.iloc[i]['Date1']
            date2 = self.df.iloc[i]['Date2']
            image_file1 = os.listdir(os.path.join(self.img_dir, subject, date1))[0]
            image_path1 = os.path.join(self.img_dir, subject, date1, image_file1)
            image_file2 = os.listdir(os.path.join(self.img_dir, subject, date2))[0]
            image_path2 = os.path.join(self.img_dir, subject, date2, image_file2)
            
            VIT1 = self.df.iloc[i]['WholeBrain_1']
            GreyMatter1 = self.df.iloc[i]['GreyMatter_1']
            Ventricles1 = self.df.iloc[i]['SegVentricles_1']
            VIT2 = self.df.iloc[i]['WholeBrain_2']
            GreyMatter2 = self.df.iloc[i]['GreyMatter_2']
            Ventricles2 = self.df.iloc[i]['SegVentricles_2']
            
            VIT1 = (VIT1 - min_and_max['brain_volume'][0]) / (min_and_max['brain_volume'][1] - min_and_max['brain_volume'][0]) * 2 - 1
            GreyMatter1 = (GreyMatter1 - min_and_max['grey_matter'][0]) / (min_and_max['grey_matter'][1] - min_and_max['grey_matter'][0]) * 2 - 1
            Ventricles1 = (Ventricles1 - min_and_max['ventricle_volume'][0]) / (min_and_max['ventricle_volume'][1] - min_and_max['ventricle_volume'][0]) * 2 - 1
            VIT2 = (VIT2 - min_and_max['brain_volume'][0]) / (min_and_max['brain_volume'][1] - min_and_max['brain_volume'][0]) * 2 - 1
            GreyMatter2 = (GreyMatter2 - min_and_max['grey_matter'][0]) / (min_and_max['grey_matter'][1] - min_and_max['grey_matter'][0]) * 2 - 1
            Ventricles2 = (Ventricles2 - min_and_max['ventricle_volume'][0]) / (min_and_max['ventricle_volume'][1] - min_and_max['ventricle_volume'][0]) * 2 - 1
            
            self.data_list.append({
                'image_path1': image_path1,
                'image_path2': image_path2,
                'Subject': subject,
                'Date1': date1,
                'Date2': date2,
                'VIT1': VIT1,
                'GreyMatter1': GreyMatter1,
                'Ventricles1': Ventricles1,
                'VIT2': VIT2,
                'GreyMatter2': GreyMatter2,
                'Ventricles2': Ventricles2})

        

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        sample = self.data_list[idx]
        x1, x2 = sitk.ReadImage(sample['image_path1']), sitk.ReadImage(sample['image_path2'])
        x1, x2 = sitk.GetArrayFromImage(x1), sitk.GetArrayFromImage(x2)  # z, y, x
        x1, x2 = torch.tensor(x1).unsqueeze(0).float(), torch.tensor(x2).unsqueeze(0).float()  # 1,
        
        x1 = x1 - x1.min()
        x1 = x1 / x1.max()  # normalize to [0,1]
        
        x2 = x2 - x2.min()
        x2 = x2 / x2.max()  # normalize to [0,1]
        
        if self.transform is not None:
            sample["x1"] = self.transform(x1)
            sample["x2"] = self.transform(x2)


        if self.concat_pa:
            #ipdb.set_trace()
            sample["pa1"] = torch.cat(
                [torch.tensor([sample[k+'1']]) for k in self.parents], dim=0
            )
            sample["pa2"] = torch.cat(
                [torch.tensor([sample[k+'2']]) for k in self.parents], dim=0
            )
        #print(sample["pa"].shape)
        #quit()
        return sample

def get_dataloader(args, dataset_type='ADNI', interval=1):
    
    df = pd.read_csv('/home/yujiali/dataset/ADNI/T1/excel/all_seg.csv')
    infrence_dataset = ADNI_3d_Dataset_inference(
        interval=interval,
        dataset_type=dataset_type,
        transform=Compose(
            [
                SpatialPad(spatial_size=(192, 224, 192), mode='symmetric'),
                CenterSpatialCrop((160, 192, 160)),
                # Resize((args.input_res, args.input_res, args.input_res)),
                Resize((args.input_res, args.input_res, args.input_res)),
            ]
        ),
        min_and_max={
            'brain_volume': [df['WholeBrain'].min(), df['WholeBrain'].max()],
            'grey_matter': [df['GreyMatter'].min(), df['GreyMatter'].max()],
            'ventricle_volume': [df['SegVentricles'].min(), df['SegVentricles'].max()]
        }
    )
    
    return torch.utils.data.DataLoader(
        infrence_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    
reverse_transform = Compose(
    [
        Resize((160, 192, 160))
    ]
)

def preprocess_batch_3d(batch: Dict[str, Tensor], device:torch.device):
    batch["x1"] = batch["x1"].to(device)
    batch["pa1"] = batch["pa1"].to(device).float()
    
    batch['pa2'] = batch['pa2'].to(device).float()

    return batch



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_res', type=int, default=160, help='input resolution')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--img_save_dir', type=str, default='/home/yujiali/cf_mri_2/mia_added_comaparison_methods/dscm/inference_result', help='inference result save dir')
    parser.add_argument('--latent_dim', type=int, default=1024, help='潜在向量维度')
    args = parser.parse_args()
    
    print(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VAE3D(latent_dim=args.latent_dim).to(device)
    
    model.load_state_dict(torch.load(args.resume, map_location=device))
    model.eval()
    
    
    for dataset in ['ADNI', 'NACC']:
        for interval in [1, 2, 4]:
            args.dataset = dataset
            args.interval = interval
            

            dataloader = get_dataloader(args, dataset_type=dataset, interval=interval)
            
            img_save_dir = os.path.join(args.img_save_dir, f'{args.dataset}_interval={args.interval}year')
            os.makedirs(img_save_dir, exist_ok=True)
            

            for idx, batch in enumerate(dataloader):
                batch = preprocess_batch_3d(batch, device)
                x1, pa2 = batch['x1'], batch['pa2']
                subject, src_date, dst_date = batch['Subject'][0], batch['Date1'][0], batch['Date2'][0]
                sample_idx = 0
                
                with torch.no_grad():
                    cf_img = model.gen_cf(x1, pa2)

                    cf_img = reverse_transform(cf_img.squeeze(0))
                    
                dst_dir = os.path.join(img_save_dir, subject, dst_date)
                os.makedirs(dst_dir, exist_ok=True)
                img = cf_img.cpu().numpy().squeeze()
                import SimpleITK as sitk
                sitk_img = sitk.GetImageFromArray(img)
                sitk.WriteImage(sitk_img, os.path.join(dst_dir, f'cf_mri.nii.gz'))
                
                print(f'{idx+1}/{len(dataloader)} done!')
                    