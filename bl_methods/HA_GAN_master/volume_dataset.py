from torch.utils.data import Dataset
from sklearn.model_selection import KFold
import numpy as np
import glob
from monai.transforms import CenterSpatialCrop, SpatialPad, Resize
from e4e.utils.data_utils import make_dataset
import torch
import SimpleITK as sitk
import pdb
class Volume_Dataset(Dataset):

    def __init__(self, data_dir, mode='train', fold=0, num_class=0):
        self.sid_list = []
        self.data_dir = data_dir
        self.num_class = num_class

        for item in glob.glob(self.data_dir+"*.npy"):
            self.sid_list.append(item.split('/')[-1])

        self.sid_list.sort()
        self.sid_list = np.asarray(self.sid_list)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        train_index, valid_index = list(kf.split(self.sid_list))[fold]
        print("Fold:", fold)
        if mode=="train":
            self.sid_list = self.sid_list[train_index]
        else:
            self.sid_list = self.sid_list[valid_index]
        print("Dataset size:", len(self))

        self.class_label_dict = dict()
        if self.num_class > 0: # conditional
            FILE = open("class_label.csv", "r")
            FILE.readline() # header
            for myline in FILE.readlines():
                mylist = myline.strip("\n").split(",")
                self.class_label_dict[mylist[0]] = int(mylist[1])
            FILE.close()

    def __len__(self):
        return len(self.sid_list)

    def __getitem__(self, idx):
        img = np.load(self.data_dir+self.sid_list[idx])
        class_label = self.class_label_dict.get(self.sid_list[idx], -1) # -1 if no class label
        return img[None,:,:,:], class_label

class Volume_Dataset_MRI(Dataset):

    def __init__(self, data_dir, mode='train', fold=0, num_class=0):
        #self.sid_list = []
        self.data_dir = data_dir
        self.num_class = num_class

        self.paths = sorted(make_dataset(self.data_dir,  data_type='mri_3d', walk=True))
        self.paths.sort()
       # pdb.set_trace()
        #kf = KFold(n_splits=5, shuffle=True, random_state=0)
        #train_index, valid_index = list(kf.split(self.paths))[fold]
        #train_index = 
        #print("Fold:", fold)
        train_index = int(len(self.paths)*0.8)
        if mode=="train":
            self.paths = self.paths[:train_index]
        else:
            self.paths = self.paths[train_index:]
        print("Dataset size:", len(self.paths))


    def __len__(self):
        return len(self.paths)
    
    def _preprocess_img(self, img):

        
        img = torch.tensor(img)

        img = SpatialPad(spatial_size=(256, 256, 256))(img.unsqueeze(0)).squeeze()
        img = CenterSpatialCrop(roi_size=(256, 256, 256))(img.unsqueeze(0)).squeeze()

            
        img = img/torch.max(img)
        img = (img - 0.5)*2
        img = img

        return img


    def __getitem__(self, idx):

        img_path = self.paths[idx]
        img = sitk.ReadImage(img_path)
        img = sitk.GetArrayFromImage(img) 
        img = self._preprocess_img(img)
        class_label = -1
        return img[None,:,:,:], class_label
    

if __name__ == '__main__':
    
    import ants
    trainset = Volume_Dataset_MRI(data_dir='/home1/yujiali/dataset/brain_MRI/ADNI/T1/aligned_brain_MNI', \
                                  fold=1, num_class=0)
    img1 = trainset[1][0]
    real_images_crop = img1[:,50:-70,:,:].cpu().numpy().squeeze()
    ants.from_numpy(real_images_crop).to_file('./crop_test2.nii.gz')
    