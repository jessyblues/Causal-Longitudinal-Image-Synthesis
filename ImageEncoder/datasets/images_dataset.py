from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import SimpleITK as sitk
import torch
import pdb


class ImagesDataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		from_path = self.source_paths[index]
		from_im = Image.open(from_path)
		from_im = from_im.convert('RGB')

		to_path = self.target_paths[index]
		to_im = Image.open(to_path).convert('RGB')
		if self.target_transform:
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im

		return from_im, to_im

class MRI_Dataset(Dataset):

	def __init__(self, source_root, target_root, opts, target_transform=None, source_transform=None):
		self.source_paths = sorted(data_utils.make_dataset(source_root))
		self.target_paths = sorted(data_utils.make_dataset(target_root))
		self.source_transform = source_transform
		self.target_transform = target_transform
		self.opts = opts

	def __len__(self):
		return len(self.source_paths)

	def __getitem__(self, index):
		
		from_path = self.source_paths[index]

		from_im = sitk.ReadImage(from_path)
		from_im = sitk.GetArrayFromImage(from_im) 
		from_im = from_im/torch.max(torch.tensor(from_im))
		from_im = from_im.unsqueeze(0)

		to_path = self.target_paths[index]
		
		to_im = sitk.ReadImage(to_path)
		to_im = sitk.GetArrayFromImage(to_im) 
		to_im = to_im/torch.max(torch.tensor(to_im))
		to_im = to_im.unsqueeze(0)


		if self.target_transform:
			#print(to_im.shape)
			to_im = self.target_transform(to_im)

		if self.source_transform:
			from_im = self.source_transform(from_im)
		else:
			from_im = to_im.clone()

		return from_im, to_im
