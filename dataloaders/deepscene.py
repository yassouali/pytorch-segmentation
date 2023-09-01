# Originally written by Dustin Franklin, adapted by Markus Schiffer
# https://github.com/dusty-nv/pytorch-segmentation/blob/master/datasets/deepscene.py

from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import re
from PIL import Image


class DeepSceneDataset(BaseDataSet):
	"""
	DeepScene Freibrug Forest dataset
	http://deepscene.cs.uni-freiburg.de/
	"""
	def __init__(self, **kwargs):
		self.num_classes = 7
		self.palette = palette.DeepScene_palette

		self.mask_mapping = {}

		for i in range(0, len(self.palette), 3):
			self.mask_mapping[tuple(self.palette[i:i+3])] = i // 3
		
		self.images = []
		self.targets = []

		super(DeepSceneDataset, self).__init__(**kwargs)

	def gather_images(self, images_path, labels_path):
		def sorted_alphanumeric(data):
			convert = lambda text: int(text) if text.isdigit() else text.lower()
			alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
			return sorted(data, key=alphanum_key)

		image_files = sorted_alphanumeric(os.listdir(images_path))
		label_files = sorted_alphanumeric(os.listdir(labels_path))

		if len(image_files) != len(label_files):
			print('warning:  images path has a different number of files than labels path')
			print('   ({:d} files) - {:s}'.format(len(image_files), images_path))
			print('   ({:d} files) - {:s}'.format(len(label_files), labels_path))
			
		for n in range(len(image_files)):
			image_files[n] = os.path.join(images_path, image_files[n])
			label_files[n] = os.path.join(labels_path, label_files[n])
			
		return image_files, label_files
	
	def _set_files(self):
		if self.split in ["training", "validation"]:
			
			if self.split == "training":
				train_images, train_targets = self.gather_images(os.path.join(self.root, 'train/rgb'),
											    	    os.path.join(self.root, 'train/GT_color'))
				
				self.images.extend(train_images)
				self.targets.extend(train_targets)

			elif self.split == "validation":
				val_images, val_targets = self.gather_images(os.path.join(self.root, 'test/rgb'),
											     os.path.join(self.root, 'test/GT_color'))

				self.images.extend(val_images)
				self.targets.extend(val_targets)
			
			self.files = self.images

		else: raise ValueError(f"Invalid split name {self.split}")
	
	def _load_data(self, index):
		image_id = self.images[index]
		image = np.asarray(Image.open(self.images[index]).convert("RGB"), dtype=np.float32)
		target_rgb = np.asarray(Image.open(self.targets[index]).convert("RGB"), dtype=np.float32)
		target = np.zeros(target_rgb.shape[:2], dtype=np.int32)
		for cls in self.mask_mapping:
			target[(target_rgb == cls).all(axis=2)] = self.mask_mapping[cls]
		return image, target, image_id

class DeepScene(BaseDataLoader):
	def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
		
		self.MEAN = [0.485, 0.456, 0.406]
		self.STD = [0.229, 0.224, 0.225]

		kwargs = {
			'root': data_dir,
			'split': split,
			'mean': self.MEAN,
			'std': self.STD,
			'augment': augment,
			'crop_size': crop_size,
			'base_size': base_size,
			'scale': scale,
			'flip': flip,
			'blur': blur,
			'rotate': rotate,
			'return_id': return_id,
			'val': val
		}

		self.dataset = DeepSceneDataset(**kwargs)
		super().__init__(self.dataset, batch_size, shuffle, num_workers, val_split)
