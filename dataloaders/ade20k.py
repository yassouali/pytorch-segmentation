from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import torch
import cv2
from PIL import Image
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms


class ADE20KDataset(BaseDataSet):
    """
    ADE20K dataset 
    http://groups.csail.mit.edu/vision/datasets/ADE20K/
    """
    def __init__(self, **kwargs):
        self.num_classes = 150
        self.palette = palette.ADE20K_palette
        super(ADE20KDataset, self).__init__(**kwargs)

    def _set_files(self):
        if self.split in  ["training", "validation"]:
            self.image_dir = os.path.join(self.root, 'images', self.split)
            self.label_dir = os.path.join(self.root, 'annotations', self.split)
            self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]
        else: raise ValueError(f"Invalid split name {self.split}")
    
    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.jpg')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(Image.open(label_path), dtype=np.int32) - 1 # from -1 to 149
        return image, label, image_id

class ADE20K(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):

        self.MEAN = [0.48897059, 0.46548275, 0.4294]
        self.STD = [0.22861765, 0.22948039, 0.24054667]

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

        self.dataset = ADE20KDataset(**kwargs)
        super(ADE20K, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)



class ADE20KTest(Dataset):
    def __init__(self, root, crop_size = None):
        self.MEAN = [0.28689529, 0.32513294, 0.28389176]
        self.STD = [1., 1., 1.]
        self.num_classes = 150
        self.crop_size = crop_size
        self.root = root

        self.image_dir = os.path.join(self.root, 'images', 'testing')
        self.files = [os.path.basename(path).split('.')[0] for path in glob(self.image_dir + '/*.jpg')]

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(self.MEAN, self.STD)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_path = self.files[index]
        image = np.asarray(Image.open(image_path).convert('RGB'), dtype=np.float32)
        image_id = os.path.splitext(os.path.basename(image_path))[0]

        if self.crop_size:
            h, w, _ = image.shape
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            # Center Crop
            h, w, _ = image.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]

        image = Image.fromarray(np.uint8(image))
        return self.normalize(self.to_tensor(image)), image_id    

