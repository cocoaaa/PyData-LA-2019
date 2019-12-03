from pathlib import Path
import numpy as np
import random
from typing import Iterable, Dict, List, Optional, Tuple

from skimage import io as skiio
import PIL

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from helpers import load_txt
import pdb 


class TabularDataset(Dataset):
    """
    Tabular dataset
    Args:
    data (dict): with keys X for data_x and y for target floats
    
    eg: 
    
    A dataset for linear regression
    | x1 | x2 | y |
    | ---| ---| --- |
    | 10 | 20 | 12.5 |
    | 30 | 25 | 14.5 |
    """
    def __init__(self, data):
        self.X = data["X"]
        self.y = data["y"]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = {'X': self.X[idx], 'y': self.y[idx]}
        return sample

def get_2d_ds(n_samples):
    pass


class SegDataset(Dataset):
    """
    Segmentation task dataset
    - x: input image
    - y: label image
    """
    
    def __init__(self, 
                 ids_file,
                 x_dir,
                 y_dir,
                 get_label=True,
                 x_suffix='.jpg',
                 y_suffix='.png',
                 transforms=None,
                 seed=None,
                verbose=False):
        """
        Args:
        - ids_file: path to a text file containing (only) samples' ids in each line
            - eg: '0000\n0201\n0299'
        - x_dir (str or Path): path to the directory containing data images (x)
        - y_dir (str or Path): path to the directory containing label images (y)
        - get_label (bool): if True, returns `y` (as np.array of shape (h,w,1) in addition to `x`
        - transform (tuple or None): a tuple of transforms to apply to `x` and `y`, respectively
        """
        
        # Read data line by line
        self.ids = load_txt(ids_file)

        self.x_dir = Path(x_dir) if not isinstance(x_dir, Path) else x_dir
        self.y_dir = Path(y_dir) if not isinstance(y_dir, Path) else y_dir
        self.x_suffix, self.y_suffix = x_suffix, y_suffix
        self.get_label = get_label
        self.transforms = transforms
        self.seed = seed
        
        if verbose:
            print("Created dataset: ", len(self.ids),'[', self.ids[0], self.ids[-1],']')
            print('\txdir: ', self.x_dir)
            print('\tydir: ', self.y_dir)
            print('\tnumber of ids: ', len(self.ids))

    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        """
        Returns a tuple of two tensors.
        - sample (tuple): (torch.tensor, None) or (torch.tensor, torch.tensor) depending on `get_label` 
                        at initialization
        First element is 3 dim tensor for RGB of shape (nC, h, w) whose values are in range [0,1].
        If `get_label`, the second element is two-dim tensor of shape (h,w) of `uint8` whose values 
        indicate the label at the pixel location.           
        """
        
        x_fn = self.x_dir/(self.ids[idx]+self.x_suffix)
        x = skiio.imread(x_fn) #np.array
        y_fn, y = None, None
        
        if self.get_label:
            y_fn = self.y_dir/(self.ids[idx]+self.y_suffix)
            y = skiio.imread(y_fn)[:,:,0] #  Take only a single color channel; two-dim np.array
        
        if self.transforms:
            seed = self.seed or random.randint(0,2**32)
            random.seed(seed)
            x = self.transforms[0](x)
            
            if self.get_label:
                random.seed(seed)
                y = self.transforms[1](y)
            
        if not isinstance(x, torch.Tensor):
            if isinstance(x, PIL.Image.Image):
                x = np.asarray(x)
                
            if not isinstance(x, np.ndarray):
                raise ValueError('x must be a numpy array at this point', type(x))
            # numpy -> tensor
            x = transforms.ToTensor()(x) #reorders the axes and remaps values to range [0,1] from [0,255]
            
        if self.get_label and not isinstance(y, torch.Tensor):
            if isinstance(y, PIL.Image.Image):
                y = np.asarray(y)
            if not isinstance(y, np.ndarray):
                raise ValueError('y must be a numpy array at this point', type(y))
                
            y = torch.from_numpy(y)

        # To remove later
        assert x.ndim == 3, f'x dim must be 3: {x.ndim}'
        if self.get_label:
            assert y.ndim == 2, f'y dim must be 2: {y.ndim}'
            assert y.dtype == torch.uint8, f'y.dtype must in torch.uint8: {y.dtype}'
        
        if self.get_label:
            return (x,y)
        else:
            return x


class ImageDataset(Dataset):
    def __init__(self, imgs, labels=None, get_label=False,
                 transforms=None,
                 seed=None):
        """
        :param imgs (np.ndarray): a collection of images as np.ndarray
        :param labels: if labels is given, get_label is overwritten to be always True
        :param get_label: returns both x and y if True
        :param transforms: None or Dict[str, transforms]: a dictionary of x and y transforms
        :param seed: None or int
        """
        assert imgs.ndim in [3, 4]
        self.imgs = imgs
        self.labels = labels
        if self.labels is not None:
            get_label = True
        self.get_label = get_label
        self.transforms = transforms
        self.seed = seed

        if labels is not None:
            assert len(self.imgs) == len(self.labels), f"len(imgs) must equal to len(labels): {len(imgs), len(labels)}"

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Returns a sample as a dictionary with keys 'x' and/or 'y' (depending on self.get_label)

        The convention of returning a dictionary conforms to the default `collate_fn` of `torch.utils.data.DataLoader`
        Refer to https://pytorch.org/docs/stable/data.html
        """
        x = self.imgs[idx]
        if x.ndim == 2:
            x = x[None, :, :]

        if self.transforms is not None:
            seed = self.seed or random.randint(0, 2 ** 32)
            random.seed(seed)
            x = self.transforms['x'](x)

            if self.get_label:
                y = self.labels[idx]
                random.seed(seed)
                try:
                    y = self.transforms['y'](y)
                except KeyError as e:
                    print(e)
                    print('Using the same transforms as x')
                    y = self.transforms['x'](y)
                return {'x':x, 'y':y}
        return {'x':x}


##Test
# def test_imgdataset():
#   TRAIN_DATA_DIR = Path('/Users/hayley/Workspace/Class/CS669-RL/hw2/nbs/../data/train')
#   train_x_ds = ImgDataset(
#       image_ids_file = TRAIN_DATA_DIR/'train_ids.txt', 
# #       image_dir = TRAIN_DATA_DIR/'images',
#       suffix='.jpg',
#       is_label=False,
#       transform=x_tsfm
#   )

#   train_y_ds = ImgDataset(
#       image_ids_file = TRAIN_DATA_DIR/'train_ids.txt',
#       image_dir = TRAIN_DATA_DIR/'tf_segmentation',
#       suffix='.png',
#       is_label=True,
#       transform=y_tsfm
#   )

#   ## Validation dataset
#   val_x_ds = ImgDataset(
#       image_ids_file = TRAIN_DATA_DIR/'val_ids.txt',
#       image_dir = TRAIN_DATA_DIR/'images',
#       suffix='.jpg',
#       is_label=False)

#   val_y_ds = ImgDataset(
#       image_ids_file = TRAIN_DATA_DIR/'val_ids.txt',
#       image_dir = TRAIN_DATA_DIR/'tf_segmentation',
#       suffix='.png',
#       is_label=True
#   )
  
#   # train data viz
#   idx = 11
#   for idx in range(0,200,20):
#       train_x = train_x_ds[idx]
#       train_y = train_y_ds[idx]
#       print(np.unique(np.asarray(train_y)))
#       display(train_x)
#       display(train_y)     
      
# def test_segdataset():
#   train_ds = SegDataset(
#       ids_file=TRAIN_DATA_DIR/'train_ids.txt', 
#       x_dir=TRAIN_DATA_DIR/'images',
#       y_dir=TRAIN_DATA_DIR/'tf_segmentation',
#       verbose=True
#   )

# Convenience 
