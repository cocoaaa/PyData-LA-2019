import torch
from torch import optim
from torch import nn
import os, sys, time
import numpy as np
from pathlib import Path
import joblib, pdb
from pathlib import Path

import torch 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from helpers import get_pad_size

from PIL import Image

import pdb 

# Globals
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_SIZE = 224


class To3DimNpImage():
    """
    If the input np.ndarray is 2dimensional, append a new dimension to the first dim
    so that the output np.ndarray has the shape of (1, H, W)
    """

    def __call__(self, arr):
        assert isinstance(arr, np.ndarray), f"{arr} must be np.nadarray: {type(arr)}"
        assert arr.ndim in [2, 3], f"{arr} must be 2 or 3 dim: {arr.ndim}"
        if arr.ndim == 2:
            arr = arr[None, :, :]
        return arr


class ToFloatNpImage():
    def __call__(self, arr):
        return np.array(arr/255., np.float32)
    
class TensorDivider():
    def __init__(self, factor):
        self.factor=factor
        
    def __call__(self, t):
        return self/t

class ToIntNpImage():
    def __call__(self,arr):
        return np.uint8(arr * 255)
        
class TailPadder():
    """Pad the input PIL.Image to the out_size 
    - if out_size is int, out_size = out_w = out_h
    - else: out_size = (out_h, out_w)
    
    Args:
        - out_size (int or tuple)
        
    Calls:
        - acts on PIL.Image 
        - returns a (padded) PIL.Image
    """
    def __init__(self, out_size, fill=0, padding_mode='constant'):
        assert isinstance(out_size, (int, tuple)), "out_size must be int or tuple"
        self.out_size = out_size
        self.fill = fill
        self.mode = padding_mode
        
    def __call__(self, pil_img):
        out_h, out_w = self.out_size if isinstance(self.out_size, tuple) else (self.out_size, self.out_size)
        ny_pad, nx_pad = get_pad_size(pil_img.height, pil_img.width, (out_h, out_w))
        return transforms.Pad((0,0,nx_pad, ny_pad), self.fill, self.mode)(pil_img)


def get_transforms(resize_x, resize_y,
                   random_rot_degree=20, pad_out_size=500, fill=255,
                   brightness=0.2, contrast=0.2, saturation=0.1, hue=0,
                  channel_means=IMAGENET_MEAN, channel_stds=IMAGENET_STD):
    """
    resize_x : size of x to resize the original input data x
    resize_y : size of y to resizes the label mask y
    """
    
    tr_x_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        TailPadder(pad_out_size, fill=fill),
        transforms.Resize(resize_x, interpolation=Image.BILINEAR),
         # Data augs
         transforms.RandomRotation(random_rot_degree, fill=fill),
         transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
         transforms.ToTensor(),
         transforms.Normalize(mean=channel_means, std=channel_stds),
    ])

    tr_y_tsfm = transforms.Compose([
        lambda np_2d: transforms.ToPILImage()(np_2d[:,:,None]),
        TailPadder(pad_out_size, fill=fill),
        transforms.Resize(resize_y, interpolation=Image.NEAREST),

         # Data augs' geometric transforms only 
         transforms.RandomRotation(random_rot_degree, fill=fill),
         lambda pil_y: torch.from_numpy(np.asarray(pil_y))
    ])

    val_x_tsfm = transforms.Compose([
        transforms.ToPILImage(),
        TailPadder(pad_out_size, fill=fill),
        transforms.Resize(resize_x, interpolation=Image.BILINEAR),

         transforms.ToTensor(),
         transforms.Normalize(mean=channel_means, std=channel_stds),
    ])

    val_y_tsfm = transforms.Compose([
        lambda np_2d: transforms.ToPILImage()(np_2d[:,:,None]),
        TailPadder(pad_out_size, fill=fill),
        transforms.Resize(resize_y, interpolation=Image.NEAREST),

         lambda pil_y: torch.from_numpy(np.asarray(pil_y))
    ])
    
    return {'train': {'x': tr_x_tsfm, 'y': tr_y_tsfm},
            'val': {'x': val_x_tsfm, 'y': val_y_tsfm}}
                      
