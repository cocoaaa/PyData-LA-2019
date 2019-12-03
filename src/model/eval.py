import os,sys
from pathlib import Path
import numpy as np
import random
# import skimage as ski
from skimage import io as skiio
import PIL
from helpers import load_txt
import pdb 

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import holoviews as hv
# DEVICE =dd torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def infer_on_train(trained, dataset, idx, device,
         return_hv=False):
    """
    Run inference on the `idx`th sample from dataloader using `trained` model
    
    Args:
    - dataset (Dataset): returns a (batch of) x
    - idx (int): index into the dataset samples
    
    Returns 
    - `Holoviews` layout object if `return_hv` is True else None
    """
    trained = trained.to(device)
    trained.eval()

    with torch.no_grad():
        x,y = dataset[idx]
        print(x.shape, y.shape)
        x,y = x.to(device), y.to(device)
        x.unsqueeze_(0)
        y.unsqueeze_(0)
        print(x.shape, y.shape)
        pred_y = trained(x)
        _, pred_label = torch.max(pred_y, 1)
        pred_label_np = pred_label.cpu().numpy().squeeze()
        x_np = x.squeeze().cpu().numpy().transpose(1,2,0)
        y_np = y.squeeze().cpu().numpy()
        print(f'pred_label shape: {pred_label.shape}')
        print(f'unique labels: ', np.unique(pred_label_np))
        
    if return_hv:
        overlay_gt = hv.RGB(x_np) * hv.Image(y_np, group='mask').opts(axiswise=True)
        overlay_pred = hv.RGB(x_np) * hv.Image(pred_label_np, group='mask').opts(axiswise=True)
        overlay_masks =  (
            hv.Image(y_np *(y_np <= 21), group='mask').opts(axiswise=True, shared_axes=False) 
            + hv.Image(pred_label_np, group='mask').opts(axiswise=True, shared_axes=False)
        )
#         return overlay1.opts(shared_axes=False)
#         return (overlay_gt+overlay_pred).opts(shared_axes=False)
        return (overlay_masks)
    
    else:
        return pred_label_np
                

def infer_on_test(trained, dataset, idx, device,
         return_hv=False):
    """
    Test `idx`th sample from dataloader using `trained` model
    
    Args:
    - dataset (Dataset): returns a (batch of) x
    - idx (int): index into the dataset samples
    
    Returns:
    - pred_label_np (np.ndarray): (H,W) shaped np array for predicition mask 
    - if return_hv is True: 
        - returns the `Holoviews` layout object of the x, prediciton mask images
    """
    trained = trained.to(device)
    trained.eval()

    with torch.no_grad():
        x = dataset[idx]
        print(x.shape)
        x = x.to(device)
        x.unsqueeze_(0)
        print(x.shape)
        pred_y = trained(x)
        _, pred_label = torch.max(pred_y, 1)
        pred_label_np = pred_label.cpu().numpy().squeeze()
        x_np = x.squeeze().cpu().numpy().transpose(1,2,0)

        print(f'pred_label shape: {pred_label.shape}')
        print(f'unique labels: ', np.unique(pred_label_np))

    if return_hv:
        overlay = hv.RGB(x_np) * hv.Image(pred_label_np, group='mask')
#         display(overlay.opts(shared_axes=False))
        return overlay

    else:
        return pred_label_np
    
def predict(model, np_imgs, device):
    """
    Test `idx`th sample from dataloader using `trained` model
    
    Args:
    - dataset (Dataset): returns a (batch of) x
    - idx (int): index into the dataset samples
    
    Returns:
    - pred_label_np (np.ndarray): (H,W) shaped np array for predicition mask 
    - if return_hv is True: 
        - returns the `Holoviews` layout object of the x, prediciton mask images
    """
    model = model.to(device)
    model.eval()
    np_imgs = np.array(np_imgs) #swap axes orders
    x = torch.from_array(np_imgs.transpose((0, -1, -2, -3)))
    pirnt(x.shape)
#     assert x.ndim == 4
    
    with torch.no_grad():
        x = x.to(device)
        pred_scores = model(x) # (bs, n_classes, h, w)
        pred_scores_np = pred_scores.cpu().numpy().transpose((0, -2, -1, -3))
        print(f'pred_scores shape: {pred_scores.shape}')

    return pred_scores_np

def get_pred_masks(trained, dataloader, device, show=False):
    """
    Run inference on all data from iterated via dataloader
    
    Args:
    - dataloader (SegDataset): must be a SegDataset object with `get_label=False`
    
    Returns:
    - preds (n_tests, H, W, n_test ) numpy array, representing predicted label masks
    """

    trained = trained.to(device)
    trained.eval()

    preds_np = []
    with torch.no_grad():
        for x in dataloader:
#             print(x.shape) 
            x = x.to(device)
            pred_y = model(x) #input x should be 4dim: (BS=1, n_classes=21, H,W)
            _, pred_label = torch.max(pred_y, 1) #output would be 3dim (BS=1, H,W)
            pred_label_np = pred_label.cpu().numpy().squeeze()
            preds_np.append(pred_label_np)
            
            
            if show:
                ## Visualize
                x_np = x.squeeze().cpu().numpy().squeeze().transpose(1,2,0)
            
                print(f'pred_label shape: {pred_label.shape}')
                print(f'unique labels: ', np.unique(pred_label_np))
                overlay = hv.RGB(x_np) * hv.Image(pred_label_np, group='mask')
                display(overlay.opts(shared_axes=False))
#                 pdb.set_trace()
    return np.asarray(preds_np)

