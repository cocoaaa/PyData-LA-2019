import os,sys, time
from pathlib import Path
import numpy as np
import random
import holoviews as hv
from skimage import io as skiio
import PIL

import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pdb 
from tqdm.autonotebook import tqdm


# My libs
from helpers import load_txt, now2str, append2file
from metrics import runningScore, averageMeter

def evaluate(model, val_dl, loss_fn, device, params={}):
    """
    Evaluate the model on val_dl using the input loss_fn and specified parameters
    on `device`
    
    Good references:
    - https://is.gd/EP8LKv
    - use yaml file for config parsing: [pytorch-semseg](https://is.gd/Gbcq4H)
    
    Args:
    - val_dl (DataLoader): 
        - validation dataloader
        
    - params (dict)
    
    Returns:
    - result(dict) : {'loss': val_loss, 'acc': val_acc}
    """
    
    # Get parameters
    ignore_idx = params.get('ignore_idx', 255)
    ignore_bg = params.get('irgnore_bg', True)
    debug = params.get('debug', False)
    n_classes = params.get('n_classes', 21)
    
    # Put model on the right device
    model = model.to(device)
    start = time.time()
   
    # Eval
    model.eval()
    
    ## Initialize metrics
    running_metrics = runningScore(n_classes)
    loss_meter = averageMeter()
    acc_meter = averageMeter()
    
    with torch.no_grad():
        for x,y in tqdm(val_dl, desc='eval-iter'): #tqdm(range(max_epoch), desc='Epoch')
            x,y = x.to(device), y.to(device)
            pred_y = model(x)

            # Eval loss
            loss = loss_fn(pred_y, y.long())
            loss_meter.update(loss.item())

            # Eval metrics
            _, pred_label = torch.max(pred_y, 1)

            mask = y != ignore_idx
            if debug:
                print('n pixels in 0,...,K-1: ', mask.sum().item())
            if ignore_bg:
                mask = mask & (y != 0)
            if debug:
                print('ignore background? :', ignore_bg)
                print('n pixels in 1,...,K-1: ', mask.sum().item())
                display(
                    hv.Image(mask.cpu().numpy().squeeze(), label='mask')
                    * hv.Image(y.cpu().numpy().squeeze(), label='gt y')
                    + hv.Image(pred_label.cpu().numpy().squeeze(), label='pred_label')
                )
            n_correct = (pred_label[mask] == y[mask].long()).sum().item()
            n_valid_pixels = mask.sum().item()
            acc = n_correct / n_valid_pixels
           
            if debug:
                # debugging
                print('n_valid_pixels: ', n_valid_pixels)
                print('n_correct: ', n_correct)
                pdb.set_trace()
            
            running_metrics.update(y.cpu().numpy(), pred_label.cpu().numpy())
            acc_meter.update(acc)
    # Log evaluation for this epoach
    
    # debug
    if debug: 
        print(f'avg acc from averageMeter: {acc_meter.avg:.9f}')
        print(f'avg loss from averageMeter: {loss_meter.avg:.9f}')
    
    result = {'running_metrics': running_metrics,
              'loss_meter': loss_meter,
              'acc_meter': acc_meter}
        
    return result
          