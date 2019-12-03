import os,sys, time
from pathlib import Path
import numpy as np
import random
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

def train(model, train_dl, loss_fn, optimizer, lr_scheduler, device, params, memo=[]):
    """
    Good references:
    - https://is.gd/EP8LKv
    - use yaml file for config parsing: [pytorch-semseg](https://is.gd/Gbcq4H)
    
    Args:
    - train_dl (DataLoader): 
        - training dataloader
    - lr_scheduler (TriangleLR, ConstLR, or None)
    - params (dict)
    
    Returns:
    - result(dict) : {'loss': val_loss, 'acc': val_acc}
    
    ## todo: check if model's parameters and optimizers are in sync -- 'sync' meaning 
    ## in the same device
    """
    # Get parameters
    n_classes = params.get('n_classes', 21)
    ignore_idx = params.get('ignore_idx', 255)
    ignore_bg = params.get('irgnore_bg', True)
    debug = params.get('debug', False)

    # Put model in the right device, mode
#     model = model.to(device)
    model.train()
    
    # Initialize metrics
    running_metrics = runningScore(n_classes)
    loss_meter = averageMeter()
    acc_meter = averageMeter()
    train_losses, train_accs = [], []
    n_correct, n_samples = 0, 0
    
    start = time.time()
    for batch_i, (batch_x, batch_y) in enumerate(tqdm(train_dl, desc='train-iter')):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

        # Model forward computation
        pred_y = model(batch_x)
        loss = loss_fn(pred_y, batch_y)
        train_losses.append(loss.item())
        loss_meter.update(loss.item())
        
        # Set the optimizer's lr according to the lr_scheduler
        ## RESUME HERE -- DEBUG (memo returned has length that doesn't make sense)
        lr_scheduler.step()
        memo.append(optimizer.param_groups[0]['lr']) #to remove
#         print('iter: ', len(memo), '-- lr : ', optimizer.param_groups[0]['lr'])
        # Gradient backprop: update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Train Stats
        _, pred_label = torch.max(pred_y, 1)

        mask = (batch_y != ignore_idx)

        if debug:
            print('n pixels in 0,...,K-1: ', mask.sum().item()) #divide by x.shape[0]
            print('mask for batch_y shape (should be 3 dim): ', mask.shape)

        if ignore_bg:
            mask = mask & (batch_y != 0)

        if debug:
            print('ignore background? :', ignore_bg)
            print('mask for batch_y shape (should be 3 dim): ', mask.shape)
            print('n pixels in 1,...,K-1: ', mask.sum().item())
            pdb.set_trace()
            
        running_metrics.update(batch_y.data.cpu().numpy(), pred_label.cpu().numpy())

        n_correct = (pred_label[mask] == batch_y[mask]).sum().item()
        n_valid_pixels = mask.sum().item()
        acc = n_correct / n_valid_pixels
        acc_meter.update(acc)
        train_accs.append(acc)
        

    # Log after each train epoch
    end = time.time()

    # debug
    if debug: 
        print('Train')
        print(f'avg acc from averageMeter: {acc_meter.avg:.9f}')
        print(f'avg loss from averageMeter: {loss_meter.avg:.9f}')
        print('scheduler step count: ', lr_scheduler.step_count)


    result = {'running_metrics': running_metrics,
              'loss_meter': loss_meter,
              'acc_meter': acc_meter,
             'train_losses': train_losses,
             'train_accs': train_accs,
             }
    
    return result

def train_for_n_iters(model, train_dl, loss_fn, optimizer, lr_scheduler, device, params):
    """
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
    n_iters = params.get('n_iters')
    n_classes = params.get('n_classes', 21)
    ignore_idx = params.get('ignore_idx', 255)
    ignore_bg = params.get('irgnore_bg', True)
    debug = params.get('debug', False)

    # Put model in the right device, mode
    model = model.to(device)
    model.train()
    
    # Initialize metrics
    running_metrics = runningScore(n_classes)
    loss_meter = averageMeter()
    acc_meter = averageMeter()
    train_losses, train_accs = [], []
    n_correct, n_samples = 0, 0
    iter_idx = 0

    pbar = tqdm(total=n_iters)
    start = time.time()
    while True:
        for batch_i, (batch_x, batch_y) in enumerate(train_dl):
            
            # Return 
            if iter_idx >= n_iters:
                # Log before returning
                end = time.time()

                # debug
                if debug: 
                    print('Train')
                    print(f'avg acc from averageMeter: {acc_meter.avg:.9f}')
                    print(f'avg loss from averageMeter: {loss_meter.avg:.9f}')

                result = {'running_metrics': running_metrics,
                          'loss_meter': loss_meter,
                          'acc_meter': acc_meter,
                         'train_losses': train_losses,
                         'train_accs': train_accs}
                pbar.close()
                return result

            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()

            # Model forward computation
            pred_y = model(batch_x)
            loss = loss_fn(pred_y, batch_y)
            train_losses.append(loss.item())
            loss_meter.update(loss.item())

            # Set the optimizer's lr according to the lr_scheduler
            lr_scheduler.step()

            # Gradient backprop: update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            iter_idx += 1
            pbar.update(1)

            # Train Stats
            _, pred_label = torch.max(pred_y, 1)

            mask = (batch_y != ignore_idx)

            if debug:
                print('n pixels in 0,...,K-1: ', mask.sum().item()) #divide by x.shape[0]
                print('mask for batch_y shape (should be 3 dim): ', mask.shape)

            if ignore_bg:
                mask = mask & (batch_y != 0)

            if debug:
                print('ignore background? :', ignore_bg)
                print('mask for batch_y shape (should be 3 dim): ', mask.shape)
                print('n pixels in 1,...,K-1: ', mask.sum().item())
                pdb.set_trace()

            running_metrics.update(batch_y.data.cpu().numpy(), pred_label.cpu().numpy())

            n_correct = (pred_label[mask] == batch_y[mask]).sum().item()
            n_valid_pixels = mask.sum().item()
            acc = n_correct / n_valid_pixels
            acc_meter.update(acc)
            train_accs.append(acc)
        

