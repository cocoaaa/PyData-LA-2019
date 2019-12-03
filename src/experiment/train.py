from pathlib import Path
import numpy as np
import random
from typing import List, Union, Collection, Callable, Any, NewType, TypeVar, Optional

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
from utils import to_device, assert_mean_reduction
from metrics import runningScore, averageMeter


def train_one_epoch(model, train_dl, optimizer, loss_fn, device, metric_fns=None):
    """
    Returns epoch's average loss (ie. loss per sample) and other metrics
    """
    to_device(model, device)
    model.train()
    result = {}
    
    # Average loss from the training this epochs
    epoch_loss = 0.
    sample_count = 0
    for sample in train_dl:
        x, y = sample['X'].to(device), sample['y'].to(device)
        y_pred = model(x)
        loss_mean = loss_fn(y_pred, y)

        # Backprop
        optimizer.zero_grad()
        loss_mean.backward()
        optimizer.step()

        epoch_loss += loss_mean.item() * len(x)
        sample_count += len(x)

    # Collect epoch stats
    epoch_loss /= sample_count
    result['epoch_loss'] = epoch_loss
    ## add other metrics
    # for mname, metric_fn in metric_fns.items():
    # ...
    
    return result


def train(n_epochs, model, optimizer, loss_fn, train_dl, tbw, losses=None, verbose=False):
    """
    Assumes model and optmizer are properly linked, 
    ie. optimizer's parameters == model.parameters()
    - loss_fn should use the loss averaged over the mini-batch
    
    - If losses is not None, append each epoch's loss to the input `losses`. 
        Otherwise, create a new empty list and append to it.
        
    - tbw (TBWrapper of tensorboard.writer.SummaryWriter)
    """
    assert_mean_reduction(loss_fn)
    if losses is None: losses = []
        
    # Log weight and bias before new trianing runs
    global_ep = len(losses) 
    log_first_grad = (global_ep > 0)
 
    if verbose:
        print('Start training at ep: ', global_ep)
        print('Logging to ep idx: ', global_ep)
    tbw.log_params_hist(model, global_ep, log_grad=log_first_grad)
    global_ep += 1

    for ep in range(n_epochs):
        
        # Run train for one epoch and collect epoch stats
        epoch_result = train_one_epoch(model, optimizer, loss_fn, train_dl, device)
        epoch_loss = epoch_result['epoch_loss']
        losses.append(epoch_loss)
        
        # Log to tensorboard
#         print('Logging to ep idx: ', global_ep)
        tb.add_scalar('train/loss', epoch_loss, global_ep)
        
        ## Log weight
        tbw.log_params_hist(model, global_ep, log_grad=True)
        global_ep += 1
        
        if verbose:
            print('number of samples in this epoch: ', sample_count)
        ## Log train and val (and optionally test) metrics
        ## todo
            
    return losses