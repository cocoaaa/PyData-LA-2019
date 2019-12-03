from pathlib import Path
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pdb 
from tqdm.autonotebook import tqdm

# My libs
# from metrics import runningScore, averageMeter

def evaluate(model, val_dl, loss_fn, device, metric_fns=None):
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

    # Put model on the right device
    model = model.to(device)
    
    # Eval
    model.eval()
    
    total_loss = 0.
    n_samples = 0
    with torch.no_grad():
        for sample in tqdm(val_dl, desc='eval-iter'): #tqdm(range(max_epoch), desc='Epoch')
            x, y = sample['X'], sample['y']
            x,y = x.to(device), y.to(device)
            pred_y = model(x)

            # Eval loss
            loss_mean = loss_fn(pred_y, y)
            total_loss += loss_mean * len(x)
            n_samples += len(x)

            # Eval metrics
    
    result = {'eval_loss': total_loss/n_samples}
    # add  other metrics to result

    
    return result
          