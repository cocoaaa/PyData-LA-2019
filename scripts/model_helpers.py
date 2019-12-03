import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os,sys
import pdb

################################################################################
### Path setup
################################################################################
PROJ_ROOT = Path(os.getcwd()).parent;print(PROJ_ROOT)
DATA_DIR = PROJ_ROOT/'data/raw'
SRC_DIR = PROJ_ROOT/'scripts'
paths2add = [PROJ_ROOT, SRC_DIR]

# Add project root and src directory to the path
for p in paths2add:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        print("Prepened to path: ", p)
        
        
################################################################################
### Helpers
################################################################################
#todo: rename it to "unfold_linear_nested_modulelist" or "unfold_modulelists
def remove_sequential(m, all_layers=None):
    if all_layers is None:
        all_layers = []
        
    for layer in m.children():
        if len(list(layer.children())) > 1:
            remove_sequential(layer, all_layers)
        else: # a leaf node
            all_layers.append(layer)
    return all_layers

def test_remove_sequential():
    pass

        
################################################################################
### Learning Rate Schedulers
################################################################################
def epoch2cycle(epoch):
    """
    Given an epoch, returns in which cycle the epoch belongs to:
    Epoch e belongs to cycle c if e is in range [2**c -1, 2**(c+1)-2] 
    """
    return int(np.log2(epoch+1))

def test_epoch2cycle():
    for e in range(16):
        print(e, epoch2cycle(e))
        
def sgdr_lr(epoch, it, niter_per_epoch, lr_start, lr_end, policy='linear'):
    """
    Learning rate scheduler based on SGDR
    Using the linear function from learning rate of `lr_start` and `lr_end`,
    returns a learning rate at the `it`th Iteration of `epoch` Epoch. 
    The (global) step count is computed as:
        step = epoch * niter_per_epoch + it
    Args:
    - epoch (int): zero-indexed epoch number
    - it (int): zero-indexed iteration within the epoch
    - niter_per_epoch (int): number of iterations in an epoch. am
        - eg: often equals to the length of dataloader, `len(trainloader)`
    - lr_start (float): starting learning rate value
    - lr_end (float): ending learning rate value
    - policy (str): 'linear', 
        - todo: add 'triangle', 'exponential'(??)
    Returns:
    - (float): learning rate at `it`th iteration of `epoch`th epoch
    """
    
#     def gen_func (#todo
    # First which cycle does this epoch belong to?
    cycle = epoch2cycle(epoch)
    num_prev_epochs = 2**cycle-1
    offset_in_steps = num_prev_epochs * niter_per_epoch
    
    # Get the linear function in the timeframe of this cycle
    # In other words, cycle_function is defined as:
    # cycle_func(0) = lr_start; cycle_func(num_xs-1) = lr_end
    num_epochs = 2**cycle
    num_xs = num_epochs * niter_per_epoch
    start_x = 0; end_x = num_xs -1
    
    # slope and intercept
    m = (lr_end - lr_start) / end_x
    b = lr_start
    
    # First possible return value: the float of learning rate value. 
    ## This is not efficient because we build this function at each step 
    ## during the trianing 
    
    # Compute shift in step direction
    lr_func = lambda step: m * (step - offset_in_steps) + b
    
    # Compute global step count
    step = epoch * niter_per_epoch + it
    return lr_func(step)

    ## Slightly better option is to return a function that takes in the global step count
    ## But this is inconsistent with the argument signature (ie. not what a user would expect)
#     return lr_func 

    # So, it's better to return a function that takes in a consistent set of argument 
    # as in this function's call. 
    ## Return a function that takes in the exactly same set of arguments to this function 
#     return lr_func = lambda (epoch, it, niter_per_epoch): m*(epoch * n_iter_per_epoch + it) + b

def test_sgdr_lr(lr_max = 1, lr_min = 0.5, niter_in_epoch=10):
    print('updated')
    lr_start = lr_max
    lr_end = lr_min
    niter_per_epoch = niter_in_epoch
    print(f"lr_start, lr_end: {lr_start}, {lr_end}")
    print(f"niter_per_epoch {niter_per_epoch}")
    
    # Get the lr function that computes the learning rate at (global) step count
    # todo: using closure
#     lr_at_step = sgdr_lr(epoch, it, niter_per_epoch, lr_start, lr_end, policy='linear')
    for epoch in [0,1,2]:
        print("="*80)
        for it in range(niter_in_epoch):
            print(f"Epoch {epoch} Iter {it}: "
                  f" {sgdr_lr(epoch, it, niter_per_epoch, lr_start, lr_end)}")


# test_sgdr_lr()