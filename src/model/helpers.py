# model.helpers.py

import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

import typing
from pathlib import Path
import pdb

## Model Helpers
def get_trainable(model_params):
    return (p for p in model_params if p.requires_grad)


def get_frozen(model_params):
    return (p for p in model_params if not p.requires_grad)


def all_trainable(model_params):
    return all(p.requires_grad for p in model_params)


def all_frozen(model_params):
    return all(not p.requires_grad for p in model_params)


def freeze_all(model_params):
    for param in model_params:
        param.requires_grad = False
        
def unfreeze_all(model_params):
    for param in model_params:
        param.requires_grad = True
        
def freeze_with_substr(model, substr):
    """
    Freeze parameters with 'substr' in its name
    """
    for name, param in model.named_parameters():
        if substr in name:
            param.requires_grad = False
            

def freeze_convs_in_conv_blocks(model):
    for name, child in model.named_children():
        if 'conv_block' not in name:
            continue
        for n2, m in child.named_modules():
            if isinstance(m, nn.Conv2d):
#                 print(f'{name}.{n2}')
                for p in m.parameters():
                    p.requires_grad_(False)
    

def freeze_bns_in_conv_blocks(model):
    for name, child in model.named_children():
        if not 'conv_block' in name:
            continue
        for mname, m in child.named_modules():
            if isinstance(m, nn.BatchNorm2d):
#                 print(f'{name}.{mname}')
                for p in m.parameters():
    #                 continue
                    p.requires_grad_(False)


def get_param_states(model):
    trainables = []
    frozens = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainables.append(name)
        else:
            frozens.append(name)
    return {'trainable': trainables,
            'frozen': frozens}
    

# Model save, load
def save_state(state, out_fn):
    """
    Args:
    - state (dict): contains model's `state_dict`, and may contain
        other key,value pair such loss, lr, metrics.
        - eg: 
        state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
            }
            
    - out_fn (str or Path): path to the output file
    """
    out_fn = Path(out_fn) if isinstance(out_fn, str) else out_fn
    if not out_fn.parent.exists():
        out_fn.parent.mkdir(parents=True)
        print('Created: ', out_fn.parent)
    torch.save(state, out_fn)

def load_state(in_fn,
               map_location,
               model,
               optimizer=None
):
    """
    map_location: eg. 'cuda:0'
    """
    state = torch.load(in_fn, map_location=map_location)
    model.load_state_dict(state['model_state'])
    if optimizer:
        optimizer.load_state_dict(state['optimizer_state'])
    
    return state       
        
# Model save, load
def save_checkpt(epoch:int, 
                 model, 
                 optimizer, 
                 loss:int, 
                 out_fn):
    state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
    }
    save_state(state, out_fn)

    