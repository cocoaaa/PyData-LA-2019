#helpers.py
## todo: separate to io.helpers.py and viz.helpers.py
import os, sys, random

import numpy as np
import datetime as dt
from collections import Counter

import holoviews as hv
from holoviews import opts
import torch

## IO Helpers
def load_txt(txt_file, to_sort=True):
    """
    Read data line by line
    """
    data = []
    with open(txt_file, 'r') as reader:
        for l in reader:
            data.append(l.strip())
    if to_sort:
        data.sort()
        
    return data
  
def write2lines(myItrb, out_fn):
    """
    Write each item in myItrb to a line
    """
    with open(out_fn, 'w') as writer:
        for item in myItrb:
            writer.write(str(item)+'\n')
            
def append2file(content, fn):
    with open(fn, 'a') as f:
        f.write(str(content)+'\n')
    
def now2str():
    return dt.datetime.now().strftime('%Y%m%d-%H%M%S')

def get_cls_name(x):
    try:
        return x.__name__
    except:
        return x.__class__.__name__
def get_module_name(x):
    return x.__module__
    

## Datatype conversions
def npify(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return arr

## Transform helpers
def get_pad_size(img_h, img_w, out_size):
    """
    Computes the number of paddings in height and width
    Args:
    - img_h (int)
    - img_w (int)
    - out_size (int or tuple): if tuple, it must be (out_h, out_w)
    
    Returns:
    - tuple: (number_to_pad_in_height, number_to_pad_in_width)
    """
    out_h, out_w = out_size if isinstance(out_size, tuple) else (out_size, out_size)
    assert out_h >= img_h, f'out_h must be < img_h. img_h is {img_h}'
    assert out_w >= img_w, f'out_w must be < img_w. img_w is {img_w}'
    
    nx_pad = max(out_w - img_w, 0)
    ny_pad = max(out_h - img_h, 0)
    return ny_pad, nx_pad
  
#Stat helpers
def count_freqs(batch_y, nClass, ignore_idx=None, verbose=False):
    """
    batch_y (mini-batch of 2D labels): shape is (bs, h, w) and each value
    is one of ints in [0,1,...nClass-1]
    
    Returns
    - a dictionary of counts of unique values in batch_y
    """
    _c = Counter()
    for y in batch_y:
        _c.update(Counter(y.flat))
        

    c = dict()
    for label in _c:
        if label >= nClass or label == ignore_idx:
            continue
        c[label] = _c[label]
        
    if verbose:
        print('Before truncating by nClass and filtering out idx to ignore: ')
        print('\tN unique classes: ', len(_c))
        print('After popping, len(c): ', len(c))
      
    return c

## 3D tensor to numpy conversion
def to_np(t):
    assert t.ndim in [3,4], f'input tensor must be 3 or 4 dim: {t.ndim}'
    
    if t.ndim == 3:
        t_np = t.numpy()[None, :, :, :] 
    else:
        t_np = t.numpy()
    return t_np.transpose((0, -2, -1, -3)).squeeze()

def to_tensor(nda):
    assert nda.ndim in [3,4], f'input array must be 3 or 4 dim: {nda.ndim}'
    if nda.ndim == 3:
        nda = nda[None, :, :, :]
    transposed = nda.transpose((0, -1, -3, -2))
    return torch.from_numpy(transposed.squeeze())
        

    
## Set random seed
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
     