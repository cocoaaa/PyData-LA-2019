#helpers.py
## todo: separate to io.helpers.py and viz.helpers.py
import os, sys, random, re

import numpy as np
import datetime as dt
from collections import Counter

import holoviews as hv
from typing import Iterable, Dict, Callable
import torch

import pdb
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
    return dt.datetime.now().strftime('%Y%m%d_%H%M')

_camel_re1 = re.compile('(.)([A-Z][a-z]+)')
_camel_re2 = re.compile('([a-z0-9])([A-Z])')
def camel2snake(name:str)->str:
    s1 = re.sub(_camel_re1, r'\1_\2', name)
    return re.sub(_camel_re2, r'\1_\2', s1).lower()

def arr2str(arr: Iterable, decimal: int = 3, delimiter: str = ','):
    arr = [str(np.around(ele, decimal)) for ele in arr]
    return delimiter.join(arr)

def snake2camel(s):
    "Convert snake_case to CamelCase"
    return ''.join(s.title().split('_'))

def class2attr(self, cls_name):
    return camel2snake(re.sub(rf'{cls_name}$', '', self.__class__.__name__) or cls_name.lower())

def get_cls_name(x, camel=False):
    name = None
    try:
        name = x.__name__
    except AttributeError:
        name = x.__class__.__name__

    if camel:
        name = camel2snake(name)
    return name

def get_module_name(x):
    return x.__module__

def npify(arr):
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return arr

# context managers
def replacing_yield(o, attr, val):
    """context manager to temporarily replace an attribute
    Copied from: https://github.com/fastai/fastai_dev/blob/master/dev/13_learner.ipynb#L23
    """
    old = getattr(o, attr)
    try:
        yield setattr(o, attr, val)
    finally:
        setattr(o, attr, old)

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

def to_device(model, device):
    """
    Check if the model's parameters are already in `device`
    Only perform the model.to(device) if not.
    This modifies model if necessary. After this function is applied it is guaranteed that all the 
    parameters of model are in `device`'s memory
    """
    p = next(model.parameters())
    if p.device == device:
        return
    model.to(device)

def get_device(model):
    """Assumes all parameters of the model are in one device
    """
    p = next(model.parameters())
    return p.device

def assert_mean_reduction(loss_fn):
    if hasattr(loss_fn, 'reduction'):
        assert loss_fn.reduction == 'mean', f'''loss_fn should compute (mini-batch) averaged mean:  
        {loss_fn.reduction}'''



## Visualiza a batch of datasets
def show_batch(batch, max_num=2):
    """
    batch (tuple of tensors): returned by SegDataset
    Visualize the batch x and y overlay as hv Elements
    """
    batch_x, batch_y = batch
    bs = len(batch_x)
    assert bs == len(batch_y), len(batch_y)
    for i,(x, y) in enumerate(zip(batch_x, batch_y)):
        if i >= max_num:
            break
        x_np = x.detach().numpy().transpose((1,2,0))
        y_np = y.detach().numpy()
        print("x: ", x_np.dtype, x_np.shape)
        print("y: ", y_np.dtype, y_np.shape)
        overlay = hv.RGB(x_np) + hv.Image(y_np, group='mask')
        display(overlay)
#         pdb.set_trace()
        
    
def show_filter_batch(batch, n_cols=4, max_show=16):
    """
    Show a batch of filter tensors using holoviews
    
    Args:
    - batch (torch.tensor): 4 dimensional torch.tensor. (bs, nC, H, W)
    
    Returns:
    - holoviews layout object
    """
    bs = len(batch)
    batch_np = batch.cpu().numpy().transpose((0, -2, -1, -3))[:max_show]
    print('Batch in numpy: ', batch_np.shape)
    n_rows = int(float(bs/n_cols))
    
    layout = []
    for img in batch_np:
        layout.append(hv.Image(img, group='mask').opts(shared_axes=False))
    return hv.Layout(layout).cols(n_cols)
    
    
    
    
## Set random seed
def random_seed(seed_value, use_cuda:bool):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
     