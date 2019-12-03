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
def to_npimg(t):
    """
    Returns a new view of tensor t (nc, h, w) as a numpy array of (h,w,nc)
    Note: this is an operation which assumes the input is image data with (h,w,nc).
        - if t is 3-dim, we assume t's dimensions represent (n_channels, h, w) and 
        *not* (num_grayscale_images, h, w)
        - if t has 4 dims, we assume t's dimensions correspond to (batch_size, n_channels, h, w)
    These assumptions matter because tensor and numpy images use different conventions in 
    interpreting each dimension: 
        - tensor: (nc, h, w) or (n_batchs, nc, h, w)
        - numpy_imgarray : (h,w,nc) or (n_batches, h, w, nc)
    n_channels (`nc`) is equivalent to `num_planes` in PyTorch documentation.
    """
    if t.dim() not in [3, 4]:
        raise TypeError("input must have dim of 3 or 4: {}".
                        format(t.dim()))
                        
    if t.dim() == 3:
        return t.permute(1,2,0).numpy()
    else:
        return np.array(list(map(to_npimg, t)))
