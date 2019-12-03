import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms

import numpy as np
import matplotlib as mpl
# mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from pathlib import Path
import os,sys
import pdb

################################################################################
### Path setup
################################################################################
PROJ_ROOT = Path(os.getcwd()).parent
DATA_DIR = PROJ_ROOT/'data/raw'
SRC_DIR = PROJ_ROOT/'scripts'
paths2add = [PROJ_ROOT, SRC_DIR]

# print("Project root: ", str(ROOT))
# print("this nb path: ", str(this_nb_path))
# print('Scripts folder: ', str(SCRIPTS))

# Add project root and src directory to the path
for p in paths2add:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        print("Prepened to path: ", p)
      
from convert_helpers import to_npimg

################################################################################
### Helpers
################################################################################
def show_tensor(img_t, figsize=(10,10), title=""):
    """
    Show an 3dim image tensor (nc, h, w)
    """
    # unnormalize
    img_t = img_t/2 + 0.5
    img_np = to_npimg(img_t)
    f,ax = plt.subplots(figsize=figsize)
    ax.imshow(img_np)
    ax.set_title(title)
    f.show()
    return f, ax

def test_show_tensor(dataloader):
    for i in range(3):
        batch_X, batch_y = iter(dataloader).next()
        grid = torchvision.utils.make_grid(batch_X)
        title = batch_y2str(batch_y)
        show_tensor(grid, figsize=(20,20), title=title)
        
if __name__ == '__main__':
    # todo: define dataloader
    # dataloader = 
#     test_show_tensor(dataloader)
    pass