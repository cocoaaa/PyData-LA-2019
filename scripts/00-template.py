import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision 
import torchvision.transforms as transforms

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os,sys

from typing import Collection, Union, Optional, Iterable, Mapping

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