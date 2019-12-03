# %load_ext autoreload
# %autoreload 2

import os, sys
import numpy as np
import scipy as sp
import intake
    
from pathlib import Path
from pprint import pprint
import pdb

import matplotlib.pyplot as plt
# %matplotlib inline

# ignore warnings
import warnings
if not sys.warnoptions:
    warnings.simplefilter('ignore')
    
# Don't generate bytecode
sys.dont_write_bytecode = True

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
from 



