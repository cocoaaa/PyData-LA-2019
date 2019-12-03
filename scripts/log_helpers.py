from __future__ import print_function 
from __future__ import division

import os,sys
import re
from pathlib import Path
import numpy as np

import pdb
from inspect import getmro
from functools import partial
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

# if sys.version_info <= (3,5):
try:
    import geopandas as gpd
    import apls_tools # dependent on geopandas inside itself
except ImportError: # will be 3.x series
    pass

###############################################################################
### Add this file's path ###
###############################################################################
file_dir =  os.path.dirname(os.path.realpath(__file__))
print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    
###############################################################################
### Print out helpers ###
###############################################################################
def nprint(*args, header=True):
    if header: print("="*80)
    for x in args:
        print(x)
        
# nprint with header=False as kwarg
nhprint = partial(nprint, header=False)

def sprint(anArray, sample_size):
    """
    "s"ample "print":
    Print out random `n` number of samples from the iterable `anArray`
    
    Args:
    - anArray (iterable)
    - sample_size (int): number of samples to print out
    """
    n = len(anArray)
    if not isinstance(anArray, np.ndarray):
        anArray = np.array(anArray)
        
    idx = np.random.randint(0, n, sample_size)
    nprint(anArray[idx], header=False)

    