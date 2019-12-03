import numpy as np
import pandas as pd
from pathlib import Path
import os,sys

from typing import Collection, Union, Optional, Iterable, Mapping
from collections import defaultdict
import warnings

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
def parse_othertags_column(df, debug=False):
    """
    Extract (k,v) paris from the 'other_tags' column of the input dataframe
    Assume this column follows the format of:
    
    "keyname1"=>"value1", "key2"=>"value2", ...
    
    Args:
    - df (pandas.DataFrame)
    
    Returns:
    - info (dict): python dict of key,value pairs.
        - value is a list of length of the input dataframe (i.e. num of rows)
        - key is a str, value is a list of string objs, after str.lower() is applied
        - If a row's `other_tags` column doesn't contain certain key, its value is
        assigned to None
    """
    if 'other_tags' not in df.columns:
        warnings.warn("The input dataframe doesn't have a column named `other_tags`",
                      warnings.RuntimeWarning)
        return {}
    
    info_dict = defaultdict(list)
    value = None  # default for any value is None
    for idx, r in df.iterrows():
        # r is pd.Series
        # Only process this column if it has tag(s)
        # skip if the value is an empty string or None
        if r.other_tags is None or len(r.other_tags) == 0 :
            continue
            
        # Has tag(s)
        tags = r.other_tags.split(',')
        for kv_str in tags:
            if "=>" not in kv_str: 
                continue 
                
            k,v = kv_str.split("=>")
            k,v = k.strip('"').lower(), v.strip('"').lower()
            info_dict[k].append(v) 
            
            if debug:
                print("k,v: ", k, v)
    return info_dict 

def recursive_counter(arr: Iterable, d: Mapping) -> None:
    """
    Modifies the input dictyionary directly!
    """
    if isinstance(arr, str):
        if arr in d:
            d[arr] += 1  
        else:
            d[arr] = 1
    else: 
        for i, ele in enumerate(arr):
            # assume arr is a list of str
            recursive_counter(ele, d)
                
################################################################################
### Tests
################################################################################
def test_recursive_counter():
    labels = defaultdict(int)
    df = pd.DataFrame.from_dict({0:['one', 'two', ['three, four'], [['five', 'six'],'seven']]})    
    recursive_counter(df[0], labels)
    print(labels)
    

def test_parse_othertags_column():
    pass

    


