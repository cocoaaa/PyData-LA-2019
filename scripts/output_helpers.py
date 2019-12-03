from __future__ import print_function 
from __future__ import division
# from __future__ import unicode_literals
# from future.utils import raise_with_traceback

import os,sys
import matplotlib.pyplot as plt
# import cv2
import re
from pathlib import Path
import numpy as np
from functools import partial

import PIL
import pdb, inspect

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

# todo - handle this better?
try:
    import cv2
except ImportError:
    print ('Could not import cv2')
    pass

# todo - handle this better?
try:
    import torch
except ImportError:
    print ('Could not import torch')
    pass
###############################################################################
### Add this file's path ###
###############################################################################
file_dir =  os.path.dirname(os.path.realpath(__file__))
print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    
###############################################################################
### Globals ###
###############################################################################
ROADS = ["motorway", "primary", "secondary", "tertiary", 
        "Residential", "unclassifed", "cart"]


###############################################################################
### Object Inspection helpers ###
###############################################################################
def get_parent_classes(someInstance):
    """
    Prints the given instance/object `someInstance`'s parent classes
    in the mro resolving order, starting at its own class
    
    Args:
    - someInstance: a realized instance
    """
    return inspect.getmro(someInstance.__class__)
    

###############################################################################
### Print out helpers ###
###############################################################################
def nprint(*args, header=True):
    if header:
        print("="*80)
    for x in args:
        print(x)

# nprint with header=False as kwarg
nhprint = partial(nprint, header=False)

def print_mro(x):
    """
    x: any python object or class
    
    Prints mro of x. Top is the most basic/foundation class.
    
    """
    x_cls = x if inspect.isclass(x) else x.__class__
    nhprint(*inspect.getmro(x_cls)[::-1])

def print_tree(startpath, max_level=float('Inf'), max_nfiles=float('Inf'), 
               hide=True, debug=False ):
    for root, dirs, files in os.walk(startpath):

        level = root.replace(startpath, '').count(os.sep)
        if debug: 
            nprint(level)  

        indent = ' ' * 4 * (level)
        subindent = ' ' * 4 * (level + 1)
        
        # Skip dot folders
        if hide and os.path.basename(root).startswith('.'):
            del dirs[:]
            continue 
            
        print('{}{}/'.format(indent, os.path.basename(root)))
        if level >= max_level: 
            print('{}{}{}'.format(subindent, len(files), " files"))
            print('{}{}{}'.format(subindent, len(dirs), " folders"))
            del dirs[:]

            if debug:
                print('{}{}'.format(subindent, 
                                    "max level reached, move to the next sibling node"))
            continue
            
        for i,f in enumerate(files):
            if i >= max_nfiles:
                print("{}...{} more files".format(subindent, len(files)-i))
                break
            print('{}{}'.format(subindent, f))
            
def print_tensor_info(t):
    if not torch.is_tensor(t):
        raise TypeError('Input must be a tensor: {}'.
                        format(type(t)))
    nprint('Tensor info')
    print('size: ', t.size())
    print('type: ', type(t))
    print('min: {}, max: {}'.format(t.min(), t.max()))
    
def print_pil_info(img):
    if not isinstance(img, PIL.Image.Image):
        raise TypeError('Input image should be PIL.Image.Image type: {}'.
                        format(type(img)))
    nprint("PIL Image Info")
    print("image mode: ", img.mode)
    print("img size: ", img.size)
    print("band extremes: ", img.getextrema())
    print("a pixel sample: ", img.getpixel((0,0)), 
          ", type: ", type( img.getpixel((0,0)) ))
           
###############################################################################
### Get Basic Image Metadata ###
###############################################################################
def get_img_shape(img_path):
    """
    Given a path to an image, return its height and width
    
    Args:
    - img_path(str or Path): path to an image file
    Returns:
    - shape (tuple of ints): len(shape) is the dimension of the image. 
        That is, if grayscale image, len(shape) is 2; if rgb, len is three;
        If rgba, len is four, etc. 
    """
    if not isinstance(img_path,str):
        img_path = str(img_path)
    img = cv2.imread(img_path,-1)
    return img.shape

###############################################################################
### RoadType Mapping ###
###############################################################################
def rt2idx_and_idx2rt():
    """
    Returns a mapping from roadtype string to index as used in Spacenet dataset
    """
    global ROADS
    return {rt:i for i,rt in enumerate(ROADS)}, {i:rt for i,rt in enumerate(ROADS)}

###############################################################################
### File Path (string) Manipulation ###
###############################################################################
def get_imgId(img_path):
    """
    Extract img ID from the path to the img file. 
    For example, if a path is /home/hayley/Data/Vector/spacenetroads_AOI_2_Vegas_img1323.geojson
    this function returns 1323
    
    Args:
    - img_path (str or pathlib.Path): str or path object to the file
    Returns:
    - int for the image ID
    """
    if isinstance(img_path, Path):
        img_path = str(img_path)
        
    fname = img_path.split('_')[-1]
    return int(re.findall(r'\d+', fname)[0])

def get_shortname(fpath, start=-2):
    """
    Extract a short filename from the full filepath. 
    Take the filename only, split it by '_', take elements from `start` index to the end
    and join it using '_'. 
    
    Args:
    - fpath (str or Pathlib.path): path to a file
    - start (int): index to start including the filename 
    Returns:
    - shortname (str)
    ---
    eg1: 
    - dir_name = Path('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/'
                    'RGB-PanSharpen-8bits-Crop/RGB-PanSharpen-8bits_AOI_2_Vegas_img352.tif')
    - shortname = 'Vegas_img352.tif'
    
    eg2: 
    - dir_name = Path('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/'
                    'spacenetroads/spacenet_8bits_AOI_2_Vegas_img352.tif')
    - shortname = 'Vegas_img352.tif'
    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
    return '_'.join(fpath.name.split('_')[start:])

def construct_fname(dir_name, imgId):
    """
    eg: 
    - dir_name = Path('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/'
                    'RGB-PanSharpen-8bits-Crop/')
    
    - template = Path('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/'
                    'RGB-PanSharpen-8bits-Crop/RGB-PanSharpen-8bits_AOI_2_Vegas_img352.tif')
    - imgId = int(1900)
    
    Returns:
    - Path('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/'
            'RGB-PanSharpen-8bits-Crop/RGB-PanSharpen-8bits_AOI_2_Vegas_img1900.tif')
    """
    g = dir_name.iterdir()
    found = False
    while not found:
        template = next(g) # first filename
        if template.suffix == '.tif':
            found = True
    base = template.parent
#     *stem_head, stem_tail = template.stem.split("_") #not compatible in python2
    stem_head = template.stem.split("-")[:-1]
    stem_tail = template.stem.split("_")[-1]
    suffix = template.suffix

    new_stem_tail = stem_tail[:3] + str(imgId)
    new_stem = "_".join(stem_head + [new_stem_tail])
    new_name = new_stem + suffix
    
    new_path = base/new_name
    return new_path


def get_road_vec_path(img_path, vec_suffix='.geojson'):
    """
    Given the path to an image in the spacenet challenge dataset, 
    return the string of the path to the corresponding road vector file.
    
    Args:
    - img_path (str or pathlib.Path): path to an image file
    - vec_suffix (str): vectorfile format. Default is '.geojson'
    
    Returns:
    - vec_path (str): path to the corresponding vector file
    """
    if isinstance(img_path, str):
        img_path = Path(img_path)
    root = img_path.parent.parent
    pid = img_path.stem.split('_')[1:]
    pid.insert(0,'spacenetroads')
    
    vec_dir = root / 'geojson/spacenetroads/'
    vec_stem = '_'.join(pid) 
    vec_suffix = '.geojson'
    vec_path = vec_dir / (vec_stem + vec_suffix)
    
    return str(vec_path)

def get_buffer_vec_path(road_vec_path):
    """
    Given a path to a road vector file,
    return the path to the correpsonding buffer vector file.
    
    Args:
    - road_vec_path (str or Path)
    Returns:
    - buff_path (str): path to the buffer vector file
    """
    
    if isinstance(road_vec_path, str):
        road_vec_path = Path(road_vec_path)
    fname = 'buffer_' + "_".join(road_vec_path.name.split('_')[1:])
    buff_path = road_vec_path.parent.parent/"buffer"/fname
    return str(buff_path)


def append_str_to_dirpath(dir_path, string):
    """
    Given a path to a directory, append the string to its name and return 
    the new path. 
    - eg: 
    ```python
   p = Path('~/Data/RGB-PanSharpen')
   crop_p = append_str_to_dirpath(p, '-Crop') # Path('~/Data/RGB-PanSharpen-Crop')
   ```
   
    Args:
    - dir_path (Path or str)
    - string (str): the string to append to the end of the dir_path
    
    Returns:
    - new_path (Path)
    """
    if isinstance(dir_path, str):
        dir_path = Path(dir_path)
    if not dir_path.is_dir():
        raise ValueError("First argument should be a directory")
        
    return dir_path.parent / (dir_path.name + string)    
    

def get_crop_dir(dir_path):
    """
    Given the path to the rgb (or mask) directory,
    return the Pathlib object to the corresponding crop directory
    eg: "AOI_2_Vegas_Roads_Sample/RGB-PanSharpen" 
        -> "AOI_2_Vegas_Roads_Sample/RGB-PanSharpen-Crop"
    eg: "AOI_2_Vegas_Roads_Sample/RGB-PanSharpen-8bits" 
        -> "AOI_2_Vegas_Roads_Sample/RGB-PanSharpen-8bits-Crop"
    Args:
    - dir_dir (Path): a full path to the directory with 1300x1300 images (.tif) 
        - eg: RGB-PanSharpen, RGB-PandSharpen-8bits, Mask
    Returns:
    - crop_dir (Path): a full path to the correpsonding crop directory
    """
    return append_str_to_dirpath(dir_path, "-Crop")
    

def get_mask_dir(rgb_dir):
    """
    Given a rgb or rgb8bits directory path,
    return its correpsonding mask directory
    """
    if isinstance(rgb_dir, str):
        rgb_dir = Path(rgb_dir)
    return rgb_dir.parent/'Mask'

def get_8bits_dir(rgb_dir):
    """
    Given an absolute path to a rgb directory,
    return its correspoding 8bits rgb directory.
    If its name contains `8bits`, the input will be returned.
    """ 
    if "8bits" in rgb_dir.name:
        return rgb_dir
    else:
        return append_str_to_dirpath(rgb_dir, "-8bits")

def get_mask_filename(rgb_file):
    """
    Given a full path to a rgb file, 
    return the full path to its corresponding mask file
    - eg. RGB-PanSharpen_AOI_2_img200.tif -> Mask_AOI_2_img200.tif (1300x1300 original size file)
    - eg. RGB-PanSharpen_AOI_2_img200_0.tif -> Mask_AOI_2_img200_0.tif (cropped tile)
    
    Args:
    - rgb_file (str or Path): full path to a rgb file (either the original-sized or a cropped tile)
    
    Returns:
    - mask_file (Path): full path to its corresponding mask file
    """
    if isinstance(rgb_file, str):
        rgb_file = Path(rgb_file)
        
    mask_dir = get_mask_dir(rgb_file.parent)
    fname = "mask_" + "_".join(rgb_file.name.split("_")[1:])
    return mask_dir/fname

## todo: add below which basically does the same thing as `get_mask_filename(rgb_file)`
# def get_buffer_img_path(rgb_path):
#     pass

def get_tile_filename(tile_dir):
    """
    Eg: tile_dir = Path('/home/hayley/Data_Spacenet/AOI_5_Khartoum_Roads_Train/RGB-PanSharpen-8bits-Crop')
        -> 'aoi_5_khartoum_roads_train_rgb-pansharpen-8bits-crop.txt'
    """
    return "_".join([tile_dir.parent.name, tile_dir.name]).lower()+ '.txt'

## vector file to raster file path conversion
def replace_filename_prefix(fpath, prefix, delimiter='_'):
    """
    Given a path to a file, replace the current prefix with the new prefix.
    
    Args:
    - fpath (str or Path): path to a file
    - prefix (str): new prefix
    - delimiter (str): it is use to split the filename into components
        - prefix of the filename correpsonds to the first component in the parts list
    
    Returns:
    - new_fpath (Path): path to the new file with the filename's prefix replaced
    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
        
    new_name = delimiter.join([prefix] + fpath.name.split(delimiter)[1:])
    return replace_part(fpath, -1, new_name)

def replace_part(fpath, idx, new_name):
    """
    Given a path to a file, replace the name of its component at `idx` level from the root
    with `new_name`
    - eg: rgb_path = '/home/hayley/Data_Spacenet/SpaceNet_Roads_Sample/AOI_2_Vegas_Roads_Sample/'
                     'RGB-PanSharpen/RGB-PanSharpen_AOI_2_Vegas_img699.tif
          new_path = change_part(rgb_path, , level=1)
          # '/home/hayley/Data_Spacenet/SpaceNet_Roads_Sample/AOI_2_Vegas_Roads_Sample/'
            'GBR/RGB-PanSharpen_AOI_2_Vegas_img699.tif
    Args:
    - fpath (str or Path): path to a file
    - idx (0<=int<len(fpath.parts), or negative idx): 
        index of the part to be replaced from Path.parts() list
    - new_name (str): new name to replace the old part
    
    Returns:
    - new_fpath (Path): path to the new file
    """
    if isinstance(fpath, str):
        fpath = Path(fpath)
    parts = list(fpath.parts)
    parts[idx] = new_name
    return Path('').joinpath(*parts)
    
def get_mask_path_from_vec_path(vec_path):
    """
    Given a path to a vector file (either buffer or road),
    return the path to the correpsonding mask raster file.
    
    Args:
    - vec_path (str or Path)
    Returns:
    - mask_path (str): path to the mask raster file
    """
    if isinstance(vec_path, str):
        vec_path = Path(vec_path)
    fname = 'mask_' + "_".join(vec_path.stem.split('_')[1:]) + '.tif'
    mask_path = vec_path.parent.parent.parent/"Mask"/fname
    return str(mask_path)

##todo:
## alternative way to get the mask file from rgb file path
def get_mask_path_from_rgb_path(rgb_path):
    """
    Given a path to a rgb file,
    return the path to the correpsonding mask raster file.
    
    Args:
    - rgb_path (str or Path)
    Returns:
    - mask_path (str): path to the mask raster file
    """
    step1 = replace_filename_prefix(rgb_path, 'mask')
    step2 = replace_part(step1, -2, 'Mask')
    return step2

def get_rgb_path_from_mask_path(mask_path):
    step1 = replace_filename_prefix(mask_path, 'RGB-PanSharpen')
    step2 = replace_part(step1, -2, 'RGB-PanSharpen')
    return step2
    
###############################################################################
### File Count ###
###############################################################################
def count_ftype(dir_paths, ftype, verbose=False):
    """
    Given a list of directories and filetype, count how many files of such filetype
    there are in total.
    
    Args:
    - dir_paths (list): a list of directory paths (Path objects)
    - ftype (str): eg. ".tif", ".png", ".geojson". Note the preceding "." 
    - verbose (bool): if True, print out each directory path names and counts
    
    Returns:
    - cnt (int): total count of files of type `ftype` in all directories
        in `dir_path`
    """
    total_cnt = 0
    for dir_path in dir_paths:
        if isinstance(dir_path, str):
            dir_path = Path(dir_path)
        
        cnt = 0
        for fname in dir_path.iterdir():
            if fname.suffix != ftype: continue
            cnt += 1
        total_cnt += cnt
        
        if verbose:
            print(str(dir_path),": ", cnt)
    return total_cnt
                   
def count_tif(dir_paths, verbose=False):
    """
    Count the number of tif files in the given dir_paths.
    
    Args:
    - dir_path (list of (str or Path)): list of absolute path to dirs
    
    Returns:
    - cnt (int): number of tif files
    """
    return count_ftype(dir_paths, '.tif', verbose=verbose)

###############################################################################
### File Check ###
###############################################################################
def is_empty(vec_path):
    """
    Given a path to .geojson vector file,
    check if the file is empty
    
    Args:
    - vec_path (str or Path)
    Returns:
    - is_empty (bool): True is the file is empty, False otherwise
    """
    pass

def check_rgb_mask_matching(rgb_files, mask_files):
    """
    Given iterables containing paths to rgb and mask images,
    check if `i`th files from each iterable form the correct pair 
    of rgb and mask image.
    *Note*: Both iterables should already have sorted the filenames in the 
    alphabetically ascending order.
    
    Args:
    - rgb_files (iterable):  contains str/Path to rgb image(s)
        - eg: a list of rgb filenames, Path object to a rgb directory
    - mask_files (iterable): contains str/Path to mask image(s)
    
    Returns:
    - flag (bool): True if `i`th mask filename is the correct mask name for 
        the `i`th rgb filename, for all `i`
    """
    if isinstance(rgb_files, str):
        rgb_files = Path(rgb_files)
    if isinstance(mask_files, str):
        mask_files = Path(mask_files)
        
    rgb_it = rgb_files.iterdir() if isinstance(rgb_files, Path) else iter(rgb_files)
    mask_it = mask_files.iterdir() if isinstance(mask_files, Path) else iter(mask_files)
    for rgb_file, mask_file in zip(rgb_it,mask_it):
        if Path(rgb_file).suffix != '.tif': continue
        correct_name = get_mask_path_from_rgb_path(rgb_file)
        if str(mask_file) != str(correct_name):
            raise ValueError("RGB, Mask filename mismatch: ", 
                             "\n", rgb_file,
                             "\n", mask_file,
                             "\n", correct_name
                            )
    return True
###############################################################################
### Plotting Helpers ###
###############################################################################
def show_in_8bits(fnames, fig=None, axes=None, ncols=3, figsize=(20,20), verbose=False):
    """
    show raster image in 8 bits. It silently performs 16 to 8bit conversion 
    if the input is 16 bits by calling `convert_to_8bits` in apls_tools.py
    """
    from math import ceil
    nrows = max(1, int(ceil(len(fnames)/ncols)))
    
    if fig is None or axes is None: 
        fig,axes = plt.subplots(nrows,ncols,figsize=figsize)
    axes = axes.flatten()

    for i,fname in enumerate(fnames):
        if not isinstance(fname, str):
            fname = str(fname)
        img = cv2.imread(fname,-1)

        if img.dtype == 'uint16':
            if verbose:
                print("{} is 16bits. We'll convert it to 8bits".format(fname))
            apls_tools.convert_to_8Bit(fname, './temp.tif')
            img = cv2.imread('./temp.tif',-1)
        assert(img.dtype != 'uint16')

        axes[i].imshow(img)
        axes[i].set_title("_".join(fname.split('_')[-2:]))
    
    # Delete any empty axes
    for j in range(len(fnames),len(axes)): 
        fig.delaxes(axes[j])

    return fig,axes[:len(fnames)]

#todo: rename it to show_img_road
def show_img_vec(img_path, road_path=None, figsize=(20,20)):
    """
    Plots an img in `img_path` (with 16->8bits conversion if needed)
    and a geopandas DataFrame object in `vec_path`
    
    Args:
    - img_path (pathlib.Path or str): path to the raster image
    - road_path (pathlib.Path or str): path to the road vector file (.geojson)
    """
    
    f,ax = show_in_8bits([str(img_path)], figsize=figsize)
    ax2 = f.add_subplot(122)
    if road_path is None:
        road_path = get_road_vec_path(img_path)
    elif isinstance(road_path, Path):
        road_path = str(road_path)
    gdf = gpd.read_file(road_path)
    gdf.plot(ax=ax2, figsize=figsize)
    ax2.set_title(get_shortname(road_path))
    return f
    
def show_arrs(arrs, fig=None, axes=None, ncols=3, figsize=(20,20), verbose=False):
    """
    Show np.arrays of 2d or 3d
    """
    from math import ceil
    nrows = max(1, int(ceil(len(arrs)/ncols)))
    
    if fig is None or axes is None: 
        fig,axes = plt.subplots(nrows,ncols,figsize=figsize)
    axes = axes.flatten()
    
    for i,arr in enumerate(arrs):
        axes[i].imshow(arr)
    
    # Delete any empty axes
    for j in range(len(arrs),len(axes)): 
        fig.delaxes(axes[j])

    return fig,axes[:len(arrs)]
    
def show_rgb_with_mask(rgb, mask, alpha=0.5, figsize=(20,20), title=""):
    """
    Show the rgb image overlayed with the mask image with the given alpha value applied
    to the mask image
    
    Args:
    - rgb (str/Path or np.array):
        - if str or Path, it's a path to the rgb file
        - if np.array, it's a 1, 3 or 4D numpy.array with (h,w,d,a) dimensions
    - mask (str/Path or np.array): 
        - if str or Path, it's a path to the mask file to overlay on top of the rgb
        - if np.array, uint8 type numpy array of the same size of `rgb_arr`
        - the image array must be 2 or 3 dimensional. If 3dim, one of the axis will be 
        assumed to be 1 and flattened out
    - alpha (float, [0,1])
    - figsize (tuple)
    
    Returns:
    - f, ax containing the rgb_arr overlayed by the mask image
    """
    f, ax = plt.subplots(figsize=figsize)
    
    # Set axis title
    ax_title =  "RGB w/ road buffer mask"
    if len(title)>0:
        ax_title += ": " + title
    ax.set_title(ax_title)
    
    # Read the images if the paths are the input args
    if isinstance(rgb, (str, Path)):
        rgb = cv2.imread(str(rgb), -1)
    if isinstance(mask, (str, Path)):
        mask = cv2.imread(str(mask), -1)
    
    if not isinstance(rgb, np.ndarray):
        raise TypeError('rgb must be an np array at this point')
    if not isinstance(mask, np.ndarray):
        raise TypeError('mask must be an np array at this point')

    # Show rgb and mask overlay
    ax.imshow(rgb)
    
    # remove an empty dimension if mask np.array is 3dimensional
    # np.squeeze() is inplace
    ax.imshow(np.squeeze(mask), alpha=alpha)
    
    return f,ax

###############################################################################
### Write Helpers ###
###############################################################################
#  gdf to geojson file
def gdf2geojson(gdf, out_name):
    gdf.to_file(out_name, driver='GeoJSON')
    
def arr2csv(arr, out_name, dialect='excel'):
    """
    Given an arr (eg. a list of strings), write each item to a column at each row
    
    Args:
    - arr (list type eg. list or np.array)
    """
    import csv
    with open(out_name, 'w') as f:
        writer = csv.writer(f, dialect=dialect)
        for row in arr:
            writer.writerow([row,])
    print('Wrote a csv file: ', out_name)
    
def arr2txt(arr, out_name):
    pass


###############################################################################
### Split dataset helpers ###
###############################################################################
def split_idx(n, ratio=[0.8,0.2], seed=None, verbose=False):
    arr = np.arange(n)
    
    # set random seed is seed is speficied
    if seed is not None:
        np.random.seed(seed)
    
    np.random.shuffle(arr) # inplace shuffling #todo: set seed?
    n0 = int(np.ceil(n*ratio[0]))
    idx0, idx1 = arr[:n0], arr[n0:]
    
    if verbose:
#         print("shffled: ", arr)
        print("Idx counts: ", n0, ", ", n-n0)

    return idx0, idx1
