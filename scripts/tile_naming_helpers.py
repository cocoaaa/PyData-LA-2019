from __future__ import print_function 
import os,sys,re
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import datetime as dt
import pdb
from inspect import getmro
from typing import Union, Iterable, Collection

###############################################################################
### Add this file's path ###
###############################################################################
file_dir =  os.path.dirname(os.path.realpath(__file__))
print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
from log_helpers import nprint
import spacenet_globals as spg


def get_sp_mask_dir(rgb_dir):
    """
    Given a rgb or rgb8bits directory path,
    return its correpsonding mask directory.
    It handles the cropped file's parent directory matching correctly. 
    - eg. RGB-PanSharpen-8bits-Crop/RGB-PanSharpen_AOI_2_img200_0.tif -> Mask-Crop/Mask_AOI_2_img200_0.tif
    
    """
    if isinstance(rgb_dir, str):
        rgb_dir = Path(rgb_dir)
    if not rgb_dir.is_dir():
        raise ValueError("Input rgb_dir must be a directory")
    
    if 'Crop' in str(rgb_dir):
        mask_dir = 'Mask-Crop'
    else:
        mask_dir = 'Mask'
    return rgb_dir.parent/mask_dir

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

def get_sp_mask_fn(rgb_file):
    """
    Given a full path to a rgb file, 
    return the full path to its corresponding spacenet mask file
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


def get_osm_mask_fn(rgb_fn, prefix='OSM-Mask'):
    """
    Useful when the rgb and osm-mask of size 1300x13000 were input to `crop_helpers.create_tiles_with_latlon`
    function's default output path naming scheme.
    This will give a correct osm-mask tile filename for both cropped and uncropped rgb files
    
    Eg:
    rgb_fn: '/Tile/650x650/vegas/rgb/rgb_vegas_img1454_1.tif'
    osm_fn: f'/Tile/650x650/vegas/{prefix}/{prefix}_vegas_img1454_1.tif'
    
    """
    return change_tile_prefix(rgb_fn, prefix)

def change_tile_prefix(rgb_fn, prefix):
    """
    Useful when the rgb and osm-mask of size 1300x13000 were input to `crop_helpers.create_tiles_with_latlon`
    function's default output path naming scheme.
    This will give a correct osm-mask tile filename for both cropped and uncropped rgb files
    
    Eg:
    rgb_fn: '/Tile/650x650/vegas/rgb/rgb_vegas_img1454_1.tif'
    osm_fn: f'/Tile/650x650/vegas/{prefix}/{prefix}_vegas_img1454_1.tif'
    
    """
    rgb_fn = Path(rgb_fn) if isinstance(rgb_fn, str) else rgb_fn
    fn = '_'.join([prefix] + rgb_fn.name.split('_')[1:])
    return rgb_fn.parent.parent/f'{prefix}/{fn}'

def tname_rgb2mask(rtile_fn):
    

###############################################################################
### Todo: Refactor below chunck of code. Most likely no longer needed as we ###
### changed the naming to Tile-based instead of -Crop convention            ###
###############################################################################
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
    return the path to the correpsonding (uncropped) mask raster file.
    
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

def get_big_raster_path(crop_path):
    if not isinstance(crop_path, Path):
        crop_path = Path(crop_path)
    stem = crop_path.stem
    suffix = crop_path.suffix
    
    big_name = '_'.join(stem.split('_')[:-1]) + suffix
    
    big_head = crop_path.parent.parent
    big_dir= '-'.join(crop_path.parent.stem.split('-')[:-1])
    big_path = big_head/ big_dir / big_name
    return str(big_path)

##todo:
## alternative way to get the mask file from rgb file path
def get_mask_path_from_rgb_path(rgb_path):
    """
    Given a path to a rgb file,
    return the path to the correpsonding (uncropped) mask raster file.
    
    Args:
    - rgb_path (str or Path)
    Returns:
    - mask_path (Path): path to the mask raster file
    """
    step1 = replace_filename_prefix(rgb_path, 'mask')
    step2 = replace_part(step1, -2, 'Mask')
    return step2

def get_rgb_path_from_mask_path(mask_path, choose_8bits=True, validate=True):
    """
    Given a path to an uncropped mask raster image,
    returns the path to the corresponding (uncropped) RGB-PanSharpen image
    Args:
    - mask_path (str or Path)
    - choose_8bits (bool): if True, choose RGB-PanSharpen-8bits images
    - validate (bool): if True, check if both the input and output filenames exist
    Returns:
    - (Path)
    
    """
    if 'Crop' in str(mask_path):
        raise IOError(f"Input file is for a cropped image. Use `get_rgbtile_path_from_masktile_path` function instead")
        
    rgb_prefix = 'RGB-PanSharpen-8bits' if choose_8bits else 'RGB-PanSharpen'
    step1 = replace_filename_prefix(mask_path, rgb_prefix)
    step2 = replace_part(step1, -2, rgb_prefix)
    
    if validate:
        for p in [mask_path, step2]:
            if not Path(p).exists(): warnings.warn(f"{p} doesn't exist", RuntimeWarning)
        
    return step2

def get_rgbtile_path_from_masktile_path(mtile_path, choose_8bits=True, validate=True):
    """
    Given a path to a cropped mask raster tile image (mtile),
    returns the path to the corresponding (uncropped) RGB-PanSharpen image
    
    Args:
    - mask_path (str or Path)
    - choose_8bits (bool): if True, choose RGB-PanSharpen-8bits images
    - validate (bool): if True, check if both the input and output filenames exist
    Returns:
    - (Path)
    
    """
    rgb_prefix = 'RGB-PanSharpen-8bits' if choose_8bits else 'RGB-PanSharpen'
    rgb_dirname = rgb_prefix + "-Crop"
    step1 = replace_filename_prefix(mtile_path, rgb_prefix)
    step2 = replace_part(step1, -2, rgb_dirname)
    
    if validate:
        for p in [mtile_path, step2]:
            if not Path(p).exists(): warnings.warn(f"{p} doesn't exist", RuntimeWarning)
        print("Both files exist")
        
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


###############################################################################
### Tests
###############################################################################
def test_get_big_raster_path():
    cpath = ('/home/hayley/Data_Spacenet/AOI_3_Paris_Roads_Train/'
            'RGB-PanSharpen-8bits-Crop/RGB-PanSharpen-8bits_AOI_3_Paris_img48_0.tif')
    fpath = get_big_raster_path(cpath)
    nprint(cpath, fpath)
    
def test_get_rgb_path_from_mask_path():
    mtile_path = '/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/Mask-Crop/mask_AOI_2_Vegas_img1001_0.tif'
    mpath = get_big_mask_path(mtile_path)
    rpath = get_rgb_path_from_mask_path(mpath)
    nprint(mpath, rpath)

    mask_img = reshape_as_image(rio.open(mpath).read())#.reshape((mask_img.shape[0], mask_img.shape[1], 1))
    rgb_img = reshape_as_image(rio.open(rpath).read())
    f, ax = plt.subplots(1,2, figsize=(20,20))
    ax = ax.flatten()
    ax[0].imshow(mask_img.squeeze())
    ax[1].imshow(rgb_img)
    
def test_get_rgbtile_path_from_masktile_path():
#     m_path = '/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/Mask/mask_AOI_2_Vegas_img1001.tif'
#     rgb8_path = get_rgb_path_from_mask_path(m_path, choose_8bits=True, validate=True)

    mtile_path = '/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/Mask-Crop/mask_AOI_2_Vegas_img1001_0.tif'
    rgb8tile_path = get_rgbtile_path_from_masktile_path(mtile_path, choose_8bits=True, validate=True)
    nprint(mtile_path, rgb8tile_path)
    
def test_extract_city():
    assert extract_city('h/sdf_vegas_k1.tif') == 'vegas'
    
def test_get_osm_mask_fn():
    rgb_fn = Path('/home/hayley/Data_Spacenet/Tile/650x650/vegas/rgb/rgb_vegas_img1454_1.tif')
    prefix = 'OSM-Mask'
    osmm_fn = get_osm_mask_fn(rgb_fn, prefix=prefix)
    assert osmm_fn == Path(f'/home/hayley/Data_Spacenet/Tile/650x650/vegas/{prefix}/{prefix}_vegas_img1454_1.tif')