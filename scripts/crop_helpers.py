from __future__ import print_function 
from __future__ import division

import os,sys
import re
from pathlib import Path
import numpy as np

import pdb
from tqdm import tqdm

from inspect import getmro
from typing import Union, Iterable, Collection

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
# file_dir =  os.path.dirname(os.path.realpath(__file__))
file_dir =  os.path.dirname(os.path.realpath('.'))
print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    
###############################################################################
### Import my modules ###
###############################################################################
import spacenet_globals as spg
from naming_helpers import extract_city, now_str
from output_helpers import nprint

from affine import Affine
import rasterio as rio
from rasterio.plot import reshape_as_image
from itertools import zip_longest

# Path list 
Path.ls = lambda x: [p.name for p in x.iterdir()]

def pad_for_divisible_by(img: np.ndarray, 
                         factors: Iterable[int]) -> np.ndarray:
#                         pad_thresh: float) # todo
    """
    Pad the image at the end of each dimension with zero so that each axis's dimension
    is divisible by the correpsonding factor in `factors`
    
    Args:
    #todo
    - pad_thresh (float): percentage of each dimension to decide whether to pad or to trim
    
    eg: 
    the output's row dimension size is divisible by factors[0]
    and col's dimension is divisible by factors[1]
    
    Returns:
    a (possibly) padded np.ndarray
    """
    assert img.ndim == len(factors)
    factors = np.array(factors)
    n_pads = np.array(img.shape) % factors
    n_pads = [factors[i] - n_pads[i] if n_pads[i] > 0 else 0 for i in range(len(n_pads))]
    return np.pad(img, list(zip_longest([], n_pads, fillvalue=0)), 'constant')
    

def compute_transform(tile_width, 
                    tile_height, 
                    bounds,
                    verbose=False):
    """
    Simple affine transform without rotation
    """

    xmin, ymin, xmax, ymax = bounds
    xres = (xmax-xmin)/tile_width
    yres = (ymax-ymin)/tile_height
    transform = Affine.translation(xmin, ymax) * Affine.scale(xres, -yres)
    
    if verbose:
        print(ymin, ymax)
        print('neg yres: ', -yres)
        print(transform)
        
    return transform


def create_tiles_with_latlon(fn: str, 
                             tile_width: int, 
                             tile_height: int, 
                             to_save=False, 
                             out_dir=None,
                             dtype=rio.uint8,
                             img_type='infer',
                            verbose=False,
                            debug=False) -> Iterable:
    """
    Args:
    - fn (str): original uncropped RGB-PanSharpen-8bits file
    - tile_width, tile_height (int, int): width and height of each tile
    - to_save (bool): if True, save the resulting tiles to the disk
    - out_dir (str or Path): output directory
    - dtype: data type of the output tiff file 
    - img_type ('infer' or one of spg.IMG_TYPES): img type of the input file
        This will be used for output directory naming
    - verbose (bool)
    - debug (bool)
    
    Returns:
    - tiles (list of np.array): each array has the same dimension as the input file
    - Optionally, saves the tiles to the disk if `to_save`
    
    """
    ds = rio.open(fn)
    img = reshape_as_image(ds.read())

    img = pad_for_divisible_by(img, (tile_height, tile_width, 1))
    n_iters_y, n_iters_x = np.array(img.shape[:2]) // np.array([tile_height, tile_width])
    
    if to_save:
        # Set output directory
        city = extract_city(fn)
        if img_type == 'infer':
            img_type = fn.stem.split('_')[0].split('-')[0].lower()
        assert isinstance(img_type, str) and img_type in spg.IMG_TYPES #todo: better than hard-coding this?

        if out_dir is None:
            out_dir = Path.home()/"Data_Spacenet/Tile/"

        else:
            out_dir = Path(out_dir)
        out_dir = out_dir/f"{tile_width}x{tile_height}/{city}/{img_type}"

        if not Path(out_dir).exists():
            out_dir.mkdir(parents=True)
            print(f"{out_dir} is created")
    
    tiles = []
    for i in range(n_iters_x):
        c0, c1 = i*tile_width, (i+1)*tile_width
        for j in range(n_iters_y):
            r0, r1 = j*tile_height, (j+1)*tile_height
            
            tile = img[r0:r1, c0:c1]

            min_lon, max_lat = ds.xy(r0, c0, 'ul') #.xy(ridx, cidx)
            max_lon, min_lat = ds.xy(r1, c1, 'ul')
            bounds = (min_lon, min_lat, max_lon, max_lat)

            transform = compute_transform(tile_width, tile_height, bounds, verbose=verbose)
            tiles.append({'ij': (i,j), 
                          'tile': tile, 
                          'bounds': bounds,
                         'transform': transform})
            
            if to_save:
                # Set output filename
                t_count = len(tiles)
                out_fn = f"{img_type}_{city}_{fn.stem.split('_')[-1]}_{t_count}{fn.suffix}"                                           
                out_path = out_dir/out_fn
                
                # Add the image tile path to the tile info dictionary
                tiles[-1].update(path=out_path)
                
                # Set rasterio writing mode
                with rio.Env():
                    profile = ds.profile
                    profile.update(width=tile_width,
                                   height=tile_height,
                                   transform=transform)
                    if verbose:
                        nprint('Before writing...')
                        nprint('i,j: ', i,j)
                        nprint('bounds: ', bounds)
                        nprint('transform: ', transform)
                        nprint('profile transform: ', profile['transform'])

                    with rio.open(out_path, 'w', **profile) as dst:
                        if tile.ndim == 3:
                            tile = tile.transpose(2,0,1)
                        dst.write(tile, ds.indexes)
                        
                if verbose:
                    print('Saved to: ', out_path)

                if debug:
                    res = rio.open(out_path)
                    if verbose:
                        nprint("Output tile meta...")
                        nprint('output transform mtx: ', res.transform)
                        nprint('meta: ', res.meta)
                        nprint('res: ', res.res)
#                     pdb.set_trace()
    return tiles,out_dir

def create_tiles_with_latlon_in_dir(src_dir: Union[str,Path], 
                                    tile_width: int, 
                                    tile_height: int, 
                                   to_save,
                                    out_dir=None,
                                    verbose=False,
                                    debug=False
                        ):
    """
    Creates tiles with georeference data for all images in the input `src_dir`
    """
    if isinstance(src_dir,str):
        src_dir = Path(src_dir)
        
    fns = [p for p in src_dir.iterdir() if p.suffix in ['.tif', '.tiff']]
    print(len(fns))
    
    for fn in tqdm(fns):
        _, out_dirname = create_tiles_with_latlon(fn, 
                                 tile_width, 
                                 tile_height, 
                                 to_save=to_save, 
                                 out_dir=out_dir,
                                verbose=verbose,
                                debug=debug)
    return out_dirname
                        
###############################################################################
### Tests
###############################################################################                    
def test_pad_for_divisible_by():
    temp = np.ones((8,8))
    overlay = (hv.Image(temp) 
               + hv.Image(pad_for_divisible_by(temp, (2,2)), label='factor: 2')
               + hv.Image(pad_for_divisible_by(temp, (2,3)), label='factors: (2,3)')
              ).opts(opts.Image(cmap='gray', shared_axes=False))
    display(overlay.cols(2))
