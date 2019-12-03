from __future__ import print_function 
from __future__ import division

import os,sys
import re
from pathlib import Path
import numpy as np

import pdb
from inspect import getmro
from typing import Union, Iterable, Collection

from itertools import zip_longest

import rasterio as rio
from rasterio.plot import reshape_as_image
from affine import Affine

from shapely.geometry import Polygon

import osmnx as ox
ox.config(log_console=True, use_cache=True)

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
# file_dir =  os.path.dirname(os.path.realpath('.'))
print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    
###############################################################################
### Import my modules ###
###############################################################################
import spacenet_globals as spg
import SpacenetPath as spp

from naming_helpers import extract_city, now_str
from output_helpers import nprint

###############################################################################
### Helpers ###
###############################################################################
def bounds2list(rio_bounds):
    xmin, ymin, xmax, ymax = rio_bounds
    return (xmin, ymin, xmax, ymax)

def to_osmbbox(bounds):
    """
    Args:
    - bounds (tuple or rasterio.bounds object): encodes (xmin, ymin, xmax, ymax)
    Returns:
    - tuple reordered to match OSM's bbox convention: N, S, E, W
    """
    xmin, ymin, xmax, ymax = bounds
    return (ymax, ymin, xmax, xmin)


def bounds2poly(bounds):
    """
    Args:
    - bounds (tuple of floats): (xmin, ymin, xmax, ymax)
    
    Returns
    - poly (shapely.geometry.Polygon): a polygon representing the bounds
    """
    xmin, ymin, xmax, ymax = bounds
    return Polygon([(xmin,ymin),
                   (xmin,ymax),
                   (xmax,ymax),
                   (xmax,ymin),
                   (xmin,ymin)])

def get_largest_bounds(img_dir: Union[Path, str]):
    """
    Args:
    - img_dir (Path or str): path to a directory that contains images 
    from which to compute the union of bounding boxes
    
    Returns:
    - bounds (tuple of 4 floats): (xmin, ymin, xmax, ymax)
    """
    if isinstance(img_dir, str):
        img_dir = Path(img_dir)
        
    if not img_dir.is_dir():
        raise ValueError('Input img_dir must be a directory')

    xmin, ymin, xmax, ymax = np.inf, np.inf, -np.inf, -np.inf

    for img_path in img_dir.iterdir():
        if img_path.suffix.lower() not in ['.tif', '.tiff']:
            continue

        ds = rio.open(img_path)
        x0, y0, x1, y1 = ds.bounds

        if x0 < xmin:
            xmin = x0
        if y0 < ymin:
            ymin = y0
        if x1 > xmax:
            xmax = x1
        if y1 > ymax:
            ymax = y1
    return (xmin, ymin, xmax, ymax)

###############################################################################
### Download OSM ###
###############################################################################
def get_osm_highways_in_bounds(bounds,
                               network_type='all',
                               simplify=True,
                               to_save=False,
                               out_dir=None):
    """
    Args:
    - bounds (tuple): min_x, min_y, max_x, max_y
    
    Returns:
    - gdf_e (geopandas DataFrame)
    - (Optionally) saves to a shapefile
    
    """
    bbox = to_osmbbox(bounds) #north, south, east, west (ie. max_lat, min_lat, max_lon, min_lon)
    g = ox.graph_from_bbox(*bbox, network_type=network_type, simplify=simplify)
    gdf_nodes, gdf_edges = ox.graph_to_gdfs(g)
    
    # drop unnecessay columns
    if 'ref' in gdf_nodes.columns:
        gdf_nodes.drop('ref', axis=1, inplace=True)
    
    edge_cols_keep = ['geometry', 'highway','length', 'osmid']
    gdf_edges = gdf_edges[edge_cols_keep]
    
    if to_save:
        now = now_str()
        out_dir = out_dir or f'/tmp/osmnx/highways/{now}'
        if isinstance(out_dir, str):
            out_dir = Path(out_dir)
        if not out_dir.exists():
            out_dir.mkdir(parents=True)
            print(f'{out_dir} is created')
        ox.save_gdf_shapefile(gdf_edges, 'osm_edges', str(out_dir))
        ox.save_gdf_shapefile(gdf_nodes, 'osm_nodes', str(out_dir))
        print(f"Saved gdf_edges to {out_dir}/osm_edges")
        print(f"Saved gdf_nodes to {out_dir}/osm_nodes")
        
    return gdf_nodes, gdf_edges

 
def get_osm_highways_in_dir(src_dir,
                            network_type='all',
                            simplify=True,
                            to_save=False,
                            out_dir='relative',
                            verbose=False):
    """
    Download osm highways in the bounds that is a convex hull of the union of regions 
    of the images in the `src_dir`. For instance, `src_dir = spp.sample_rgb8_dirs['vegas']`
    Args:
    - out_dir: str to a path, 'relative, Path object or None
    
    Returns
    - gdf_nodes, gdf_edges
    - Optionally, if `to_save`, it saves the two gdfs as shapefiles
    """
#     out_dir = out_dir or src_dir.parent/'OSM'
#     if isinstance(out_dir,str):
#         out_dir = Path(out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)
    if out_dir == 'relative':
        out_dir = Path(src_dir).parent/'OSM-Vector/OSM-Road-Lines
    bounds = get_largest_bounds(src_dir)
    gdf_n, gdf_e = get_osm_highways_in(bounds, to_save=to_save, out_dir=out_dir)
    
    if verbose:
        print('src dir: ', src_dir)
        print('n_imgs in the src_dir: ', len(src_dir.ls()))
        print('out osm dir: ', out_dir)
        print("convex hall's bounds: ", bounds)
        
    return gdf_n, gdf_e


###############################################################################
### Tests
###############################################################################                    
def test_get_osm_highways_in_dir():
    rgb_dir = spp.train_rgb_dirs['paris']
    gdf_n, gdf_e = get_osm_highways_in_dir(rgb_dir, to_save=True, verbose=True)
    nprint('Paris OSM highway edges')
    display(gdf_e.sample(100).plot(figsize=(20,20)))