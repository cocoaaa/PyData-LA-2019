from __future__ import print_function 
from __future__ import division

import os,sys
import re
from pathlib import Path
import numpy as np

import pdb
from inspect import getmro

import rasterio as rio
from rasterio.plot import reshape_as_image
from shapely.geometry import Polygon

import geoviews as gv
try:
    from itertools import izip as zip
except ImportError: # will be 3.x series
    pass

# if sys.version_info <= (3,5):
try:
    import geopandas as gpd
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
from naming_helpers import extract_city
from output_helpers import nprint
from geo_helpers import filter_gdf_to_polys_within_bounds

###############################################################################
### Helpers ###
###############################################################################
def compare_sp_and_osm_for_tile(rgb_fn, gdf_sp, gdf_osm):
    """
    Creates an interactive gui to explore each tile resulting
    rgb_fn: a uncropped rgb file or cropped tile
    gdf_sp and gdf_osm: ideally covers equal or larger bounds than the bounds of rgb_fn
    """
    ds = rio.open(rgb_fn)
    img = reshape_as_image(ds.read())
    bounds = ds.bounds
    polys_sp = filter_gdf_to_polys_within_bounds(gdf_sp, bounds)
    polys_osm = filter_gdf_to_polys_within_bounds(gdf_osm, bounds)
    
    overlay = (gv.Image(img, bounds=bounds) 
               * gv.Polygons(polys_sp, group='sp', datatype=['list']) 
               * gv.Polygons(polys_osm, group='osm')
              )
    return overlay
    
###############################################################################
### Tests
###############################################################################                    
