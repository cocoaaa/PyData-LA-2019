import os,sys,time
from copy import deepcopy
from pathlib import Path
from collections import Counter
import pdb

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import osmnx as ox
import joblib
# import dill

from shapely.geometry import *
from shapely.ops import unary_union

from IPython.display import display

###############################################################################
### gdf constructors ###
###############################################################################
def gseries2gdf(gseries):
    gdf = gpd.GeoDataFrame(gseries)
    gdf = gdf.rename(columns={0:'geometry'}).set_geometry('geometry')
    if hasattr(gseries, 'crs'):
        gdf.crs = gseries.crs
    return gdf

def geom2gdf(geom, crs):
    "A single geom object to gdf of length 1"
    
    gseries = gpd.GeoSeries([geom], crs=crs)
    return gseries2gdf(gseries)


def df2gdf(df, geom_col, orig_crs, target_crs=None):
    """
    Args:
    - df (pandas.DataFrame): one column must be a list of shapely.geometry objects
    - geom_col (int or string): name of the column to be used as the GeoDataFrame's 
        `geometry` column
    - orig_crs: original crs
        - eg: {'init': 'epsg:4326'} 
    - target_crs (optional): if not None, project the gdf to this crs before returning
    
    Returns:
    - gdf (geopandas.GeoDataFrame)
    """
    gdf = gpd.GeoDataFrame(df, crs=orig_crs, geometry=geom_col)
    if target_crs:
        gdf = gdf.to_crs(target_crs)
    return gdf

def list2gdf(geoms, orig_crs, target_crs=None):
    """
    Create a geopandas DataFrame object from a list of shapely geometry objects
    
    Args:
    - geoms (list of shapely.geometry objects)
    - orig_crs: a proper crs of the geometry list
    - target_crs: if not None, reproject the dataFrame to this crs
        - if None, keep the orig_crs as the output dataFrame's crs
    
    Returns:
    - gdf (geopandas.DataFrame) whose 'geometry' column is set to the input list `geoms`
    
    """
#     gdf = gpd.GeoDataFrame(geoms, crs=orig_crs, geometry=0)
#     if target_crs:
#         gdf = gdf.to_crs(target_crs)
    geom_col = 0
    return df2gdf(geoms, geom_col, orig_crs, target_crs=target_crs)

###############################################################################
### geojson IO ###
###############################################################################
def to_geojson(df, fname):
    df.to_file(fname, driver='GeoJSON')
    
###############################################################################
### Point sampling from lines ###
###############################################################################
def createPointsAlongLine(geom, distance, pts, addStart=True):
    """
    Args:
    
    - geom: LineString or MultiLineString
    - distance (float): distance between points to be sampled along the line(s)
    - pts (list:non-const): points will be added to this list
    - addStart (bool, default=True): True to add the 
    """
    
    if not isinstance(geom, (MultiLineString, LineString)):
        print(type(geom))
        pdb.set_trace()
        raise ValueError(
            "Input geometry should be either MultiLineString or LineString: {}".format(type(geom)))
     
    if isinstance(geom, MultiLineString):
        for g in geom:
            createPointsAlongLine(g, distance, pts) 
    elif isinstance(geom, LineString): #basecase
        if addStart:
            pts.append(Point(geom.coords[0]))
        
        length = geom.length
        currentdistance = distance
        while currentdistance < length: 
            pt = geom.interpolate(currentdistance)
            pts.append(pt)
            currentdistance = currentdistance + distance

def createPointsOnRow(row, distance, new_points,
                     debug=False):
    """
    Args:
    - row (namedTuple) with Index=True
    - distance (float): distance between each sample points along the line
    - new_points (list): non-const. This is the new point (row) collector
    """

    pts = []
    createPointsAlongLine(row.geometry, distance, pts)
    
    for p in pts:
        point_info = {'road_idx': row.Index,
                      'road_id': row.road_id,
                      'road_type': row.road_type,
                      'geometry': p}
        new_points.append(point_info)
        
    if debug:
        # debug log
        print("="*80)
        print("row", row.Index, ", length: ", row.length)
        print("\tdistance: ", distance)
        print("\tnum sampled points: ", len(pts))

def createPointsOnDf(gdf, distance, debug=False):
    """
    Given a geopandas DataFrame of LineString and MultiLineString data, 
    sample points at the interval of `distance` and return a geopandas 
    DataFrame of all the sampled points.
    The sampled points will keep its parent road's information in its column:
        -'road_index', 'road_id' and 'road_type' from its parent road
        -'geometry': Point in the same UTM crs as the input `gdf`
    Args:
    - gdf (geopandas.DataFrame): road data consisting of LineString or MultiLineString
        objects in a proper UTM crs
    - distance (float): distance between sampled points along each line in meters
    - debug (boolean): 
    
    Returns:
    - df_points (geopandas.DataFrame): sampled points dataframe in the same CRS as the input
        - colnames: `road_index`, `road_id`, `road_type`, `geometry`
    """
    new_points = []
    for row in gdf.itertuples(index=True):
        createPointsOnRow(row, distance, new_points, debug=debug)
    
    df_points = gpd.GeoDataFrame(new_points, crs=gdf.crs, geometry='geometry')
    
    if debug:
        phead(df_points)
        print('number of sampled points: ', len(df_points))
    return df_points

    
###############################################################################
### Create bbox from bounds ###
###############################################################################
def create_bbox_from_bounds(bounds):
    """
    Args:
    - bounds (list): (minx, miny, maxx, maxy)
    Returns:
    - Polygon (shapely.geometry)
    """
    minx, miny, maxx, maxy = bounds
    p1 = (minx, miny)
    p2 = (minx, maxy)
    p3 = (maxx, maxy)
    p4 = (maxx, miny)
    
    return Polygon([p1, p2, p3, p4])

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

###############################################################################
### Filtering Operations on Geopandas ###
###############################################################################
def filter_gdf_by_geom_types(gdf, geom_types):
    flag = gdf.geom_type.isin(geom_types)
    return gdf[~flag]

def filter_empty_geoms(gdf, inplace=False):
    gdf_copy = gdf if inplace else deepcopy(gdf)
    return gdf_copy[~gdf_copy.geometry.is_empty]
    
def filter_gdf_to_polys_within_bounds(gdf, bounds):
    """
    Args:
    - gdf (geopandas.DataFrame)
    - bounds (tuple or rasterio.bounds object): encodes (xmin, ymin, xmax, ymax)
    
    Returns:
    - a list of Polygons within the bounds
    """
    roi = bounds2poly(bounds)
    gdf_int = gdf.intersection(roi)
    gs_filtered = gdf_int[~gdf_int.is_empty]
    print("Number of intersecting rows: ", len(gs_filtered))
    return gs_filtered.to_list()

def filter_gdf_to_gs_within_bounds(gdf, bounds, verbose=False):
    """
    Any geometry in each row of `gdf` that has a non-empty intersection with the area
    surrounded by the `bounds` will be returned as a geopandas.GeoSeries object.
    Note that the returned GeoSeries object has a single column, ie. does not preserve
    any columns in the input gdf. 
    
    If you want to have the original gdf structure preserved, with its `geometry` column
    is modified to a shapely object of its intersection with the bounds, use `crop_gdf_to_bounds`
    
    
    Args:
    - gdf (geopandas.DataFrame)
    - bounds (tuple or rasterio.bounds object): encodes (xmin, ymin, xmax, ymax)
    
    Returns:
    - Filtered gdf
    """
    roi = bounds2poly(bounds)
    has_int = ~gdf.intersection(roi).is_empty
    gdf_filtered = gdf[has_int]
    if verbose:
        print("Number of intersecting rows: ", len(gdf_filtered))
        if len(gdf_filtered)> 0:
            print('Bounds of filtered gdf: ', gdf_filtered.total_bounds)
    return gdf_filtered

def crop_gdf_to_bounds(gdf, bounds, inplace=False, remove_empty=False):

    roi = bounds2poly(bounds)
    cropped = gdf.intersection(roi)

    gdf_copy = gdf if inplace else  deepcopy(gdf)
    gdf_copy.geometry = cropped
    
    if remove_empty:
        gdf_copy = filter_empty_geoms(gdf_copy, inplace=True) 
        # ^this `inplace` can be always True regardless of the input kwarg `inplace`

    return gdf_copy

def crop_gdf_to_(gdf, bounds, inplace=False, remove_empty=False):

    roi = bounds2poly(bounds)
    cropped = gdf.intersection(roi)

    gdf_copy = gdf if inplace else  deepcopy(gdf)
    gdf_copy.geometry = cropped
    
    if remove_empty:
        gdf_copy = filter_empty_geoms(gdf_copy, inplace=True) 
        # ^this `inplace` can be always True regardless of the input kwarg `inplace`

    return gdf_copy

def get_polys_at_lonlat(gdf, lon, lat):
    p = Point(lon,lat)
    gdf_selected =  gdf[gdf.intersects(p)]
    print(len(gdf_selected))
    return gdf_selected
#     pdb.set_trace()
#     return gv.Polygons(gdf_selected)#.opts(color='red')
    

                      
###############################################################################
### Tests ###
###############################################################################
def test_gseries2gdf():
    pass

def test_list2gdf():
    geoms = [Point(0,0), Point(1,0), Point(1,1)]
    gdf = list2gdf(geoms, {'init': 'epsg:4326'})
    
    print('geoms: ', geoms)
    print('gdf: ', gdf.head())
    print('\tcrs: ', gdf.crs, "; geom_type: ", gdf.geom_type[:5])
    
def test_df2gdf():
    pass
    
def test_create_bbox_from_bounds():
#     bounds = [0,0,1,1]
#     bounds = [0,0,1,2]
    bounds = [0,0,2,1]
#     bounds = [0,0,1,1]
#     bounds = [0,0,1,1]

    bbox = create_bbox_from_bounds(bounds)
    display(bbox)
    
def run_tests():
    test_gsereis2gdf()
    test_list2gdf()
    test_df2gdf()

    test_create_bbox_from_bounds()

if __name__ == "__main__":
    test_list2gdf()

#     run_tests()