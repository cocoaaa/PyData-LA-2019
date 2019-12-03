from __future__ import print_function 
from __future__ import division

import os,sys
import re
from pathlib import Path
# Path list 
Path.ls = lambda x: [p.name for p in x.iterdir()]

import numpy as np
import osmnx as ox
import geopandas as gpd
import gdal, osr, ogr
import rasterio as rio
from rasterio.plot import reshape_as_image


import pdb
from inspect import getmro
from typing import Union, Iterable, Collection, Mapping

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
from naming_helpers import extract_city, get_osm_mask_fn
from output_helpers import nprint

def generate_osm_buffer_raster_for(rgb_fns, gdf_osmbuffer, to_save=True, verbose=False):
    """
    Gen
    - rgb_fns (Iterable of str or Path): contains rgb filenames which defines 
    the bounds of OSM roadlines to be filtered from `gdf_osmbuffer`
    - gdf_osmbuffer (gpd.GeoDataFrame): constains road buffer geometries 
    
    Usage:
    - rgb_fns and gdf_osmbuffer are both on the same city (eg. 'vegas')
        - eg:
        ```
        city = 'vegas'
        rgb_dir = spp.train_rgb8_dirs[city]
        rgb_fns = [ p for p in rgb_dir.iterdir() if p.suffix in ['.tif', '.tiff']]

        osm_dir = rgb_dir.parent/'OSM'
        osm_rbuff_fn = osm_dir/'osm-buffers/osm-buffers.shp'
        nprint(city, rgb_dir, osm_dir, osm_rbuff_fn.exists())
        gdf_osm_rbuff = gpd.read_file(osm_rbuff_fn)
        
        generate_osm_buffer_raster_for(rgb_fns, gdf_osm_rbuff)
        ```
    """
    for fn in rgb_fns:
        ds = rio.open(fn)
        img = reshape_as_image(ds.read())
        bounds = ds.bounds
        out_fn = get_osm_mask_fn(fn)
        if not out_fn.parent.exists():
            out_fn.parent.mkdir()
            print(f'{out_fn.parent} is created')
        osm_arr = gdf2array(gdf_osmbuffer, 
                            src_rgb_path=fn, 
                            dst_mask_path=out_fn, 
                            verbose=verbose)
        if verbose:
            nprint('in fn: ', fn)
            print('out fn: ', out_fn)




def get_buffered_gdf(gdf, col_buffer_r='radius'):
    """
    Generates a buffer of radius specified in the column,`gdf[col_buffer_r]`,
    and modifies the input `gdf` by changing its `geometry' to this new buffer geometry.
    Thus, it maintains the original crs and all columns (that is not `geometry` column)
    
    For buffer generation, it first project the geomtry objects to utm using `osmnx` 
    (eg. latlon-> UTM projection) and uses geopandas's `buffer` function
    
    Args:
    - gdf (geopandas.GeoDataFrame)
    - col_buffer_r (str): name of the buffer radius column in `gdf`
    
    Returns:
    - the input gdf object whose `geometry` is changed to the newly generated buffer geometries
    """
    orig_crs = gdf.crs
    gdf = ox.project_gdf(gdf)
    
    # create the buffer for each road (row)
    buff = []
    for i in range(len(gdf)):
        b = gdf.iloc[[i]].buffer(gdf[col_buffer_r].values[i])
        b = b.item()
        buff.append(b)
    gdf['buff_geom'] = buff
    
    # set the crs of this output to the original crs (ie. lat/lon)
    gdf['geometry'] = gdf['geometry'].to_crs(orig_crs)
    
    # change the geometry column to buffer geometries to use .to_crs 
    gdf.set_geometry('buff_geom', inplace=True)
    gdf['buff_geom'] = gdf.geometry.to_crs(orig_crs)
    
    # To save, we need to make a copy with the 'geometry' 
    # column is set to the buffer geometries
    # And, this damn thing need to be of GeoSeries (not pd.Series)
    gdf.drop('geometry', axis=1, inplace=True)
    gdf.set_geometry('buff_geom', inplace=True)

    gdf.rename(columns={'buff_geom':'geometry'}, inplace=True)
    gdf.set_geometry('geometry', inplace=True)
    gdf.crs = orig_crs
    print('crs: ', gdf.crs)

    return gdf


def gdf2array(gdf, 
              src_rgb_path,
              dst_mask_path,
              verbose=False):
    """
    Given the geopandas dataframe with 'geometry' column, 
    rasterize each object in the 'geometry' column onto the `out_file` 
    image by using road_type(s) as the burn value(s).
    The `src_rgb_path` file is used to set the metadata of the output file,
    eg. the resolution in x and y axes and transformation matrix.
    
    Args:
    - gdf (gpd.GeoDataFrame): the shapes to be rasterized. 
        Must have columns 'geometry' and 'road_type'
        - eg: "buffer_AOI_2_..._img1434.geojson"
    - src_rgb_path (str or Path): path to the rgb file
        - eg: "RGB-PanSharpen_AOI_2_..._img1434.tif"

    - dst_mask_path (str or Path): path to the output filename
        - eg: "mask_AOI_2_..._img1434.tif"
    
    Returns:
    - out_arr (np.array): 8bit unit type `.tif` format file. Each value indicates 
        the road type at that location
    """
    if not ('geometry' in gdf.columns and 'road_type' in gdf.columns):
        raise IOError("Input dataframe must have 'geometry' and 'road_type' columns")
                      
    NoDataVal = 0 #0 indicates no road buffer geometry, ie. no road detected
    
    if not isinstance(src_rgb_path, str):
        src_rgb_path = str(src_rgb_path)
    rgb_ds = gdal.Open(src_rgb_path)
    
    # set output datasource
    out_ds = gdal.GetDriverByName('GTiff').Create(str(dst_mask_path), 
                                                  rgb_ds.RasterXSize, rgb_ds.RasterYSize, 1, #number of layers
                                                  gdal.GDT_Byte) #8bit uint
    out_ds.SetGeoTransform(rgb_ds.GetGeoTransform())
    
    # set projection
    proj = osr.SpatialReference()
    proj.ImportFromWkt(rgb_ds.GetProjectionRef())
    out_ds.SetProjection(proj.ExportToWkt())
    
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(NoDataVal)
    

    # set output driver for burning buffer geometry to a raster layer 
    ## Read the buffer data in memory
    mem_driver = ogr.GetDriverByName('MEMORY')
    mem_ds = mem_driver.CreateDataSource('memData') 
    ## Open the memory datasource with write access
    tmp = mem_driver.Open('memData', 1) #0: read-access, 1:write-access
    
    ## todo: each road-type dataframe will form a layer
    ## or, just burn different road-type group with different burn value
    ## to the same, single layer
#     out_lyr = mem_ds.CreateLayer("mask", srs=proj, geom_type=ogr.wkbPolygon)
#     ## The output layer will have an attribute/field of "road_type" 
#     ## Set the road_type attribute in the output layer
#     rt_field = ogr.FieldDefn("road_type", ogr.OFTInteger)
#     out_lyr.CreateField(rt_field)
#     feat_def = out_lyr.GetLayerDefn()
    
    for rt, group in gdf.groupby('road_type'):
        rt = int(rt)
        # debug
        if verbose:
            nprint()
            print("Processing road type: ", rt)
        
        # for each road-type, we need to write to a new output layer because 
        # gdal.RasterizeLayer takes only one burn value for the input layer (I think?)
        ## todo: get the geom_type from the input vector file's geometry column type
        rt_lyr = mem_ds.CreateLayer("mask", srs=proj, geom_type=ogr.wkbPolygon)
        ## The output layer will have an attribute/field of "road_type" and geometry
        ## Set the road_type attribute in the output layer
        rt_field = ogr.FieldDefn("road_type", ogr.OFTInteger)
        rt_lyr.CreateField(rt_field)
        feat_def = rt_lyr.GetLayerDefn()
        
        # Add each road data point to the output layer
        for i in range(len(group)):
            df_row = group.iloc[[i]]
            # extract geometry from each row
            # geom is a shapely.geometry.polygon
            geom = df_row['geometry'].values[0]

            # create a ogr.feature to write to the layer (in memory)
            # using thie road_type filed value and the geometry
            out_feat = ogr.Feature(feat_def)
            out_feat.SetField("road_type", rt)
            out_feat.SetGeometry(ogr.CreateGeometryFromWkt(geom.wkt))
            ## add this feature to the output (in memory) layer
            rt_lyr.CreateFeature(out_feat)
            # clean things up
            out_feat.Destroy()
        
        if verbose:
            print("Finished for road-type: ", rt)
            print("\tnumber of features in the output layer: ", rt_lyr.GetFeatureCount())
            print("\twhich should be the same as ", len(group))

        err = gdal.RasterizeLayer(out_ds, [1], rt_lyr, burn_values=[rt])
        if err != 0:
            print("error: ", err)
    
    # get an array of the rasterized buffer layer
    mask = out_ds.ReadAsArray() #todo: change the axis order?
#     plt.imshow(mask)
    
    return mask                
###############################################################################
### Tests
###############################################################################                    

def test_generate_osm_buffer_raster_for_vegas():
    rgb_fns = [ p for p in spp.sample_rgb8_dirs['vegas'].iterdir() if p.suffix in ['.tif', '.tiff']]
    all_osmbuffers = joblib.load('../data/processed/osmbuffers_per_city_all_train.pydict')
    gdf_osmbuffer = all_osmbuffers['vegas']
    generate_osm_buffer_raster_for(rgb_fns, gdf_osmbuffer)