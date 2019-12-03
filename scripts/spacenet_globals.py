#spacenet data preprocessing global variables
import numpy as np
from Road import *

# Road type value definition
## Defined as enum in class Road
# MOTORWAY = 1
# PRIMARY = 2
# SECONDARY = 3
# TERTIARY = 4
# RESIDENTIAL = 5
# UNCLASSIFIED = 6
# CART = 7
CITIES = ['vegas', 'paris', 'shanghai', 'khartoum']

G_NUMERIC_COLS = ['bridge_typ',
                   'lane_numbe',

 'lane_number',
 'one_way_type',
 'paved',
 'road_id',
 'road_type']

G_COLS = ['bridge_typ',
         'heading',
         'lane_numbe',
         'lane_number',
         'one_way_ty',
         'paved',
         'road_id',
         'road_type',
         'origarea',
         'origlen',
         'partialDec',
         'truncated',
         'geometry']


# road_type to width mapping
# G_WIDTHS = {Road.Motorway: 3.5,
#             Road.Primary: 3.5,
#             Road.Secondary: 3.,
#             Road.Tertiary: 3.,
#             Road.Residential: 3.,
#             Road.Unclassified: 3.,
#             Road.Cart: 3.,
#             }
# G_WIDTHS = {Road.MOTORWAY.value: 3.5,
#             Road.PRIMARY.value: 3.5,
#             Road.SECONDARY.value: 3.,
#             Road.TERTIARY.value: 3.,
#             Road.RESIDENTIAL.value: 3.,
#             Road.UNCLASSIFIED.value: 3.,
#             Road.CART.value: 3.,
#             }

G_DROP_COLS = ['heading', 
               'lane_numbe', 
               'origarea', 
               'origlen', 
               'partialDec', 
               'truncated']

G_GEOM_COLS = ['geometry', 
              'buff_geo']

## For OSM Road data
OSM_ROAD_DROP_COLS = ['name', 
                      'waterway', 
                      'aerialway', 
                      'barrier', 
                      'man_made', 
                      'z_order', 
                      'other_tags']

# raster types
## during preprocessing, we use the raster types for naming default output directory
## when it's not specified 
## eg. used in crop_helpers.create_tiles_with_latlon
IMG_TYPES = ['rgb', 'mask', 'osm']