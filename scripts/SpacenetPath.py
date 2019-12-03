import os,sys
import numpy as np
from pprint import pprint
from pathlib import Path
Path.ls = lambda x: [o.name for o in x.iterdir()]
###############################################################################
### Add this file's path ###
###############################################################################
file_dir =  os.path.dirname(os.path.realpath(__file__))
# file_dir =  os.path.dirname(os.path.realpath('.'))

print("Importing: ", file_dir)

if file_dir not in sys.path:
    sys.path.insert(0, file_dir)
    
###############################################################################
### Import other helper functions
###############################################################################
from  output_helpers import get_crop_dir, nprint, count_tif
import spacenet_globals as spg

###############################################################################
### TODO
###############################################################################
# class SpacenetPath:
#     """ Set the path for Spacenet data preprocessing
#     and data-loading for train, dev, test
#     """
#     def __init__(self, root=None):
#         """
#         Args:
#         - root (str or Path): path to the root data folder
#         eg: "/home/hayley/Data_Spacenet"
#         """
#         self.root = Path('home/hayley/Data_Spacenet' if root is None else Path(root))
#         self.train_dir 
        
    
# if __name__ == '__main__':
#     dpath = SpacenetPath()
#     print(dpath.root)

###############################################################################
### Sample datasets
###############################################################################
# DATA = Path.home()/"Data_Spacenet/" # if running on arya
DATA = Path.home()/"data/" # if running on local mbp


# Sample dataset
sample_dir = DATA/"SpaceNet_Roads_Sample"
# sample_root_dirs = [sample_dir/ city for city in ["AOI_2_Vegas_Roads_Sample",  
#                                                   "AOI_3_Paris_Roads_Sample", 
#                                                   "AOI_4_Shanghai_Roads_Sample", 
#                                                   "AOI_5_Khartoum_Roads_Sample"]
#                    ]
sample_root_dirs = {
    'vegas': sample_dir/ 'AOI_2_Vegas_Roads_Sample',
    'paris': sample_dir/ "AOI_3_Paris_Roads_Sample",
    'shanghai': sample_dir/ "AOI_4_Shanghai_Roads_Sample",
    'khartoum': sample_dir/ "AOI_5_Khartoum_Roads_Sample"
}

# Original big rgb(16), rgb8bits, mask (uint)
# sample_rgb_dirs = [root/"RGB-PanSharpen" for root in sample_root_dirs]
# sample_rgb8_dirs = [root/"RGB-PanSharpen-8bits" for root in sample_root_dirs]
# sample_mask_dirs = [root/"Mask" for root in sample_root_dirs]
sample_rgb_dirs = dict(
    [(cityname, root/"RGB-PanSharpen") for (cityname, root) in sample_root_dirs.items()]
)
sample_rgb8_dirs = dict(
    [(cityname, root/"RGB-PanSharpen-8bits") for (cityname, root) in sample_root_dirs.items()]
)

sample_mask_dirs = dict(
    [(cityname, root/"Mask") for (cityname, root) in sample_root_dirs.items()]
)

sample_osm_mask_dirs = dict(
    [(cityname, root/"OSM-Mask") for (cityname, root) in sample_root_dirs.items()]
)

# Vector files (1300x1300) 
sample_road_vec_dirs = dict(
    [ (cityname, root/"geojson/spacenetroads") for (cityname,root) in sample_root_dirs.items()]
)
sample_buffer_vec_dirs = dict(
    [ (cityname, root/"geojson/buffer") for (cityname, root) in sample_root_dirs.items()]
)

###############################################################################
### Training datasets
###############################################################################
vegas_root = DATA/"AOI_2_Vegas_Roads_Train/"
paris_root = DATA/"AOI_3_Paris_Roads_Train/"
shanghai_root = DATA/"AOI_4_Shanghai_Roads_Train/"
k_root = DATA/"AOI_5_Khartoum_Roads_Train/"

train_root_dirs = {'vegas': vegas_root, 
                   'paris': paris_root, 
                   'shanghai': shanghai_root,
                   'khartoum': k_root}

# Original big rasters: rgb(16), rgb8bits, mask (uint)
train_rgb_dirs = dict( 
    [(cityname, root/"RGB-PanSharpen") for (cityname, root) in train_root_dirs.items()]
)
train_rgb8_dirs = dict( 
    [(cityname, root/"RGB-PanSharpen-8bits") for (cityname, root) in train_root_dirs.items()]
)
train_mask_dirs = dict( 
    [(cityname, root/"Mask") for (cityname, root) in train_root_dirs.items()]
)

train_osm_mask_dirs = dict(
    [(cityname, root/"OSM-Mask") for (cityname, root) in train_root_dirs.items()]
)
# Cropped 100x100 tif image tiles
## Not yet processed
# train_rgb_tile_dirs = dict(
#     [(cityname, get_crop_dir(dirname)) for (cityname, dirname) in train_rgb_dirs.items()]
# )
train_rgb8_tile_dirs =  dict(
    [(cityname, get_crop_dir(dirname)) for (cityname, dirname) in train_rgb8_dirs.items()]
)
train_mask_tile_dirs =  dict(
    [(cityname, get_crop_dir(dirname)) for (cityname, dirname) in train_mask_dirs.items()]
)
     
# vector file dirs
train_road_vec_dirs = dict(
    [(cityname, root/"geojson/spacenetroads") for (cityname,root) in train_root_dirs.items()]
)
train_buffer_vec_dirs = dict(
    [(cityname,root/"geojson/buffer") for (cityname,root) in train_root_dirs.items()]
)

###############################################################################
### Generated tile datasets
###############################################################################
tile_root = DATA/"Tile"
tile_w, tile_h = 650, 650
def get_tile_dir(tile_width: int,
                 tile_height: int,
                 city: str,
                 img_type: str):
    assert img_type in spg.IMG_TYPES
    return tile_root/f"{tile_width}x{tile_height}/{city}/{img_type}"

def test_get_tile_dir():
    for city in spg.CITIES:
        for img_type in ['rgb','mask']:
            print(city, img_type)
            tdir = get_tile_dir(tile_w, tile_h, city, img_type)
            print(tdir.exists(), len(tdir.ls()))
            
def get_tile_dirs(tile_width, tile_height, verbose=False):
    """
    Return 
    tile_dirs (dict): a dictionary of all image type's tile directory.
        - key: one of spg.IMG_TYPES
        - value: a dictionary whose keys are one of spg.CITIES
    
    Eg:
        tile_dirs['vegas']['rgb'] = Path('./data/Tile/650x650/vegas/rgb')
    """
    tile_dirs = {}
    for city in spg.CITIES:
        tile_dirs[city] = {}
        for img_type in spg.IMG_TYPES:
            tile_dirs[city][img_type] = get_tile_dir(tile_width, tile_height, city, img_type)

        if verbose:
            nprint(city, 
                   f'{[(img_type, len(tile_dir.ls())) for (img_type, tile_dir) in tile_dirs[city].items()]}')
            
    return tile_dirs
    



###############################################################################
### Load paths to datasets
###############################################################################
def get_train_fns(city):
    """
    Args:
    - city (str): one of 'vegas', 'paris', 'shanghai', 'khartoum'
    
    Returns:
    - d_fnames (dict): 
    keys = ['rgb8', 'mask', 'rbuffer'], 
    d_fnames[key] = list of filenames (str)
    """
    
    rgb8_dir = train_rgb8_dirs[city]
    mask_dir= train_mask_dirs[city]
    osm_mask_dir = train_osm_mask_dirs[city]
    buff_dir = train_buffer_vec_dirs[city]
    
    
    rgb8_fns = [p for p in rgb8_dir.iterdir() if p.suffix == '.tif']
    mask_fns = [p for p in mask_dir.iterdir() if p.suffix == '.tif']
    osm_mask_fns = [p for p in osm_mask_dir.iterdir() if p.suffix == '.tif']

    buff_fns = [p for p in buff_dir.iterdir() if p.suffix == '.geojson']

    return {'rgb8': rgb8_fns,
            'mask': mask_fns,
            'osm_mask': osm_mask_fns,
            'rbuffer': buff_fns
           }
def get_sample1300_dirs(city):
    """
    Args:
    - city (str): one of 'vegas', 'paris', 'shanghai', 'khartoum'
    
    Returns:
    - dictionary indexed by img_type (dict): 
    
        keys = ['rgb8', 'mask', 'osm_mask', 'rbuffer'], 
        d[key] = Path object pointing to the correpsonding img_type's directory
    """
    rgb8_dir = sample_rgb8_dirs[city]
    mask_dir= sample_mask_dirs[city]
    osm_mask_dir = sample_osm_mask_dirs[city]
    buff_dir = sample_buffer_vec_dirs[city]
    
    return {
        'rgb8': rgb8_dir,
        'mask': mask_dir,
        'osm_mask': osm_mask_dir,
        'rbuffer': buff_dir
    }

def get_train1300_dirs(city):
    """
    Args:
    - city (str): one of 'vegas', 'paris', 'shanghai', 'khartoum'
    
    Returns:
    - dictionary indexed by img_type (dict): 
    
        keys = ['rgb8', 'mask', 'osm_mask', 'rbuffer'], 
        d[key] = Path object pointing to the correpsonding img_type's directory
    """
    rgb8_dir = train_rgb8_dirs[city]
    mask_dir= train_mask_dirs[city]
    osm_mask_dir = train_osm_mask_dirs[city]
    buff_dir = train_buffer_vec_dirs[city]
    
    return {
        'rgb8': rgb8_dir,
        'mask': mask_dir,
        'osm_mask': osm_mask_dir,
        'rbuffer': buff_dir
    }
    
###############################################################################
### Simple sample datasets for Vegas
###############################################################################
# RGB8_DIR = Path('/home/hayley/Data_Spacenet/SpaceNet_Roads_Sample/'
#                 'AOI_2_Vegas_Roads_Sample/RGB-PanSharpen-8bits/')
# RGB16_DIR = Path('/home/hayley/Data_Spacenet/SpaceNet_Roads_Sample/'
#                  'AOI_2_Vegas_Roads_Sample/RGB-PanSharpen/')
# MASK_DIR = Path('/home/hayley/Data_Spacenet/SpaceNet_Roads_Sample/'
#                 'AOI_2_Vegas_Roads_Sample/Mask/')
# RGB8_TILE_DIR = RGB8_DIR.parent/"RGB-PanSharpen-8bits-Crop"
# RGB16_TILE_DIR = RGB16_DIR.parent/"RGB-PanSharpen-Crop"

# RGB8_FILES = [f for f in RGB8_DIR.iterdir() if f.suffix == '.tif']
# RGB16_FILES = [f for f in RGB16_DIR.iterdir() if f.suffix == '.tif']
# MASK_FILES = [f for f in MASK_DIR.iterdir() if f.suffix == '.tif']
# print(len(RGB8_FILES), len(MASK_FILES))

###############################################################################
### Get citywise RGB, Mask, buff vector folders
###############################################################################
# def get_city_data_dirs(cityname):
#     return train_rgb

###############################################################################
### Tests
###############################################################################
def test_count():
    print("Number of tif files in train_rgb8_dirs: ")
    for city,rgb8_dir in train_rgb8_dirs.items():
        nprint(city, count_tif([rgb8_dir]))

def test_get_train_fns():
    for city in ['vegas', 'paris', 'shanghai', 'khartoum']:
        d_fnames = load_train_fns(city)
        nprint(city)
        for k,v in d_fnames.items():
            print("\t", k,len(v))
            
def test_sample_dir_dict_generation():
    dir_dicts = np.array([sample_rgb_dirs, sample_rgb8_dirs, sample_mask_dirs,
                          sample_road_vec_dirs, sample_buffer_vec_dirs])
    for dir_dict in dir_dicts:
#         pprint(dir_dict)
        for dir_path in dir_dict.values():
            assert dir_path.exists()
