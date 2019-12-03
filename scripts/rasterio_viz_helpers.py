from __future__ import print_function 
from __future__ import division
# from __future__ import unicode_literals
# from future.utils import raise_with_traceback

import os,sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pdb



# # todo - handle this better?
try:
    import cv2
except ImportError:
    print ('Error importing cv2 -- pass')
    pass

try:
    import rasterio as rio
    from rasterio.plot import reshape_as_raster, reshape_as_image
    print ('Success importing rasterio')
except ImportError:
    print ('Error importing rasterio -- pass')
    pass

################################################################################
### Path setup
################################################################################
PROJ_ROOT = Path(os.getcwd()).parent
DATA_DIR = PROJ_ROOT/'data/raw'
SRC_DIR = PROJ_ROOT/'scripts'
paths2add = [PROJ_ROOT, SRC_DIR]

# print("Project root: ", str(ROOT))
# print("this nb path: ", str(this_nb_path))
# print('Scripts folder: ', str(SCRIPTS))

# Add project root and src directory to the path
for p in paths2add:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        print("Prepened to path: ", p)
      
################################################################################
### Helpers
################################################################################
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
    if not isinstance(rgb, (str, Path)):
        raise TypeError('First input should be a string or path object')
    if not isinstance(mask, (str, Path)):
        raise TypeError('Second input should be a string or path object')
    with rio.open(str(rgb),'r') as ds:
        rgb = reshape_as_image(ds.read())
    with rio.open(str(mask),'r') as ds:
        mask = reshape_as_image(ds.read())
    
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

def read_as_img(fpath):
    """
    Read the image as numpy format (height, width, nChannles) using rasterio
    
    Args:
    - fpath (str or Pathlib.Path)
    
    Returns:
    - (numpy.ndarrarry) image in (height, width, nChannles) format
        - If a grayscale image, nChannels is returned to be 1
        - So, if using `plt.imshow`, need to squeeze the output image
    """
    with rio.open(fpath, 'r') as ds:
        img = reshape_as_image(ds.read())
    return img     

def test_read_img():
    # todo: Link `scripts/naming_helpers.py`
    mtile_path = ('/home/hayley/Data_Spacenet/AOI_2_Vegas_Roads_Train/Mask-Crop/',
                  'mask_AOI_2_Vegas_img1001_0.tif')
    m_path = get_big_raster_path(mtile_path)
    rgb_path = get_rgb_path_from_mask_path(m_path)
    
    
    m_img = read_img(m_path)
    rgb_img = read_img(rgb_path)
    
    f, ax = plt.subplots(1,2, figsize=(20,20))
    ax = ax.flatten()
    ax[0].imshow(m_img.squeeze(), cmap='gray')
    ax[0].set_title (str(Path(m_path).name))
    ax[1].imshow(rgb_img)
    ax[1].set_title(rgb_path)
                     
if __name__ == '__main__':
    # todo: define dataloader
    # dataloader = 
#     test_show_tensor(dataloader)
    pass