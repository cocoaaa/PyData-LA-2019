import os, time
from collections import defaultdict
import numpy as np
from pathlib import Path
import pdb



################################################################################
# IO Helpers
################################################################################
def get_timestamp():
    return dt.datetime.now().strftime("%y%m%d_%H%M%S")
    
def get_temp_fname(prefix='', suffix=''):
    tstamp = get_timestamp()
    return ''.join(['_'.join([prefix, tstamp]), suffix])

import imageio
fname = '../outputs/levelset/2019-08-02/sdStar1_f_-1_dt_0.001_t_0_0.3.gif'
def tensor_from_video(fname):
    reader = imageio.get_reader(fname, 'ffmpeg')
    imgs = np.transpose( np.stack(list(reader.iter_data()), axis=0), (0,3,1,2))
    imgs = np.expand_dims(imgs, 0)
    assert imgs.ndim == 5, f'Video data must be 5 dimensional: batchsize, timesteps, C,H,W: {imgs.dim}'
    return torch.from_numpy(imgs)

    
################################################################################
# Python Object Inspection Helpers
################################################################################
def get_mro(myObj):
    return myObj.__class__.mro()

def show_attrs(myObj):
    import holoviews as hv
    atts = [str(att) for att in dir(myObj) if not att.startswith('_')]
    n_atts = len(atts)
    return hv.Table(pd.DataFrame(atts, columns=['att']))

def nprint(*args, header=True):
    if header:
        print("="*80)
    for arg in args:
        pprint(arg)

def attr_print(myObj):
    attrs = [att for att in dir(dimx) if not att.startswith('_')]
    pprint(attrs)
    
################################################################################
# Floating Points precision cleanup
################################################################################
def clip_close_values(arr):
    """
    A method to clean up floating points in an array so that "equal" values 
    (decided by np.isclose function) are assigned to only a unique representative
    value. For instance, arr=[ 0.000000000001, 0.000000000009] will appear to have
    different colors by both matplotlib.pyplot and holoviews, but we want the visualization
    to match the fact they actually represent the same value even though they have 
    slightly different numerical values (due to some operations that generated the array,
    eg. np.gradient.
    
    We use python dictionary and the computation complexity it \theta(arr.numele()) 
    """
    unique_vals = defaultdict(int)
    clipped = np.empty_like(arr)
    for i, val in enumerate(arr.flat):
        i_tuple = np.unravel_index(i, arr.shape)

        curr_keys = np.asarray(list(unique_vals.keys()))
        key_found = np.unique(curr_keys[np.isclose(val, curr_keys)])
        try:
            clipped[i_tuple] = key_found[0]
            unique_vals[key_found[0]] += 1 
#             print("--\tFound key: ", key_found[0])

        except IndexError:
#             print("First time!: ", val)
            clipped[i_tuple] = val
            unique_vals[val] += 1 
        except:
            print("This should never be printed")
    return clipped
#     return clipped, unique_vals


################################################################################
# Profiling Helpers
################################################################################
def timeit(method):
    # src: https://is.gd/knmjbS"
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()

        nprint(f'{method.__name__} took: {te-ts:.3f}sec')
        return result

    return timed

################################################################################
# Geocoding Conversion Helpers
################################################################################
## geocoders: addr string -> lat, lon & vice versa
def get_latlon(addr_str):
    """
    Given a string address, resolve its location in (lat, lon) degrees
    """
    geolocator = Nominatim(user_agent="myawesomeproj")
    loc = geolocator.geocode(addr_str)
    return loc.latitude, loc.longitude

def get_addr(lat, lon):
    """
    Given lat, lon in degrees, return a resolved address as a string.
    Eg: get_addr_str(52.509669, 13.376294) -> "Potsdamer Platz, Mitte, Berlin, 10117..."
    """
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    loc= geolocator.reverse(f'{lat}, {lon}')
    return loc.address


################################################################################
# Coordinate System Conversion Helpers
################################################################################
def UV2angMag(U,V):
    """
    Given: U,V as two MxN np.ndarrays for x, y coordinates (eg. outputs of np.meshgrid) such that U[j,i] and V[j,i] corresponds to x,y components of a vector.
    This function computes the angle and magnitude of the vector and stores in the 
    output arrays. Conceptually, it computes: 
        
        angle = np.empty(U.shape)
        mag = np.empty(U.shape)
        for j in V[:,0]:
            for i in U[0,:]:
            v = vec(U[j,i], V[j,i])
            angle[j,i] = angle(v)
            mag[j,i] = magnitude(v)
           
    This is useful to visualize the vectorfield using holoviews's hv.VectorField.
    
    Args:
    - U,V (MxN np.ndarray): encodes X,Y coordinate grids respectively
    
    Returns:
    - angle, mag: tuple of MxN np.ndarray that encode ang (or mag) for the grid space
    That means, angle[j][i] at (X[j][i],Y[j][i]) location
    """
    mag = np.sqrt(U**2 + V**2)
    angle = (np.pi/2.) - np.arctan2(U/mag, V/mag)

    return (angle, mag)

################################################################################
# Time Type Conversion Helpers
################################################################################
def to_datetime(tlist):
    """
    Operation to convert any non-python datetime object in a list of time objects
    to (python) dt.datetime object. 
    This is useful for making the xaxis of any time-series plots human-readable, 
    possibly due to a bug in bokeh.
    """
    mro = get_mro
    
    # check if the topmost mro is datetime.datetime object
    if all( (mro(t)[0] == dt.datetime) for t in tlist):
        return tlist

    print('Converting timevalues to python datetime object')
    dt_list= list(map(lambda t: t.to_pydatetime(), tlist))
    return dt_list
        
################################################################################
# URL helpers
################################################################################
def dict2json(d):
    return JSON(d)

def display_dict2json(d):
    display(JSON(d))

def is_valid_url(path):
    import requests
    r = requests.head(path)
    return r.status_code == requests.codes.ok
    
    
################################################################################
# Pandas helpers
################################################################################
def cols_with_null(df):
    """
    Returns any columnnames with any null value
    Args:
    - df(pandas.DataFrame)
    Returns:
    - cols (list): list of strings, each of which correseponds to the column name
    with any null values
    """
           
    cols = [c for c in df.columns if df[c].isnull().values.any()]
    return cols

def select(df, selection, axis):
    """
    Returns a new dataframe with its axis reduced to the elements in `selection`
    
    Args: 
    - selection (list): a list of column names (if axis=1) or row indices (if axis=0)
    - axis: 0 for row selection, 1 for column selection
    
    Example:
    ```python
    df = pd.DataFrame({'a': [1,2,3], 'b': [10,20,30]})
    
    # select certain columns, in a certain order
    new_cols = ['b']
    print(select(df, new_cols, axis=1))
    
    # Reorder columns
    new_cols = ['b','a']
    print(select(df, new_cols, axis=1)
    
    # Select a subset of rows
    r_sels = [0]
    print(select(df, r_sels, axis=0)
    
    ```
    """
    if axis not in [0,1]:
        raise ValueError(f'axis must be either 0 or 1:  {axis}')
    if axis == 0:
        return df.loc[df.index.isin(selection)]
    if axis == 1:
        return df[selection]
    
    
def reorder_cols(df, new_cols):
    """
    Returns a new dataframe with column orders switched to the new_cols
    Args:
    - df (pd.DataFrame)
    - new_cols (list): a list of new column names. Can be a subset of df.columns
        in which case, only specified columns will be selected in the given order
    Returns:
    - pd.DataFrame: a new dataframe object with the columns selected
    """
    if not set(new_cols).issubset(set(df.columns)):
        raise ValueError('Input column list must be a subset of the original df')
    return df[new_cols]
    

################################################################################
# Xarray helpers
################################################################################


################################################################################
# holoviews helpers
# todo: move to hv_utils.py
################################################################################
def relabel_elements(ndoverlay, labels):
    """
    ndOverlay is indexed by integer
    labels (str or iterable of strs)
    length of hv elements in the overlay must equal to the length of labels
    """
    import holoviews as hv
    from itertools import cycle
    if isinstance(labels, str):
        labels = [labels]
    if isinstance(labels, list) and len(labels) != len(ndoverlay):
        raise ValueError('Length of the labels and ndoverlay must be the same')
        
        
    it = cycle(labels) 
    relabeled = hv.NdOverlay({i: ndoverlay[i].relabel(next(it)) for i in range(len(ndoverlay))})
    return relabeled



## string manipulation
def to_fname(s):
    return '_'.join(
    list(map(lambda s: s.strip(',_-:;').lower(), s.split()))
)
    
    
################################################################################
# Tests
################################################################################
def test_get_mro():
    print( get_mro('somestring') )

def test_nprint():
    nprint(10, 100)
    nprint("line1", "line2")
    nprint(['ele1', 'ele2','ele3'])
    nprint('line1', 'line2', 'line3')

def test_dict2json():
    d = {'user': ['hayley', 'wanting', 'bob', 'yijun'],
         'age': [10,12,11, 9]
        }
    print(d)
    print(dict2json(d))

def test_cols_with_null():
    pass

def test_clip_close_values_1():
    tiny1 = 0.02020202020202011
    tiny2 = 0.020202020202020332
    original = np.atleast_2d([tiny1,tiny1, tiny2, tiny2, tiny1])
    clipped = clip_close_values(original)
    f,ax = plt.subplots(1,2)
    ax[0].imshow(original)
    ax[0].set_title('original')
    ax[1].imshow(clipped)
    ax[1].set_title('clipped')
    
def test_clip_close_values_2():
    def linear_array():
        h,w = 100,100
        xs = np.linspace(-1,1,num=w)
        ys = np.linspace(-1,1,num=h)[::-1]
        zz = np.empty((w,h))
        for i in range(len(xs)):
            for j in range(len(ys)):
                zz[j,i] = ys[j] 
        return (xs,ys,zz)
    xs,ys,zz = linear_array()
    original = np.atleast_2d(zz)
    clipped = clip_close_values(original)
    f,ax = plt.subplots(1,2)
    ax[0].imshow(original)
    ax[0].set_title('original')
    ax[1].imshow(clipped)
    ax[1].set_title('clipped')    


def test_is_valid_url():
    url = 'http://workflow.isi.edu/MINT/FLDAS/FLDAS_NOAH01_A_EA_D.001/2019/04/FLDAS_NOAH01_A_EA_D.A20190401.001.nc'
    print('url: ', url)
    print('url exists? :', is_valid_url(url))
    
def test_get_latlon():
    addr = '424 West Pico Blvd, Los Angeles, 90015'
    print(addr)
    print(get_latlon(addr))

def test_get_addr():
    lat, lon = -5, 37 # somewhere in africa
    print('lat, lon: ', lat, lon)
    print(get_addr(lat, lon))

def test_to_fname():
    s = 'Pyin Hpyu Gyi, Republic of the Union of Myanmar: Google- Earth'
    print('original: ', s)
    print('to_fname: ', to_fname(s))
    
def test_reorder_cols():
    df = pd.DataFrame({'a': [1,2,3], 'b': [10,20,30]})
    reordered = reorder_cols(df, ['b','a'])
    print('original: ')
    display(df)
    print('reordered: ')
    display(reordered)
                             
def test_all():
    test_get_mro()
    test_nprint()
    test_dict2json()
    test_cols_with_null()
    test_is_valid_url()
    test_get_latlon()
    test_get_addr()
    test_to_fname()
    
    
if __name__ == '__main__':
    test_all()
