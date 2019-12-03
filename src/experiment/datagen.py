# synthesizer.data.py
## Helpers for creating synthetic datasets for controlled experiments
import os, sys, random
import numpy as np
import scipy
from scipy.stats import multivariate_normal

import holoviews as hv
from holoviews import opts
from helpers import npify, get_module_name
import pdb 
def make_linear_data(x_generator, n_samples, true_w, true_b, noise_fn=None):
    """
    Generate a dataset with normal source distribution of X and target y's
    under the relationship of:
    
        y = x@true_w + trub_b + additive_noise

    Args:
    - x_generator (Callable): a callable that takes in a list of dims and generates samples
        - eg: np.random.rand which can be called as `np.random.rand(n_samples, n_features)`
        - eg: scipy.stats.multivariate_normal
            gauss2d_gen = scipy.stats.multivariate_normal([0,0], [[1,0],[0,1]])
            samples2d = gauss_2d.gen(n_samples)
            So, if you are using functions from scipy.stats module, pass in rv.rvs as the callable 
            
    - n_samples (int): number of data samples 
    - true_w (np.ndarray): true weights for X->y mapping. Its shape is (n_features,)
    - true_b (float): bias term for the single neuron
    - noise_fn: additive noise

    Returns:
    - dataset (dict): keys are 'X' and 'y'. 
        - X is a (n_samples, n_features) shaped np.ndarray of floats
        - y is a (n_samples,) shaped np.ndarray of floats
    """
    true_w = npify(true_w).reshape((-1,1))
    n_features = len(true_w)
    if 'scipy.stats' in get_module_name(x_generator):
        ## scipy.stats function
        ## arguments is only n_features
        X = x_generator(n_samples)
    elif 'numpy' in get_module_name(x_generator):
        ## numpy function, and takes all dims
        X = x_generator(n_samples, n_features)
    print('Data X created: ', X.shape)
    
    X = x_generator(n_samples, n_features)
    y = X@true_w + true_b

    if noise_fn:
        noises = noise_fn(n_samples)
        y += noises.reshape(y.shape)
        
    return {"X": X.astype(np.float32), # for compatibility with pytorch weight dtype
             "y": y.astype(np.float32)} # already satisfied

def gaussian_x_linear_mapping(x_gen_mu, x_gen_cov, allow_singular, n_samples, true_w, true_b, noise_fn=None):
    """
    x data comes from d-dim multivariate gaussian distribution with mean=x_gen_mu, cov=x_gen_cov
    where `d` is deduced from the length of input `true_w`
    """
    x_gen_cov = npify(x_gen_cov)
    n_features = len(true_w)
    assert len(x_gen_mu) == n_features, f"dim of x source distribution's mean doesn't match {len(x_gen_mu)}"
    assert x_gen_cov.shape == (n_features, n_features),f"Shape of x source's cov doesn't match: {x_gen_cov.shape}"
    x_generator = multivariate_normal(x_gen_mu, x_gen_cov, allow_singular=allow_singular).rvs
    
    return make_linear_data(x_generator,n_samples, true_w, true_b, noise_fn)

def possion_x_linear_mapping():
    pass

def gaussian_noise_1d(n_samples, mu, std):
    """
    Returns n_samples number of points from 1dim gaussian(mu, std)
    - noises (np.ndarray): shape is (n_samples,1)
    """
    
    rv = scipy.stats.norm(loc=mu, scale=std)
    return rv.rvs(n_samples)


## tests
def test_make_linear_data():
    true_w = [1,2]
    true_b = 10.
    n_samples = 1000
    x_gen = multivariate_normal([0,0], [[1,0],[0,1]]).rvs

    data = make_linear_data(x_gen, n_samples, true_w, true_b)
    x, y = data['X'], data['y']
    plt.scatter(x[:,0], x[:,1])

def test_gaussian_noise_1d():
    noises = gaussian_noise_1d(1000, 0, 1)
    plt.hist(noises)

def test_gaussian_x_linear_mapping():
    x_gen_mu = [0,0]
    x_gen_cov = [[1,0], [0,1]]
    true_w = [1,2]
    true_b = 10.
    n_samples = 1000
    noise_std = 0.
    data = gaussian_x_linear_mapping(x_gen_mu, x_gen_cov, n_samples, true_w, true_b)
    
    x, y = data['X'], data['y']
    hv.Scatter3D((x[:,0], x[:,1], y.squeeze())).opts(
        opts.Scatter3D(color='z', 
                       colorbar=True, 
                       width=800, 
                       height=800, 
                       title=f'''
                       true w, b: {true_w}, {true_b}
                       noise: {noise_std}
                       '''))