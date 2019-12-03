import numpy as np
from pathlib import Path
import os,sys

from typing import Collection, Union, Optional, Iterable, Mapping

import pdb

################################################################################
### Path setup
################################################################################
PROJ_ROOT = Path(os.getcwd()).parent;print(PROJ_ROOT)
DATA_DIR = PROJ_ROOT/'data/raw'
SRC_DIR = PROJ_ROOT/'scripts'
paths2add = [PROJ_ROOT, SRC_DIR]

# Add project root and src directory to the path
for p in paths2add:
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)
        print("Prepened to path: ", p)
        
        
################################################################################
### Helpers
################################################################################
def get_prob_dist(arr, **kwargs):
    """
    Computes the probability distribution of `arr` by counting the number of occurences in the array
    and divide the counts by the length of the array
    
    - kwargs: kwargs to np.histogram.  If 'density' is specified, it will be overridden to be True always
        - bins: number of bins
        - range
        - weights
        - density: will overwritten to be always True
    """
    
    kwargs.update(density=True)
    hist, bin_edges = np.histogram(arr, **kwargs)
    prob =  hist * np.diff(bin_edges)
    assert np.isclose(np.sum(prob), 1)
    return prob, bin_edges

  
def compute_entropy(prob_dist):
    """
    Computes the entropy of the prob. distribution
    """
    assert np.isclose(sum(prob_dist), 1.)
    return - sum(p * np.log2(p) if p > 0 else 0 for p in prob_dist)


################################################################################
### Prob. Distributions
################################################################################
def multivariate_gaussian(xs, mu, cov):
    """
    Vectorized version of multivariate_gaussian pmf
    This has a computational advantage of computing determinate of the covariance 
    matrix only once for multiple x points, in addition to a vectorized linalg.solve
    operation
    
    Args:
    - xs (np.array): a 2Dim array whose column is a vector of indivisual x in R^m 
    to compute the gaussian. For example, if x is a two dimensional vector, 
    `xs.shape` should be (2, num_xs)
    
    - mu (np.array): 1 dim array for a mu vector
    - cov (np.array): m by m matrix where m = dimension of x (ie. xs.shape[0])
    """
    assert xs.ndim == 2, "xs must be a two dimensional collection of column vectors"
    assert mu.ndim == 2, "mu must be two dimensional"
    assert mu.shape[1] == 1, "mu must be a column vector, ie.shape of (m,1) where m is the data dimension"

    det = np.linalg.det(cov)
    xs_m = xs - mu
#     print(xs-mu)
#     print(xs_m)
    Z = np.sqrt(2*np.pi*det)
#     energy = (x_m.T).dot(np.linalg.inv(cov)).dot(x_m)
    xsTcinv =  (np.linalg.solve(cov, xs_m)).T
#     print('xsTcinv shape: ', xsTcinv.shape)
    xs_m = xs_m.T
#     print('xs_m transposed: ', xs_m.shape)
    energies = np.array([xTcinv.dot(x_m) for (xTcinv,x_m) in zip(xsTcinv, xs_m)])
    return np.exp(-.5*energies)/Z
    
  
################################################################################
### Tests
################################################################################
import scipy.stats as ss
def test_entropies():
    a = [0.33, 0.33, 0.34]
    mine = compute_entropy(a)
    sp = ss.entropy(a, base=2)
    assert np.isclose(mine, sp)
    print('mine: ', mine)
    print('scipy: ', sp)

if __name__ == '__main__':
    test_entropies()
          
      