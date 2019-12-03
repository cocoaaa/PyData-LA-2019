import os
import sys

import numpy as np
from collections import Counter
import holoviews as hv


def hv_batch(batch):
    """
    Creates a holoviews overlay of RGB (for x data) and Image (for y, the label mask)
    
    batch is a tuple of (4d tensor, 3d tensor) as returned by a dataloader
    or a (3d tensor, 2d tensor) corresponding to a data pair returned by a dataset 
    """
    x, y = batch
    if x.dim() == 4 and y.dim() == 3:
        print('More than one image in the batch. Showing only the first one...')
        x,y = x[0],y[0]
    assert x.dim() == 3 and y.dim() == 2
    
    x_np = x.detach().numpy().transpose((1,2,0))
    y_np = y.detach().numpy()
    overlay = hv.RGB(x_np) + hv.Image(y_np, group='mask')
    return overlay

def hv_dataloader(dataloader, idx):
    """
    Creates an overlay of x and y pair from the ith iteration of the dataloader
    If the dataloader's batchsize > 1, this shows just the first pair of (x,y)
    
    Args:
    - dataloader: a dataloader that returns a pair of (batch_x, batch_y)
    - idx: ith iteration of dataloader. Currently this is not the most efficient way 
        to visualize the dataloader. 
        
    Example:
    dmap_sample_dl = hv.DynamicMap(lambda idx: hv_dataloader(sample_train_dl, idx),
                                kdims=['idx'])
    dmap_sample_dl.redim.values(idx=list(range(len(sample_train_dl))))
    """
    for i, (x,y) in enumerate(dataloader):
        if i != idx:
            continue
            
        if x.dim() == 4 and y.dim() == 3:
            print('More than one image in the batch. Showing only the first one...')
            x,y = x[0],y[0]
        assert x.dim() == 3 and y.dim() == 2
        
        x_np = x.detach().numpy().transpose((1,2,0))
        y_np = y.detach().numpy().astype(np.int)
        overlay = hv.RGB(x_np) + hv.Image(y_np, group='mask')
        return overlay
 
    
## Visualiza a batch of datasets
def show_batch(batch, max_num=2):
    """
    batch (tuple of tensors): returned by Dataloader
    Visualize the batch x and y overlay as hv Elements
    """
    batch_x, batch_y = batch
    bs = len(batch_x)
    assert bs == len(batch_y), len(batch_y)
    for i,(x, y) in enumerate(zip(batch_x, batch_y)):
        if i >= max_num:
            break
        x_np = x.detach().numpy().transpose((1,2,0))
        y_np = y.detach().numpy()
        overlay = hv.RGB(x_np) + hv.Image(y_np, group='mask')
        display(overlay)
dd        
    
def show_filter_batch(batch, n_cols=4, max_show=16):
    """
    Show a batch of filter tensors using holoviews
    
    Args:
    - batch (torch.tensor): 4 dimensional torch.tensor. (bs, nC, H, W)
    
    Returns:
    - holoviews layout object
    """
    bs = len(batch)
    batch_np = batch.cpu().numpy().transpose((0, -2, -1, -3))[:max_show]
    print('Batch in numpy: ', batch_np.shape)
    n_rows = int(float(bs/n_cols))
    
    layout = []
    for img in batch_np:
        layout.append(hv.Image(img, group='mask').opts(shared_axes=False))
    return hv.Layout(layout).cols(n_cols)
    
    