import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from model.fcn import fcn32s, fcn16s, fcn8s
from scheduler.cyclic import TriangleLR, ConstLR
from scheduler.utils import show_lr_generator
from tqdm.autonotebook import tqdm


import holoviews as hv
from holoviews import opts

import pdb

def lr_range_test(model, dataloader, loss_fn, optimizer, lr_scheduler, device, 
                  n_iters=None, beta=0.2, tol_factor=4, print_every=None):
    """
    Assumes optimizer and model's parameters are linked. 
    Assumes all parameters of the model are in the same device (= `device`).
    
    Uses exponentially averaged loss with parameter beta:
        avg_loss_{t+1} = beta * avg_loss_{t} + (1-beta) * loss_{t+1}
    beta must be very small to make sense, in the worst case, smaller than 0.5.
    
    When loss > `tol_factor` * min_loss, terminate the iterations and return 
    
    TODO: add running weight size average
    """

    if n_iters is None:
        n_iters = len(dataloader) # one-epoch
        
    model.train()

    # Initialize loop
    count = 0
    lrs = []
    avg_losses = []
    min_loss = float('inf')
    losses = [] # for learning sake (to be removed)
    pbar = tqdm(total=n_iters)
    while True:
        for x,y in dataloader:         # for labelled dataset,  eg. segmentation
            stop_cond = count > 0  and avg_losses[-1] > tol_factor * min_loss
            if count >= n_iters or stop_cond:
#                 pdb.set_trace()
                pbar.close()
                return lrs, losses, avg_losses
            
            x,y = x.to(device), y.to(device).long()#
#             pdb.set_trace()
            pred = model(x)
            loss = loss_fn(pred, y)
            losses.append(loss.item()) #todo: remove 
            
            prev_avg = avg_losses[-1] if count > 0 else 0.0
            avg_loss = beta*prev_avg + (1-beta)*loss.item()
            avg_loss /= (1 - beta**(count+1)) #correction term
            avg_losses.append(avg_loss)
            
            # update the min_loss if new loss is smaller
            if avg_loss < min_loss:
                min_loss = avg_loss

            # backprop (i.e. update the weights)
#             lr_scheduler.step() #updates optimizer's learning rate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            lr_scheduler.step() #updates optimizer's learning rate for next ieration
#             print('iter: ', count, " , ", 'lr: ', lrs[-1])
#             pdb.set_trace()
            count += 1
            pbar.update(1)
            if print_every and count%print_every == 0:
                print(f'\nIter: {count}')
                print(f'lr:  {lrs[-1]}')


def lr_range_test_reconstruction(model, dataloader, loss_fn, optimizer, lr_scheduler, device,
                  n_iters=None, beta=0.2, tol_factor=4, print_every=None):
    """
    Assumes optimizer and model's parameters are linked.
    Assumes all parameters of the model are in the same device (= `device`).

    Uses exponentially averaged loss with parameter beta:
        avg_loss_{t+1} = beta * avg_loss_{t} + (1-beta) * loss_{t+1}
    beta must be very small to make sense, in the worst case, smaller than 0.5.

    When loss > `tol_factor` * min_loss, terminate the iterations and return

    TODO: add running weight size average
    """

    if n_iters is None:
        n_iters = len(dataloader)  # one-epoch

    model.train()

    # Initialize loop
    count = 0
    lrs = []
    avg_losses = []
    min_loss = float('inf')
    losses = []  # for learning sake (to be removed)
    pbar = tqdm(total=n_iters)
    while True:
        for sample in dataloader:  # for unlabelled dataset, eg. vae
            stop_cond = count > 0 and avg_losses[-1] > tol_factor * min_loss
            if count >= n_iters or stop_cond:
                #                 pdb.set_trace()
                pbar.close()
                return lrs, losses, avg_losses
            x = sample['x']
            x = x.to(device)
            #             pdb.set_trace()
            pred = model(x)
            loss = loss_fn(pred, x) #reconstruction
            losses.append(loss.item())  # todo: remove

            prev_avg = avg_losses[-1] if count > 0 else 0.0
            avg_loss = beta * prev_avg + (1 - beta) * loss.item()
            avg_loss /= (1 - beta ** (count + 1))  # correction term
            avg_losses.append(avg_loss)

            # update the min_loss if new loss is smaller
            if avg_loss < min_loss:
                min_loss = avg_loss

            # backprop (i.e. update the weights)
            #             lr_scheduler.step() #updates optimizer's learning rate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lrs.append(optimizer.param_groups[0]['lr'])
            lr_scheduler.step()  # updates optimizer's learning rate for next ieration
            #             print('iter: ', count, " , ", 'lr: ', lrs[-1])
            #             pdb.set_trace()
            count += 1
            pbar.update(1)
            if print_every and count % print_every == 0:
                print(f'\nIter: {count}')
                print(f'lr:  {lrs[-1]}')


def run_experiment(n_iters, model_gen, dl, loss_fn, optim_gen, lr_scheduler, device, 
                   print_every=None, seed=1, to_show=True):
    """
    
    Args:
    - model_generator (Callable): returns a new model
        - Must accept two arguments, `device` and `seed`
        - It returns a model object put in `device` with any weight initialization 
        random-seeded at `seed
    - optim_gen (Callable): Must take in 'model.parameters()'
        - eg: functools.partial(torch.optim.Adam, lr=1e-3)
    - seed (None or int): random seed for clean model (model weights)
        - None if randomness in initializing model weights is desired
        - any other int to set the seed
        
    - lr_scheduler: TriangleLR or ConstLR
    
    """
    model = model_gen(device=device, seed=seed)
    optimizer = optim_gen(model.parameters())
    lrs, losses, avg_losses = lr_range_test(model, dl, loss_fn, optimizer, lr_scheduler, device, 
                                            n_iters=n_iters, print_every=print_every);
    
    # Visualization
    if to_show:
        hv_lr = show_lr_generator(lr_scheduler, n_iters)

        layout = (
            hv_lr.opts(color='red', ylim=(lr_scheduler.min_lr, lr_scheduler.max_lr)) +
            hv.Curve(losses,  label='loss').opts(color='blue') 
        )

        display(
            layout.opts(
            opts.Overlay(shared_axes=False),
            opts.Curve(padding=0.1, width=800, axiswise=True,shared_axes=False)
            ).cols(1)
        )
    
    return model, lrs, losses, avg_losses
