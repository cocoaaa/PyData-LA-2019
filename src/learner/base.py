from pathlib import Path
import numpy as np
import random
from typing import List, Dict, Union, Collection, Callable, Any, NewType, TypeVar, Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pdb 
from tqdm.autonotebook import tqdm

# My libs
from utils import to_device
from experiment import GaussianLR
# from metrics import runningScore, averageMeter

from abc import ABC, abstractmethod

class LearnerBase():
    """
    Abstract class for different Learn class
    Each Learner will be specifically defined depending on the problems
    For example, segmentation task, classification task, etc
    """
    def __init__(self, 
                 model: nn.Module,
                 dataloaders: Dict[str, DataLoader],
                 device: torch.device,
                 tbw=None,
                 global_epoch=0
                ):
        self.global_epoch = global_epoch # keep track of the number of backprop counts in unit of iteration
        self.model = model; to_device(model, device)
        self.device = device
        self.train_dl = dataloaders.get('train')
        self.val_dl = dataloaders.get('val', None)
        self.test_dl = dataloaders.get('test', None)
        
        # easy access to datasets
        self.train_ds = self.train_dl.dataset
        self.val_ds, self.test_ds = None, None
        if self.val_dl is not None:
            self.val_ds = self.val_dl.dataset
        if self.test_dl is not None:
            self.test_ds = self.test_dl.dataset
            
        # tensorboard writer wrapper
        self.tbw = tbw
    
    def one_epoch(self, optim_fn, loss_fn, metric_fns=None, to_tb=True):
        """
        Do one epoch of training -> return train stats, and evaluate on validation and/or test set(s) ->
        return val stats
        
        Args:
        - optim_fn: has already configured learning rate, (optionally momentum, weight decay, etc)
            - eg: partial(optim.SGD, lr=0.01)
            - so that it is ready to become a optimizer by optim_fn(self.model.parameters())
        - metric_fns (Dict[[str,metric_fn]])
        - to_tb (bool): if True, write to the tensorboard writer
        """
        train_result = self.train_one_epoch(optim_fn, loss_fn, metric_fns)
        if self.val_dl is not None:
            val_result = self.evaluate(loss_fn, metric_fns)
        if self.test_dl is not None:
            test_result = self.evaluate(loss_fn, metric_fns)
            
        if to_tb:
            self.log_curr_params_hist()
    
    def mult_epochs(self, n_epochs, optim_fn, loss_fn, metric_fns=None, to_tb=True):
        for ep in range(n_epochs):
            self.one_epoch(optim_fn, loss_fn, metric_fns, to_tb)

    def log_curr_params_hist(self, global_step=None):
        global_step = global_step or self.global_epoch
        self.tbw.log_params_hist(self.model, global_step)
        
    @abstractmethod
    def train_one_epoch(self, optim_fn, loss_fn, metric_fns):
#         self.train_one_epoch(self.model, self.train_dl, optimizer, loss_fn, self.device, metric_fns)
#         optimizer = optim_fn(self.model.parameters())
        pass
        
    @abstractmethod
    def _evaluate(self, loss_fn, metric_fns):
#         self.evaluate(self.model, self.test_dl, loss_fn, self.device, metric_fns)
        pass

            
            
class GLRLearner(LearnerBase):
    def train_one_epoch(self, optim_fn, loss_fn, metric_fns):
        optimizer = optim_fn(self.model.parameters())
        result = GaussianLR.train_one_epoch(self.model, self.train_dl, optimizer, loss_fn, self.device, metric_fns)
        return result
    
    def evaluate(self, loss_fn, metric_fns):
        return GaussianLR.evaluate(self.model, self.val_dl, loss_fn, self.device, metric_fns)
        
                 