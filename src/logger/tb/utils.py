from pathlib import Path
import numpy as np
from typing import List, Union, Collection, Callable, Any, NewType, TypeVar, Optional

import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

import pdb 
from helpers import now2str

class TBWrapper():
    def __init__(self, tb):
        self.tb = tb
    
    def log_params_hist(self,model, global_step, log_grad=True, filter_fns=None) -> None:
        """
        Write each model parameter's values as a list to Tensorboard writer as a histogram
        Args:
        - model (nn.Module)
        - tb (tensorboard.writer.SummaryWriter)
        - log_grad (bool): if True, log both the parameter's histogram as well as the gradient's histogram
        - filter_fns (Iterable of filter_fn to filter parameters)
            - each filter_fn takes in `nn.Parameter` and returns Boolean indicating if it passes the filter function
            - eg: 
            ```python
            def is_trainable(param: nn.Parameter): 
                return param.requires_grad():

            def is_linear(param: nn.Parameter):
                return param.weight is not None and param.bias is not None #not rigorous
            ```
        """
        if filter_fns is None: filter_fns = []
        for name, param in model.named_parameters():
            if len(filter_fns)>0:
                passes = [filter_fn(param) for filter_fn in filter_fns]
                if not np.all(passes):
                    continue
            self.tb.add_histogram(name, param.data, global_step)
            if log_grad and param.grad is not None:
                self.tb.add_histogram(f'{name}.grad', param.grad, global_step)
        
#     def add_scalar(
def tb_params_hist(model, 
                   tb, 
                   global_step,
                   log_grad=True,
                   filter_fns=None) -> None:
    """
    Write each model parameter's values as a list to Tensorboard writer as a histogram
    Args:
    - model (nn.Module)
    - tb (tensorboard.writer.SummaryWriter)
    - log_grad (bool): if True, log both the parameter's histogram as well as the gradient's histogram
    - filter_fns (Iterable of filter_fn to filter parameters)
        - each filter_fn takes in `nn.Parameter` and returns Boolean indicating if it passes the filter function
        - eg: 
        ```python
        def is_trainable(param: nn.Parameter): 
            return param.requires_grad():
        
        def is_linear(param: nn.Parameter):
            return param.weight is not None and param.bias is not None #not rigorous
        ```
    """
    if filter_fns is None: filter_fns = []
    for name, param in model.named_parameters():
        if len(filter_fns)>0:
            passes = [filter_fn(param) for filter_fn in filter_fns]
            if not np.all(passes):
                continue
        tb.add_histogram(name, param.data, global_step)
        if log_grad and param.grad is not None:
            tb.add_histogram(f'{name}.grad', param.grad, global_step)
        
# def test_tb_params_hist():
#     writer = SummaryWriter(log_dir=f'./test/{now2str()}')
    
#     model = nn.Linear(2,1)
#     loss_fn = nn.MSELoss(reduction='mean')
#     optimizer = optim.SGD(model.parameters(), lr=0.01)
#     train_dl = DataLoader(train_ds, batch_size=10)
#     for i, batch_sample in enumerate(train_dl):
#         batch_x, batch_y = batch_sample['X'], batch_sample['y']
#         pred_y = model(batch_x)
#         loss = loss_fn(pred_y, batch_y)
        
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         tb_params_hist(model, writer, i)
        
        