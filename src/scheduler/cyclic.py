import torch 
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm
import pdb
import typing

class TriangleLR():
    def __init__(self, min_lr:float, max_lr:float, stepsize:int):
        """
        min_lr (float): lower bound of the learning rate
        max_lr (float): upper bound of the lr
        stepsize (int): stepsize in number of iterations
            - 2*stepsize = cycle_length in iterations
        """
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.stepsize = stepsize
        self.slope = (self.min_lr - self.max_lr)/self.stepsize
        self.n_called = 0
        
    def __call__(self, x:int):
        """
        x (int): iteration index 
        """
        self.n_called += 1
        x = x%(2*self.stepsize)
#         print(x)
        return self.slope * abs(x - self.stepsize) + self.max_lr
    
    def step(self):
        pass
    
    def reset(self):
        self.n_called = 0
        
    
class ConstLR():
    def __init__(self, lr):
        """
        Returns a constant LR
        """
        self.lr = lr
        self.n_called = 0
    
    def __call__(self, x:int):
        """
        x (int): iteration index 
        """
        self.n_called += 1
        return self.lr
    
    def step(self):
        pass
    
    def reset(self):
        self.n_called = 0
    
    
    
class CyclicTLR():
    def __init__(self, 
                 optimizer:Optimizer, 
                 min_lr:float, max_lr:float, stepsize:int,
                 last_iter=-1,
                 verbose:bool=False):
        """
        stepsize (int): in units of iteration, not epochs
        last_iter (int): iteration to start the lr scheduling
        """
        assert isinstance(optimizer, Optimizer)
        super().__init__()
        self.optimizer = optimizer

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.stepsize = stepsize
        self.slope = (self.min_lr - self.max_lr)/self.stepsize
        self.step_count = 0
        self.last_iter = last_iter
        self.verbose = verbose
        
    def get_lr(self, x:int):
        """
        Get learning rate at iteration index, `x`
        This can be call independently of where you are in the epoch (or iterations)
        So, it's an "absolute" learning rate computing function.
        
        Args:
        - x (int): iteration index 
        """
        x = x%(2*self.stepsize)
        return self.slope * abs(x - self.stepsize) + self.max_lr

    def step(self):
        """
        Compute the [absolute] learning rate at last_iter + 1 index (of iterations),
        and updates the optimizer's parameters with the lr
        """
        x = self.last_iter + 1
        lr = self.get_lr(x)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        if self.verbose:
            print('lr: ', lr)
        self.last_iter += 1
        self.step_count += 1
    
    def reset(self):
        """ 
        Resets count of step functions called
        """
        self.step_count = 0
        
        