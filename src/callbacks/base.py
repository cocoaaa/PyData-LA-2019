import re
from pathlib import Path
import numpy as np
import random
from abc import ABC, abstractmethod
import pdb

from typing import List, Dict, Union, Collection, Callable, Any, NewType, TypeVar, Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from tqdm.autonotebook import tqdm

# My libs
from utils import camel2snake, to_device
from helpers import get_cls_name, now2str

class Callback():
    """Base class for callbacks that will be called at each stage of training/evaluating
    as desired"""
    _order = 0

    def __repr__(self):
        return get_cls_name(self)

    def __getattr__(self, item):
        return getattr(self.runner, item)  # delegate to its target runner object
        # ^ This allows the Callback an access into the Runner's current state, eg. Learner, optimizer, lr_scheduler

    @property
    def name(self):
        name = re.sub(r'Callback$', '', self.__class__.__name__)
        if name:
            return f'cb_{camel2snake(name)}'
        else:
            return f'cb_{now2str()}'

    def set_runner(self, runner):
        self.runner = runner

    #         self.model = self.runner.learner.model
    def unset_runner(self):
        self.runner = None

    def __call__(self, event_name, **kwargs):
        getattr(self, f'on_{event_name}')(**kwargs)

    def on_begin_fit(self, **kwargs) -> None:
        pass

    def on_begin_epoch(self, **kwargs) -> None:
        pass

    ## for train one epoch
    def on_begin_epoch_train(self, **kwargs) -> None:
        pass

    def on_after_epoch_train(self, **kwargs) -> None:
        pass

    ## for validate one epoch
    def on_begin_epoch_val(self, **kwargs) -> None:
        pass

    def on_after_epoch_val(self, **kwargs) -> None:
        pass

    def on_after_fit(self, **kwargs) -> None:
        pass

    ## Batch passing: common operation for train and val for an epoch
    def on_begin_batch(self, **kwargs) -> None:
        pass

    def on_after_pred(self, **kwargs) -> None:
        pass

    def on_after_loss(self, **kwargs) -> None:
        pass

    def on_after_backward(self) -> None:
        pass

    def on_after_step(self, **kwargs) -> None:
        pass

    def on_after_batch(self, **kwargs) -> None:
        pass

    def on_after_epoch(self, **kwargs) -> None:
        pass

class TrainEvalCallback(Callback):
    """Callback that counts the number of iterations for the duration of `fit`,
     and properly sets training/eval mode"""
    def on_begin_fit(self) -> None:
        "Send the model parameters to current runner's device, and sets the training iter to 0"
        print(f'{self.name}.on_begin_fit')
        to_device(self.model, self.runner.device)
        self.runner.train_iter = 0
        for name,param in self.model.named_parameters():
            print(name, param.data)


    def on_begin_epoch_train(self, **kwargs) -> None:
        "Set the train iter to zero, and set the runner and model parameter's mode to train"
        print(f'{self.name}.on_begin_epoch_train')
        print(f'train mode (runner,model)? {self.runner.training, self.model.training}')
        # self.runner.training = True
        # self.learner.model.train()


    def on_begin_epoch_val(self, **kwargs) -> None:
        "Set the runner and model's parameters to eval mode"
        print(f'\n{self.name}.on_begin_epoch_val')
        print('training?: (runner,model)', self.runner.training, self.model.training)
        # self.runner.training = False
        # self.learner.model.eval()

    def on_after_batch(self, **kwargs) -> None:
        "Increment train iter count by 1 (only if the runner is in training mode)"
        if not self.runner.training:
            return
        self.runner.train_iter += 1

    def on_after_fit(self, **kwargs) -> None:
        print(f'{self.name}.on_after_fit')
        print('Total train iters: ', self.runner.train_iter)
        for name,param in self.model.named_parameters():
            print(name, param.data)

class GradCallback(Callback):
    def on_after_backward(self):
        if not self.runner.training:
            return
        print(f'{self.name}.on_after_step')
        print('Gradient: ')
        for name, p in self.model.named_parameters():
            print(f'{name}.grad', p.grad.data)















































































