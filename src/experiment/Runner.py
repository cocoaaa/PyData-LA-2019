import re
from pathlib import Path
from typing import List, Dict, Union, Collection, Callable, Any, NewType, TypeVar, Optional
from contextlib import contextmanager, nullcontext

from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pdb
from tqdm.autonotebook import tqdm

# My libs
import defaults
from utils import to_device, replacing_yield, camel2snake, now2str, get_cls_name
from experiment.utils import save_model, load_model
from callbacks.base import Callback
# from metrics import runningScore, averageMeter


class Runner():
    def __init__(self, learner, loss_fn,
                 optim_fn=defaults.optim_fn, lr=defaults.lr, device=None, cbs=None, path=None):
        self.learner = learner
        self.loss_fn = loss_fn
        self.optim_fn = optim_fn
        self.lr = lr
        self.create_opt() #sets self.opt
        self.device = device or self.learner.device
        try:
            self.split_fn = self.learner.split_fn
        except AttributeError as e:
            print('Warning: failed to register split method')
            print(e)
        self.cbs = []
        self.add_cbs(cbs)
        self.training = None #todo: maybe link to the learner's state
        self.path = Path(path) if path is not None else Path(f'./temp-{now2str()}')
        if not self.path.exists(): self.path.mkdir(parents=True, exist_ok=False)

    def __getattr__(self, item):
        # delegate to self.learner
        return getattr(self.learner, item)

    def add_cbs(self, cbs: List, verbose=False) -> None:
        if cbs is None or len(cbs)==0: return
        # if getattr(self, 'cbs', None) is None: self.cbs = []
        [self.add_cb(cb, verbose) for cb in cbs]; return

    def remove_cbs(self, cbs: List , verbose=False) -> None:
        if cbs is None or len(cbs)==0: return
        [self.remove_cb(cb, verbose) for cb in cbs]; return

    def add_cb(self, cb: Callback, verbose=False):
        """Register cb assuming self.cbs list already exists"""
        if hasattr(self, cb.name):
            raise ValueError(f'self.{cb.name} is already registered')
        setattr(self, cb.name, cb)
        cb.set_runner(self)
        self.cbs.append(cb)
        if verbose: print(f'Registered cb: {cb.name}')

    def remove_cb(self, cb: Callback, verbose=False):
        if cb.runner is self: cb.unset_runner()
        if hasattr(self, cb.name): delattr(self, cb.name)
        if cb in self.cbs:
            self.cbs.remove(cb)
            if verbose: print(f'Deleted cb: {cb.name}')

    @property
    def cb_names(self):
        "Print registered callback with registered attribute names"
        return [cb.name for cb in self.cbs]

    @property
    def sorted_cbs(self):
        return sorted(self.cbs, key=lambda cb: cb._order)

    @contextmanager
    def adding_cbs(self, cbs):
        self.add_cbs(cbs)
        yield
        self.remove_cbs(cbs)

    @contextmanager
    def replacing_loss_fn(self, loss_fn):
        return replacing_yield(self, 'loss_fn', loss_fn)

    def create_opt(self):
        """Create self.opt from current self.optim_fn and self.lr, whatever they are"""
        self.opt = self.optim_fn(self.model.parameters(), self.lr)

    # self(event_name) executes each cb.on_{event_name} method for all callbacks in self.cbs
    def __call__(self, event_name):
        [self._call_one(cb, event_name ) for cb in self.sorted_cbs]

    def _call_one(self, cb, event_name, verbose=False):
        if verbose: print(f'{cb.name} is called on event {event_name}')
        getattr(self, cb.name)(event_name)

    def one_batch(self,
                  batch_i: int,
                  batch) -> None:
        """One iteration of forward (Optionally, backward if self.is_training)
        using current self.model, self.opt(imizer) and self.loss_fn"""
        # cache current iteration's state
        self.iter = batch_i
        self.xb, self.yb = self.split_fn(batch)
        self.xb, self.yb = self.xb.to(self.device), self.yb.to(self.device)

        self('begin_batch')
        self.pred = self.model(self.xb)
        self('after_pred')

        self.loss = self.loss_fn(self.pred, self.yb)
        self('after_loss')

        if not self.training:
            return
        self.loss.backward()
        self('after_backward')

        self.opt.step()
        self('after_step')

        self.opt.zero_grad()
        self('after_batch')
        # todo: maybe clear the cache?

    def all_batches(self):
        "Iterate once over the entire self.dl"
        self.n_iter = len(self.dl)
        for batch_i, batch in enumerate(self.dl):
            self.one_batch(batch_i, batch)

    def _do_epoch_train(self, epoch:int ) -> None:
        """Run one epoch of training
        epoch (int): index of current epoch
        1. Set self.dl to train dataloader
        2. Iterate over self.dl using self.one_epoch method
        """
        self.epoch = epoch
        self.dl = self.train_dl
        self.training = True
        self.model.train()

        self('begin_epoch_train')
        self.all_batches()
        self('after_epoch_train')

    def _do_epoch_validate(self, epoch:int , dl=None) -> None:
        """Run one epoch of validation
        epoch (int): current epoch index
        """
        self.epoch = epoch
        self.dl = dl or self.val_dl
        if self.dl is None:
            return #todo: CancleEpochValidate exception
        self.training = False
        self.model.eval()

        self('begin_epoch_val')
        with torch.no_grad():
            self.all_batches()
        self('after_epoch_val')
        # return self.recorder.values[-1] #todo

    def fit(self, n_epoch, *, cbs=None, loss_fn=None,
            lr=None, wd=defaults.wd, recreate_opt=False) -> None:
        cb_ctx = nullcontext() if cbs is None or len(cbs) ==0 else self.adding_cbs(cbs)
        loss_ctx = self.replacing_loss_fn(loss_fn) if loss_fn is not None else nullcontext()

        with cb_ctx, loss_ctx:
            if recreate_opt:
                self.create_opt()
            # self.opt.set_hypers(wd=wd, lr=self.lr if lr is None else lr)

            # begin fit
            self.n_epoch = n_epoch
            self.loss = torch.tensor(0.)
            self('begin_fit')
            for epoch in range(n_epoch):
                self('begin_epoch')
                self._do_epoch_train(epoch)
                self._do_epoch_validate(epoch)
                self('after_epoch')
            self('after_fit')

    def validate(self, *,  dl=None, cbs=None, val_epoch=defaults.val_epoch,):
        """Validate current model on the input dl or val_dl if input dl is None
        - val_epoch (int or None): it will be passed to self._do_epoch_validate's epoch
            - defaults is -1 to indicate the method is not a part of `fit` procedure
        """
        with self.adding_cbs(cbs):
            self('begin_fit')
            self('begin_epoch')
            self._do_epoch_validate(val_epoch, dl=dl)
            self('after_epoch')
            self('after_fit')

    def save(self, fn, with_opt=True):
        fn = self.path/fn
        save_model(fn, self.model, getattr(self, 'opt', None), with_opt)

    def load(self, fn, device):
        fn = self.path/fn
        if self.opt is None: self.create_opt()
        load_model(fn, self.model, self.opt, device)
        return self










































































