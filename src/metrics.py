# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from copy import deepcopy
from typing import Callable, List, Iterable, Union
from utils import class2attr, get_cls_name

import pdb
import abc
from abc import ABC, abstractmethod

import torch
from torch import Tensor

class runningScore():
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        
    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)
              
    def get_scores(self, ignore_gt=None):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = deepcopy(self.confusion_matrix)
        if ignore_gt is not None:
            hist[ignore_gt,:] = 0

        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
#         print('ignore_gt: ', ignore_gt, ' , acc cls: ', acc_cls)##debug
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter():
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Metric(ABC):
    """Abastrct class for defining a metric"""
    @property
    def name(self): return class2attr(self, 'Metric')

    @property
    @abstractmethod
    def value(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def accumulate(self, learner):
        pass

class AvgMetric(Metric):
    """Average the values of the 'func' -- taking into account for different
    batch sizes"""
    def __init__(self,
                 func: Callable[[Tensor,Tensor],Union[Iterable, float, int]]):
        """
        - self.func (Callable[Tensor, Tensor]): Callable that takes in the output of model stored as learner.pred,
        and target batch_y (stored as learner.yb) and computes the metric (either a scalar or a Tensor)
            - eg: lambda pred, yb: (pred-yb)**2.mean()
                which is equivalent to nn.MSELoss(reduction='mean')
            - It *must* compute an Avg metric. In particular, if it has 'reduction' attribute, it must be 'mean'
        """
        self.func = func
        if getattr(func, 'reduction', None):
            assert func.reduction == 'mean'
        self.total, self.count = 0., 0

    def reset(self):
        self.total, self.count = 0., 0

    def accumulate(self, learner):
        bs = len(learner.xb)
        self.total += self.func(learner.pred, learner.yb).detach() * bs
        self.count += bs

    @property
    def value(self):
        return self.total/self.count if self.count != 0 else None

    @property
    def name(self):
        return get_cls_name(self)

def compute_binary_acc(pred, target):
    """
    Computes accuracy of a binary classification at each location of the pred tensor
    using target as the ground truth labels

    Assumes each element in pred Tensor, is a float (neg or pos)
    and each element in the target is a binary label of 0 or 1
    in the context of binary classification at each element value.

    So, relevant loss function would be nn.BCEWithLogitLoss(pred,target) which is
    a (stable, combined implementation of Sigmoid and BCELoss using logsumexp trick.

    Since it is the Sigmoid function that is applied elementwise to the `pred` Tensor,
    if a value in the `pred` is >0, that means the element will make a prediction of 1

    We use this fact and take a shortcut (which also avoids possibly unstalbe exp computation)
    to compute the accuracy averaged by the number of elements in the pred Tensor
    """
    assert pred.shape == target.shape  # valid for binary classification
    _pred = pred.detach()
    correct1 = (_pred > 0) & (target == 1)
    acc = correct1.sum() / (target==1).sum()
    return acc.item()