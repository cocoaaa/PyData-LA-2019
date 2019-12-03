import torch
from torch import optim
from torch import nn
import os, sys, time
import numpy as np
    
from pathlib import Path
import joblib, pdb

class CNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=21):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        
        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=1, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(2, stride=2)
        
    def forward(self, x):
        out1 = self.relu1(self.conv1(x))
        out2 = self.relu2(self.conv2(out1))
        out = self.max_pool(out2)
        return out
    