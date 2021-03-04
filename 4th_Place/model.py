from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
#from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import copy
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv1d(ni, no, kernel, 1, pad), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(no, no, kernel, 1, pad), nn.LeakyReLU())
        
    def forward(self, x): 
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x1+x2

class ConvNet(nn.Module):
    def __init__(self, in_ch, kernel, n_outputs, p=0.1):
        super(ConvNet, self).__init__()
        pad = kernel//2
        self.conv1 = Conv(in_ch, 64, kernel, 2, pad)
        self.conv2 = Conv(64, 128, kernel, 2, pad)
        self.conv3 = Conv(128, 256, kernel, 2, pad)
        self.conv4 = Conv(256, 384, kernel, 2, pad)
        self.max_pool = nn.MaxPool1d(2, 2)
        self.dropout = nn.Dropout(p=p)
        self.fc1 = nn.Sequential(nn.Linear(384 + in_ch, 64), nn.LeakyReLU())
        self.linear = nn.Linear(64, n_outputs)

    def init_hidden(self, n):
        return torch.zeros((2*2,n,self.hidden_size), dtype = torch.float32).cuda()

    def forward(self, x):
        x_last = x[:,:,-1]
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = self.conv4(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        x = x.mean(-1)
        x = torch.cat([x, x_last], dim = 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.linear(x)
        x[:,1] += x[:,0]
        return x
