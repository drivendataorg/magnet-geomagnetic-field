from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import argparse

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, TimeSeriesSplit

import torch
import torch.nn as nn
import torch.optim as optim
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

def Last(x):
    return x.iloc[-1]
def First(x):
    return x.iloc[0]
def Grad(x):
    return (x.iloc[-1] - x.iloc[0])/60

class Dataset:
    def __init__(self, dst, sunspots, solar_wind, satellites=None):
        self.dst = dst
        self.sunspots = sunspots
        self.solar_wind = solar_wind
        if satellites is not None:
            self.satellites = satellites[['gse_x_ace', 'gse_y_ace', 'gse_z_ace']]
        else:
            self.satellites = None
        
        self.YCOLS = ["t0", "t1"]

        self.SOLAR_WIND_FEATURES = [
            "bt", "temperature", "speed", "density", 
            "bx_gse", "by_gse", "bz_gse", "bx_gsm", "by_gsm", "bz_gsm",
            "theta_gse", "phi_gse",  "theta_gsm", "phi_gsm"
        ]
        
    def impute_features(self, feature_df):
        # forward fill sunspot data for the rest of the month
        feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
        if self.satellites is not None:
            for col in self.satellites.columns:
                feature_df[col] = feature_df[col].fillna(method="ffill")
        # interpolate between missing solar wind values
        feature_df = feature_df.interpolate()
        return feature_df
    
    def aggregate_hourly(self, aggs=["mean", "std", "median", "min", "max"]):
        # group by the floor of each hour use timedelta index
        agged = self.solar_wind.groupby(["period", self.solar_wind.index.get_level_values(1).floor("H")]).agg(aggs)
        # flatten hierachical column index
        agged.columns = ["_".join(x) for x in agged.columns]
        #
        first = self.solar_wind.iloc[::60].copy()
        first.reset_index(inplace=True)
        last = self.solar_wind.iloc[59::60].copy()
        last.reset_index(inplace=True)
        last[['period', 'timedelta']] = first[['period', 'timedelta']]
        first.set_index(['period', 'timedelta'], inplace=True)
        last.set_index(['period', 'timedelta'], inplace=True)
        grad = (last - first) / 60
        first.columns = [c + '_First' for c in first.columns]
        last.columns = [c + '_Last' for c in last.columns]
        grad.columns = [c + '_Grad' for c in grad.columns]
        #
        agged[first.columns] = first
        agged[last.columns] = last
        agged[grad.columns] = grad
        #
        return agged
    
    def preprocess_features(self, subset=None, scaler=None):
        # select features we want to use
        if subset:
            self.solar_wind = self.solar_wind[subset]
        # aggregate solar wind data and join with sunspots
        hourly_features = self.aggregate_hourly().join(self.sunspots)
        if self.satellites is not None:
            hourly_features = hourly_features.join(self.satellites)
        # impute missing values
        self.data = self.impute_features(hourly_features)
    
    def process_labels(self):
        y = self.dst.copy()
        y.dst = y.groupby("period").dst.shift(-1).fillna(y.dst.mean())
        y["t1"] = y.groupby("period").dst.shift(-2).fillna(y.dst.mean())
        y.columns = self.YCOLS
        return y
    
    def prepare(self):
        self.preprocess_features(self.SOLAR_WIND_FEATURES)
        self.XCOLS = self.data.columns.tolist()
        labels = self.process_labels()
        self.data = labels.join(self.data)
        self.XCOLS = self.data.columns.tolist()[2:]

class MagDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, time_steps):
        self.X = X
        self.Y = Y
        self.time_steps = time_steps
        
    def __len__(self):
        return self.X.shape[0] - self.time_steps
    
    def __getitem__(self, idx):
        x = self.X[idx: idx + self.time_steps]
        x[:,-1] = x[-1:,-1]
        x = torch.FloatTensor(x.transpose(1,0))
        y = torch.FloatTensor(self.Y.iloc[idx + self.time_steps - 1].values.flatten())
        return x, y
        
def train_model_snapshot(model, criterion, metric, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle, printing=True):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0
    model_w_arr = []
    for cycle in range(num_cycles):
        #initialize optimizer and scheduler each cycle
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs_per_cycle*len(dataloaders['train']))
        for epoch in range(num_epochs_per_cycle):
            if printing:
                print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
                print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                running_loss = 0.0
                running_corrects = 0
                # Iterate over data.
                for inputs, targets in dataloaders[phase]:
                    inputs = inputs.to(device)
                    #inputs_sun = inputs_sun.to(device)
                    targets = targets.to(device)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)#, inputs_sun)
                        loss = criterion(outputs, targets)
                        metric_loss = metric(outputs, targets)
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            #loss = loss 
                            loss.backward()
                            optimizer.step()
                            scheduler.step()
                    # statistics
                    running_loss += metric_loss.item() * inputs.size(0)
                epoch_loss = np.sqrt(running_loss / dataset_sizes[phase])
                if printing:
                    print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
            if printing:
                print('lr', scheduler.get_last_lr()[0])
                print()
        # deep copy snapshot
        model_w_arr.append(copy.deepcopy(model.state_dict()))
    time_elapsed = time.time() - since
    if printing:
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model_w_arr

def make_criterion(p):
    def Criterion(inp, targ):
        return (F.l1_loss(inp, targ, reduction='none') + (torch.log2((inp - targ)**2 + 1)**p)).mean()
    return Criterion

def set_seed(s = 0):
    #set all seeds
    torch.manual_seed(s)
    np.random.seed(s)

set_seed()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', help='Data path', required=True, type=str)
args = parser.parse_args()

DATA_PATH = Path(args.data_path)
MODEL_PATH = Path(".")

dst = pd.read_csv(DATA_PATH / "dst_labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)
sunspots = pd.read_csv(DATA_PATH / "sunspots.csv")
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)
satellites = pd.read_csv(DATA_PATH / "satellite_positions.csv", usecols=['period', 'timedelta', 'gse_x_ace', 'gse_y_ace', 'gse_z_ace'])
satellites.timedelta = pd.to_timedelta(satellites.timedelta)
satellites.set_index(["period", "timedelta"], inplace=True)
solar_wind = pd.read_csv(DATA_PATH / "solar_wind.csv")
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

_t = datetime.now()
ds = Dataset(dst, sunspots, solar_wind, satellites=satellites)
ds.prepare()
print(datetime.now() - _t)
print(ds.data.head(2))

time_steps = 96

train_x = ds.data[ds.XCOLS]
train_y = ds.data[ds.YCOLS]
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
datasets = {'train': MagDataset(train_x, train_y, time_steps), 'val': MagDataset(train_x, train_y, time_steps)}
dataloaders = {'train': torch.utils.data.DataLoader(datasets['train'], batch_size=256, shuffle=True, num_workers=8),
               'val': torch.utils.data.DataLoader(datasets['val'], batch_size=256, shuffle=False, num_workers=8)}
dataset_sizes = {x:len(datasets[x]) for x in ['train', 'val']}
metric = nn.MSELoss()
for s in [2, 3, 5, 7, 11, 13, 17]:
    for pwr in [1.5, 2.4, 2.5]:
        set_seed(s)
        model = ConvNet((ds.data.shape[1]-2), 7, 2, 0.1)
        model = model.to(device)
        criterion = make_criterion(pwr)
        model_w_arr = train_model_snapshot(model, criterion, metric, 0.001, dataloaders, dataset_sizes, device, num_cycles=1, num_epochs_per_cycle=6)
        model.load_state_dict(model_w_arr[0])
        model = model.cpu()
        model.eval()
        torch.save(model.state_dict(), MODEL_PATH / 'conv_{:d}_{:.1f}'.format(s, pwr))
    
data_config = {'timesteps': time_steps, 'n_feat': (ds.data.shape[1]-2), 'n_models': 1, 'n_filters': 7, 'dropout': 0.1}
with open(MODEL_PATH / "scaler.pck", "wb") as f:
    pickle.dump(scaler, f) 
with open(MODEL_PATH / "config.json", "w") as f:
    json.dump(data_config, f)
