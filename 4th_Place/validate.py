from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
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

from datetime import datetime
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from itertools import combinations 

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
        
    def _kfold_indices(self, strategy, n_splits=5, gap=1500):
        groups = self.data.groupby('period')
        indices = []
        lens = []
        for group in ['train_a', 'train_b', 'train_c']:
            grp = groups.get_group(group)
            indices.append( np.array(list( range(grp.shape[0])) ) )
            lens.append(grp.shape[0])
        folds = {s: (np.empty((0,), dtype=int), np.empty((0,), dtype=int)) for s in range(n_splits+1)}
        for i, idx in enumerate(indices):
            if strategy == 0:
                raise NotImplementedError
            else:
                kf = TimeSeriesSplit(n_splits=n_splits+1)
                for s, (train, val) in enumerate(kf.split(idx)):
                    # growing traning 10%, 25%, ... 80%
                    folds[s] = np.hstack([folds[s][0], idx[train][:-gap] + sum(lens[:i])]), np.hstack([folds[s][1], idx[val] + sum(lens[:i])])
        for s in range(len(folds)-1):
            folds[s] = folds[s][0], folds[s][1], folds[s+1][1]
        del folds[s+1]
        assert len(folds) == n_splits
        return folds.values()
    
    def kfold(self, strategy, n_splits=5, gap=1500):
        for fold in self._kfold_indices(strategy, n_splits, gap):
            train_x, train_y = self.data[self.XCOLS].iloc[fold[0]], self.data[self.YCOLS].iloc[fold[0]]
            val_x, val_y = self.data[self.XCOLS].iloc[fold[1]], self.data[self.YCOLS].iloc[fold[1]]
            test_x, test_y = self.data[self.XCOLS].iloc[fold[2]], self.data[self.YCOLS].iloc[fold[2]]
            yield train_x, train_y, val_x, val_y, test_x, test_y

def train_model_snapshot(model, criterion, metric, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle, sch=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0
    model_w_arr = []
    for cycle in range(num_cycles):
        #initialize optimizer and scheduler each cycle
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-2)
        if sch:
            scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs_per_cycle*len(dataloaders['train']))
        for epoch in range(num_epochs_per_cycle):
            print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in dataloaders:
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
                            if sch:
                                scheduler.step()
                    # statistics
                    running_loss += metric_loss.item() * inputs.size(0)
                epoch_loss = np.sqrt(running_loss / dataset_sizes[phase])
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
            #if sch:
                #print('lr', scheduler.get_last_lr()[0])
            print()
        # deep copy snapshot
        model_w_arr.append(copy.deepcopy(model.state_dict()))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    return model_w_arr

def validate(model_w_arr, dataloader, device, metric):
    ensemble_loss = 0.0
    num_cycles = len(model_w_arr)
    val_pred = []
    #predict on validation using snapshots
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        #inputs_sun = inputs_sun.to(device)
        targets = targets.to(device)
        # forward
        # track history if only in train
        outputs = torch.zeros((inputs.shape[0], 2), dtype = torch.float32).to(device)
        for weights in model_w_arr:
            model.load_state_dict(weights)
            model.eval()
            outputs += model(inputs)#, inputs_sun)
        outputs /= num_cycles
        val_pred.append(outputs.detach().cpu().numpy())
        loss = metric(outputs, targets)
        ensemble_loss += loss.item() * inputs.size(0)
    ensemble_loss /= dataset_sizes['val']
    ensemble_loss = np.sqrt(ensemble_loss)
    val_pred = np.concatenate(val_pred, axis = 0)
    return ensemble_loss, val_pred

class MagDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, time_steps):
        self.X = X
        self.Y = Y
        self.time_steps = time_steps
        
    def __len__(self):
        return self.X.shape[0] - self.time_steps
    
    def __getitem__(self, idx):
        x = self.X[idx: idx + self.time_steps]
        x = torch.FloatTensor(x.transpose(1,0))
        y = torch.FloatTensor(self.Y.iloc[idx + self.time_steps - 1].values.flatten())
        return x, y

def make_criterion(p):
    def Criterion(inp, targ):
        return (F.l1_loss(inp, targ, reduction='none') + (torch.log2((inp - targ)**2 + 1)**p)).mean()
    return Criterion

class Conv(nn.Module):
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.pad = pad
        self.layer1 = nn.Sequential(nn.Conv1d(ni, no, kernel, 1, pad), nn.LeakyReLU())
        self.layer2 = nn.Sequential(nn.Conv1d(no, no, kernel, 1, pad), nn.LeakyReLU())
        
    def forward(self, x): 
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        return x1+x2

class FC(nn.Module):
    def __init__(self, ni, no):
        super().__init__()
        
        layers = [nn.Linear(ni, no), nn.LeakyReLU()]
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)

class ConvNet(nn.Module):
    def __init__(self, in_ch, kernel, n_outputs, p=0.1):
        super(ConvNet, self).__init__()
        pad = kernel//2
        self.conv1 = Conv(in_ch, 64, kernel, 2, pad)
        self.conv2 = Conv(64, 128, kernel, 2, pad)
        self.conv3 = Conv(128, 256, kernel, 2, pad)
        self.conv4 = Conv(256, 384, kernel, 2, pad)
        self.max_pool = nn.MaxPool1d(2, 2) #nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(p=p)
        self.fc1 = FC(384 + in_ch, 64)
        self.linear = nn.Linear(64, n_outputs)
        
    def forward(self, x):
        x_last = x[:,:,-1]
        #x = x[:,:,:-1]
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv3(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        #print(x.shape)
        x = self.conv4(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        #
        x = x.mean(-1) 
        x = torch.cat([x, x_last], dim = 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.linear(x)
        x[:,1] += x[:,0]
        return x

def set_seed(s = 0):
    torch.manual_seed(s)
    np.random.seed(s)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set all seeds
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

time_steps = 96
t2_steps = 12
pwr = 2.2
SEEDS = [2, 3, 5, 7, 11, 13, 17]

parser = argparse.ArgumentParser(description='')
parser.add_argument('--data_path', help='Data path', required=True, type=str)
args = parser.parse_args()

DATA_PATH = Path(args.data_path)

dst = pd.read_csv(DATA_PATH / "dst_labels.csv")
dst.timedelta = pd.to_timedelta(dst.timedelta)
dst.set_index(["period", "timedelta"], inplace=True)

sunspots = pd.read_csv(DATA_PATH / "sunspots.csv")
sunspots.timedelta = pd.to_timedelta(sunspots.timedelta)
sunspots.set_index(["period", "timedelta"], inplace=True)

satellites = pd.read_csv(DATA_PATH / "satellite_positions.csv")
satellites.timedelta = pd.to_timedelta(satellites.timedelta)
satellites.set_index(["period", "timedelta"], inplace=True)

solar_wind = pd.read_csv(DATA_PATH / "solar_wind.csv")
solar_wind.timedelta = pd.to_timedelta(solar_wind.timedelta)
solar_wind.set_index(["period", "timedelta"], inplace=True)

_t = datetime.now()
ds = Dataset(dst, sunspots, solar_wind, satellites)
ds.prepare()
print(datetime.now() - _t)

pwr_ensemble = {}
for pwr in [1.5,2.4,2.5]:
    ensemble = {}
    ensemble_sc = {}
    for seed in SEEDS:
        print(' * ' * 20)
        print('seed', seed)
        print(' * ' * 20)
        set_seed(seed)
        #
        train_scores, val_scores, test_scores = [], [], []
        val_scores2, test_scores2 = [], []
        train_preds, val_preds, test_preds = [], [], []
        val_preds2, test_preds2 = [], []
        scalers = []
        regs = []
        for fold, (train_x, train_y, val_x, val_y, test_x, test_y) in enumerate(ds.kfold(1, 5, time_steps)):
            scaler = StandardScaler()
            train_x = scaler.fit_transform(train_x.values)
            val_x = scaler.transform(val_x.values)
            test_x = scaler.transform(test_x.values)
            scalers.append(scaler)
            #
            datasets = {
                'train': MagDataset(train_x, train_y, time_steps),
                'val': MagDataset(val_x, val_y, time_steps),
                'test': MagDataset(test_x, test_y, time_steps)
            }
            dataloaders = {
                'train': torch.utils.data.DataLoader(datasets['train'], batch_size=256, shuffle=True, num_workers=8),
                'val': torch.utils.data.DataLoader(datasets['val'], batch_size=128, shuffle=False, num_workers=4),
                'test': torch.utils.data.DataLoader(datasets['test'], batch_size=128, shuffle=False, num_workers=4)
            }
            dataset_sizes = {x: len(datasets[x]) for x in datasets}
            #
            model = ConvNet(train_x.shape[1], 7, 2)
            model = model.to(device)
            criterion = make_criterion(pwr) #nn.MSELoss()
            metric = nn.MSELoss()
            #train a model on this data split using snapshot ensemble
            model_w_arr = train_model_snapshot(
                model, criterion, metric, 0.001, dataloaders, dataset_sizes, device, 
                num_cycles=1, num_epochs_per_cycle=6, sch=True
            )
            #
            train_sc, train_pred = validate(model_w_arr, dataloaders['train'], device, metric)
            train_scores.append(train_sc)
            train_preds.append(train_pred)
            #
            val_sc, val_pred = validate(model_w_arr, dataloaders['val'], device, metric)
            val_scores.append(val_sc)
            val_preds.append(val_pred)
            #
            test_sc, test_pred = validate(model_w_arr, dataloaders['test'], device, metric)
            test_scores.append(test_sc)
            test_preds.append(test_pred)
        print('train', np.mean(train_scores), np.std(train_scores))
        print('val', np.mean(val_scores), np.std(val_scores))
        print('test', np.mean(test_scores), np.std(test_scores))
        ensemble[seed] = (val_preds, test_preds)
        ensemble_sc[seed] = (train_scores, val_scores, test_scores)
    pwr_ensemble[pwr] = ensemble

comb = (1.5,2.4,2.5)
p1 = np.array([np.array(t[0]) for t in pwr_ensemble[comb[0]].values()]).mean(0)
p2 = np.array([np.array(t[0]) for t in pwr_ensemble[comb[1]].values()]).mean(0)
p3 = np.array([np.array(t[0]) for t in pwr_ensemble[comb[2]].values()]).mean(0)
p = (p1+p2+p3)/3
sc = 0
for fold, (train_x, train_y, val_x, val_y, test_x, test_y) in enumerate(ds.kfold(1, 5, time_steps)):
    sc += mean_squared_error(val_y[time_steps:], p[fold], squared=False)
sc /= 5
print('validation rmse:', sc)

test_p1 = np.array([np.array(t[1]) for t in pwr_ensemble[comb[0]].values()]).mean(0)
test_p2 = np.array([np.array(t[1]) for t in pwr_ensemble[comb[1]].values()]).mean(0)
test_p3 = np.array([np.array(t[1]) for t in pwr_ensemble[comb[2]].values()]).mean(0)
test_p = (test_p1+test_p2+test_p3)/3
test_sc = 0
for fold, (train_x, train_y, val_x, val_y, test_x, test_y) in enumerate(ds.kfold(1, 5, time_steps)):
    test_sc += mean_squared_error(test_y[time_steps:], test_p[fold], squared=False)
test_sc /= 5
print('test rmse:', test_sc)
