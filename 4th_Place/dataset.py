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

def Last(x):
    return x.iloc[-1]
def First(x):
    return x.iloc[0]
def Grad(x):
    return (x.iloc[-1] - x.iloc[0])/60

class Dataset:
    def __init__(self, sunspots, solar_wind, satellites):
        self.sunspots = sunspots
        self.solar_wind = solar_wind
        self.satellites = satellites
        self.YCOLS = ["t0", "t1"]
        self.SOLAR_WIND_FEATURES = [
            "bt", "temperature", "speed", "density", 
            "bx_gse", "by_gse", "bz_gse", "bx_gsm", "by_gsm", "bz_gsm",
            "theta_gse", "phi_gse",  "theta_gsm", "phi_gsm"
        ]
        
    def impute_features(self, feature_df):
        # forward fill sunspot data for the rest of the month
        #feature_df.smoothed_ssn = feature_df.smoothed_ssn.fillna(method="ffill")
        if self.satellites is not None:
            for col in self.satellites.columns:
                feature_df[col] = feature_df[col].fillna(method="ffill")
        # interpolate between missing solar wind values
        feature_df = feature_df.interpolate(limit_direction='both', limit=200)
        return feature_df
    
    def aggregate_hourly(self, aggs=["mean", "std", "median", "min", "max"]):
        # group by the floor of each hour use timedelta index
        agged = self.solar_wind.groupby(self.solar_wind.index.floor("H")).agg(aggs)
        # flatten hierachical column index
        agged.columns = ["_".join(x) for x in agged.columns]
        #
        first = self.solar_wind.iloc[::60].copy()
        first.reset_index(inplace=True)
        last = self.solar_wind.iloc[59::60].copy()
        last.reset_index(inplace=True)
        last['timedelta'] = first['timedelta']
        first.set_index('timedelta', inplace=True)
        last.set_index('timedelta', inplace=True)
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
        hourly_features = self.aggregate_hourly()#.join(self.sunspots)
        hourly_features['smoothed_ssn'] = self.sunspots
        if self.satellites is not None:
            hourly_features = hourly_features.join(self.satellites)
        self.data = self.impute_features(hourly_features)
    
    def prepare(self):
        self.preprocess_features(self.SOLAR_WIND_FEATURES)
        self.XCOLS = self.data.columns.tolist()
