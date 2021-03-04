from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import json
from typing import Tuple
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

from dataset import Dataset
from model import ConvNet

# Load in serialized model, config, and scaler
with open("config.json", "r") as f:
    CONFIG = json.load(f)

time_steps = CONFIG["timesteps"]

with open("scaler.pck", "rb") as f:
    scaler = pickle.load(f)

ensemble = []
for s in [2, 3, 5, 7, 11, 13, 17]:
    for pwr in [1.5, 2.4, 2.5]:
        model = ConvNet(CONFIG['n_feat'], CONFIG['n_filters'], 2, CONFIG['dropout'])
        model.load_state_dict(torch.load('conv_{:d}_{:.1f}'.format(s, pwr)))
        model.eval()
        ensemble.append(model)

# THIS MUST BE DEFINED FOR YOUR SUBMISSION TO RUN
def predict_dst(
    solar_wind_7d: pd.DataFrame,
    satellite_positions_7d: pd.DataFrame,
    latest_sunspot_number: float,
) -> Tuple[float, float]:
    """
    Take all of the data up until time t-1, and then make predictions for
    times t and t+1.
    Parameters
    ----------
    solar_wind_7d: pd.DataFrame
        The last 7 days of satellite data up until (t - 1) minutes [exclusive of t]
    satellite_positions_7d: pd.DataFrame
        The last 7 days of satellite position data up until the present time [inclusive of t]
    latest_sunspot_number: float
        The latest monthly sunspot number (SSN) to be available
    Returns
    -------
    predictions : Tuple[float, float]
        A tuple of two predictions, for (t and t + 1 hour) respectively; these should
        be between -2,000 and 500.
    """
    # Re-format data to fit into our pipeline
    sunspots = pd.DataFrame(index=solar_wind_7d.iloc[::60].index, columns=["smoothed_ssn"])
    sunspots["smoothed_ssn"] = latest_sunspot_number
    
    ds = Dataset(sunspots, solar_wind_7d, satellite_positions_7d[['gse_x_ace', 'gse_y_ace', 'gse_z_ace']])
    ds.prepare()
    assert ds.data.isna().sum().sum() == 0, ds.data.isna().sum().values.tolist()
    
    features = ds.data.iloc[-time_steps:].values
    features = scaler.transform(features)
    features = features.transpose(1,0)
    features = torch.FloatTensor(features).unsqueeze(0)

    # Make a prediction
    prediction_at_t0, prediction_at_t1 = 0.0, 0.0
    idx = 0
    for s in [2, 3, 5, 7, 11, 13, 17]:
        for pwr in [1.5, 2.4, 2.5]:
            pred = ensemble[idx](features)[0].detach().numpy()
            prediction_at_t0 += pred[0]
            prediction_at_t1 += pred[1]
            idx += 1
    
    prediction_at_t0 /= len(ensemble)
    prediction_at_t1 /= len(ensemble)
    
    # Optional check for unexpected values
    if not np.isfinite(prediction_at_t0):
        prediction_at_t0 = -12
    if not np.isfinite(prediction_at_t1):
        prediction_at_t1 = -12

    return prediction_at_t0, prediction_at_t1
