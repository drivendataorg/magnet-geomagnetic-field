import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F


def to_numpy(function):
    """
    A decorator for transforming the input
    of a function from pandas dataframes to numpy arrays
    """
    def _inner(*arrays):
        new_arrays = [array.to_numpy()
                      if isinstance(array, pd.DataFrame)
                      else array
                      for array in arrays]
        return function(*new_arrays)
    return _inner


@to_numpy
def rmse(target, yhat):
    return np.sqrt(np.square(target - yhat).mean())


def compute_metrics(data: pd.DataFrame,
                    target='t',
                    yhat='yhat',
                    suffix=''):
    return {f'h0_rmse{suffix}': rmse(data[f'{target}0'], data[f'{yhat}_t0']),
            f'h1_rmse{suffix}': rmse(data[f'{target}1'], data[f'{yhat}_t1']),
            f'rmse{suffix}': rmse(data[[f'{target}0', f'{target}1']],
                                  data[[f'{yhat}_t0', f'{yhat}_t1']])
              }


def get_raw_importances(model):
    methods = ['feature_importances_', 'coef_']
    for method in methods:
        importances = getattr(model, method, None)
        if importances is not None:
            return importances
    return None


def feature_importances(model, features):
    importances = get_raw_importances(model)
    if importances is None:
        return None
    importances = np.abs(importances)
    fi = pd.DataFrame({'feature': features,
                       'importance': importances})
    fi.sort_values(by='importance', ascending=False, inplace=True)
    fi.reset_index(drop=True, inplace=True)
    return fi


def compute_metrics_per_period(data, target='t',
                               yhat='yhat', suffix=''):
    errors = []
    for period, data_period in data.groupby('period'):
        period_errors = compute_metrics(data_period, target=target,
                                        yhat=yhat, suffix=suffix)
        period_errors['period'] = period
        errors.append(period_errors)
    return pd.DataFrame(errors)


def torch_rmse(yhat, y):
    return torch.sqrt(F.mse_loss(yhat, y))


def calculate_error_on_test(train_data):
    t7_days = pd.to_timedelta(7, unit='d')
    correct_period = train_data[train_data['period'] == 'train_a']
    first_2500 = correct_period[correct_period['timedelta'] >= t7_days]
    return compute_metrics(first_2500.iloc[:2501])
