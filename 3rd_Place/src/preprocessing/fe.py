
from itertools import groupby
import numpy as np
from scipy.stats import linregress
import pandas as pd
from typing import List, Dict
from load_data import join_multiple_dict
from collections import defaultdict


def consecutive_count(sequence: np.ndarray) -> List[int]:
    """
    This method calculates the length of all sub-sequences
    where the array sequence is either True or 1.

    # Examples
    --------
    >>> x = [0,1,0,0,1,1,1,0,0,1,0,1,1]
    >>> consecutive_count(x)
    >>> [1, 3, 1, 2]

    >>> x = [0,True,0,0,True,True,True,0,0,True,0,True,True]
    >>> consecutive_count(x)
    >>> [1, 3, 1, 2]

    # Params
    sequence: np.ndarray
        An iterable containing only 1, True, 0 and False values
    # Returns
    List[int]: A list with the length of all sub-sequences.
    """
    output = [sum(seq) for value, seq in groupby(sequence) if value == 1]
    if len(output) == 0:
        output.append(0)
    return output


def calculate_linreg_features(values: np.array,
                              attrs: List[str] = ['slope', 'intercept']):
    """
    Calculates a linear least-squares regression for values of the time series
    and returns the slope and intercept values of the fitted linear model
    # Params
    values: `np.array`
        time series values
    attrs: `List[str]`
        List of linear regression characteristics, possible extracted attributes are
        "pvalue", "rvalue", "intercept", "slope" and "stderr",
        see the documentation of linregress for more information.

    # Returns
    Dict[str, float]:
        A dictionary with keys:
            slope: float
            intercept: float
    """
    linreg = linregress(np.arange(len(values)), values)
    return {attr: getattr(linreg, attr)
            for attr in attrs}


def calculate_dx_features(values: np.ndarray):
    """
    this function will calculate change rate and
    an estimate of the complexity of the time series
    # Params
    values: `np.array`
        time series values

    # Returns
    Dict[str, float]:
        A dictionary with keys:
            cid_ce: float
            change_rate: float
    """
    output = {}
    difference = (values - np.roll(values, 1))[1:]
    non_zero = np.nonzero(values[:-1])
    output['cid_ce'] = np.sqrt(np.nanmean(np.square(difference)))
    output['change_rate'] = np.nanmean(difference[non_zero] / values[:-1][non_zero])
    return output


def compute_fourier_stats(values):
    """
    This function will calculate the fast fourier transform
    of the time series and extract the mean and std for both the
    real and imaginary parts. Also, it computes the mean, std
    and the 10%, 50% and 90% percentiles of the power spectrum.
    # Params
    values: `np.array`
        time series values

    # Returns
    Dict[str, float]:
        A dictionary with keys:
            rfft_real_mean: float
            rfft_real_std: float
            power_spectrum_mean: float
            power_spectrum_q0.1: float
            power_spectrum_q0.5: float
            power_spectrum_q0.9: float
            power_spectrum_std: float
            rfft_imag_mean: float
            rfft_imag_std: float
    """

    # computes the rfft
    rfft = np.fft.rfft(values)
    # power spectrum
    psd = np.real(rfft * np.conj(rfft)) / (len(values)**2)
    # real part
    real_rfft = np.real(rfft) / len(values)
    # imag part
    imag_rfft = np.imag(rfft) / len(values)
    # compute the stats
    stats = {'rfft_real_mean': real_rfft.mean(),
             'rfft_real_std': real_rfft.std(),
             'power_spectrum_mean': psd.mean(),
             'power_spectrum_q0.1': np.quantile(psd, 0.1),
             'power_spectrum_q0.5': np.quantile(psd, 0.5),
             'power_spectrum_q0.9': np.quantile(psd, 0.9),
             'power_spectrum_std': psd.std(),
             'rfft_imag_mean': imag_rfft.mean(),
             'rfft_imag_std': imag_rfft.std()}
    return stats


def time_iter(data, periods):
    for period in periods:
        yield period, data.iloc[-period:]


def point_in_range(values, mean: float, std: float, p: float = 1.96):
    """
    Returns the length of the longest consecutive subsequence in values
    that that lies within p standard deviations away from the mean

    # Params
    values: `np.array`
        time series values
    mean: float
        mean of the time series
    std: float
        std of the time series
    p: float
        std proportion

    # Returns
    Dict[str, float]:
        A dictionary with keys:
            consecutive_in_range: float
    """
    points_in_range = (np.abs(values - mean) <= p*std)
    max_consecutive = max(consecutive_count(points_in_range))
    max_consecutive /= len(points_in_range)
    return {'consecutive_in_range': max_consecutive}


# mean and std periods
mean_std_periods = [1*60, 5*60, 10*60, 48*60]
# lin features periods
lin_periods = [48*60]
# derivate periods
outlier_periods = [48*60]
# fourier periods
fourier_periods = [72*60]


def _calculate_features(values: pd.Series,
                        compute_abs_lin=False) -> Dict[str, float]:
    """
    Computes all features for a single time series
    # Params
    values: `pd.Series`
        time series values
    compute_abs_lin: `bool`
        Wheter or not to calculate the linear characteristics
        of the absolute time series
    # Returns
    Dict[str, float]
        A dictionary with all computed features.
    """
    feature_dict = defaultdict(dict)
    # fill nan values propating forward the last valid output
    # also propate backwards in cause the time series starts with nan values
    imputed_values = values.fillna(method='ffill').fillna(method='backfill')
    non_nan_values = values.dropna()
    # if there is not data, return empty dict
    if len(non_nan_values) == 0:
        return feature_dict

    # calculate mean and std features
    for hours, last_n_values in time_iter(values, mean_std_periods):
        mean_std = last_n_values.agg(('mean', 'std')).to_dict()
        feature_dict[f'{hours//60}h'].update(mean_std)

    # calculate linear features
    for hours, last_n_values in time_iter(values, lin_periods):
        # drop nan values
        last_n_values = last_n_values.dropna()
        # if all values nan, continue with the next period
        if len(last_n_values) < 1:
            continue

        linear_properties = calculate_linreg_features(last_n_values)
        feature_dict[f'{hours//60}h'].update(linear_properties)
        if compute_abs_lin:
            # take absolute value of the time series
            abs_last_n_values = last_n_values.abs()
            # calculate the abs linear features
            abs_linear_properties = calculate_linreg_features(abs_last_n_values)
            feature_dict[f'{hours//60}h']['abs'] = abs_linear_properties

    # calculate derivates and outlier features
    for hours, last_n_values in time_iter(values, outlier_periods):
        # pd.series to np.ndarray
        last_n_values = last_n_values.to_numpy()
        dx_features = calculate_dx_features(last_n_values)
        pin_range = point_in_range(last_n_values,
                                   mean=feature_dict['48h']['mean'],
                                   std=feature_dict['48h']['std'])
        feature_dict[f'{hours//60}h'].update(dx_features)
        feature_dict[f'{hours//60}h'].update(pin_range)
    # calculate fourier features
    for hours, last_n_values in time_iter(imputed_values, fourier_periods):
        # pd.series to np.ndarray
        last_n_values = last_n_values.to_numpy()
        fourier_features = compute_fourier_stats(last_n_values)
        feature_dict[f'{hours//60}h'].update(fourier_features)
    return feature_dict


def calculate_features(data: pd.DataFrame,
                       timestep=np.nan) -> Dict[str, float]:
    """
    Computes all features for all time series in the dataframe
    # Params
    values: `pd.DataFrame`
        A pandas dataframe, we assume that every column is a time series
    timestep: `timedelta`
        A valid timedelta value
    # Returns
    Dict[str, float]
        A dictionary with all computed features for all timeseries.
    """
    features = defaultdict(dict)
    features['timedelta'] = timestep
    for feature, values in data.items():
        # if the time series is a coordinate
        # compute the absolute linear features
        is_coor = feature.startswith(('bx_', 'by_', 'bz_'))
        # compute the timeseries features
        features[feature] = _calculate_features(values, is_coor)
    return join_multiple_dict(features)
