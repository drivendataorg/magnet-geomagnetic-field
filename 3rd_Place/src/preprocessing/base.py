from typing import List, Dict
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from joblib import Parallel, delayed
from preprocessing.fe import calculate_features


def fillna_features(X: pd.DataFrame,
                    features: List[str] = None,
                    interpolate: bool = True):
    """
    Fill the nan values of a DataFrame
    # Parameters
    X: `pd.DataFrame`
        A pandas DataFrame
    features: `List[str]`, optional (default=None)
        A subset of the columns of the dataframe
    interpolate: `bool`, optional (default=True)
        if True, we will use interpolation to fill the gaps,
        otherwise, we propagate last valid observation forward.
    # Returns
    pd.DataFrame: the input pandas dataframe without nan values
    """
    if features is None:
        features = X.columns
    # fillnan values with the closest non-nan value
    for feature, values in X[features].items():
        if is_numeric_dtype(values) and values.isna().any():
            values = (values.interpolate()
                      if interpolate else
                      values.fillna(method='ffill'))
            values = values.fillna(method='backfill')
            X[feature] = values
    return X


def create_target(dst_values: pd.DataFrame):
    """
    Create the target dataframe for the actual time `t`
    and for the next hour `t+1`
    # Parameters
    dst_values: `pd.DataFrame`
        Dst values pandas dataframe

    #Returns
    pd.DataFrame:
        A pandas dataframe with the t and (t + 1 hour) dst values
    """
    target = dst_values.loc[:, ['period', 'timedelta']].reset_index(drop=True)
    target['t0'] = dst_values['dst'].values
    target['t1'] = target.groupby('period')['t0'].shift(-1).fillna(-12)
    return target


def merge_daily(data: pd.DataFrame,
                other: pd.DataFrame) -> pd.DataFrame:
    """Merge 2 dataframes by period and number of days.
    # Parameters
    data: `pd.DataFrame`
        the main pandas dataframe
    other: `pd.DataFrame`
        the pandas dataframe to be merged.
        We assumed this dataframe has a frequency of daily observations
    # Returns
    pd.DataFrame: A DataFrame of the two merged dataframes.
    """
    # create the day column for both dataframe
    data.loc[:, 'day'] = data['timedelta'].dt.days
    other.loc[:, 'day'] = other['timedelta'].dt.days
    # drop duplicated observations
    other.drop_duplicates(subset=['period', 'day'],
                          inplace=True)
    # merge by period and day
    data = data.merge(other.drop('timedelta', axis=1),
                      on=['period', 'day'],
                      how='left')
    # fill nan values propating last valid value
    data = fillna_features(data,
                           features=other.columns,
                           interpolate=False)
    # drop day column for both dataframe
    data.drop('day', inplace=True, axis=1)
    other.drop('day', inplace=True, axis=1)
    return data


def stl_preprocessing(data: pd.DataFrame):
    """
    satellite positions preprocessing
    # Returns
    pd.DataFrame: the processed satellite positions dataframe"""
    # drop dscovr columns
    to_drop = ['gse_x_dscovr', 'gse_y_dscovr', 'gse_z_dscovr']
    data.drop(to_drop, inplace=True, axis=1)
    working_features = ['gse_x_ace', 'gse_y_ace', 'gse_z_ace']
    period_data = data.groupby('period')
    # calculate the direction (up or down) of
    # each satellite coordinate
    for feature in working_features:
        direction = period_data[feature].diff().fillna(0)
        data.loc[:, f'{feature}_direction'] = np.clip(direction, -1, 1)
    return data


def solar_wind_preprocessing(solar_wind: pd.DataFrame,
                             features: List[str] = None):
    """solar wind pre-preprocessing
    # Parameters
    solar_wind: `pd.DataFrame`
        solar wind dataframe
    features: List[str], optional (default=None)
        A subset of column of the dataframe
    # Returns
        the processed solar wind dataframe
    """
    # take the log the temperature
    solar_wind.loc[:, 'temperature'] = np.log(solar_wind['temperature'] + 1)
    # take the sqrt of the speed
    solar_wind.loc[:, 'speed'] = np.sqrt(solar_wind['speed'])
    # if the features are given, return this features only
    if features is not None:
        solar_wind = solar_wind.loc[:, features]
    return solar_wind


def split_data_in_chunks(data: pd.DataFrame,
                         time_lenght=pd.to_timedelta(7, unit='d')
                         ) -> Dict[str, pd.DataFrame]:
    """
    A function to split the data into chunks of seven days.
    this is used to simulate how the data is given in the test phase
    # Parameters
    data: `pd.DataFrame`
        A solar wind pandas dataframe
    time_lenght: timedelta, optional(defualt=7 days)
        the time window of each chunk
    # Returns
    Dict[str, pd.DataFrame]:
        A dict where each key is a valid timedelta value
        and its value is last 7 days of solar wind data
        before time t
    """
    one_minute = pd.to_timedelta(1, unit='m')
    output = {}
    # for each hour in the dataset
    for timestep in data.index.ceil('H').unique():
        # if there is not enough available data
        # to complete the window, skip it
        if timestep < time_lenght:
            continue
        # create a chunk of the last 7 days of data up until (t - 1) minutes
        output[timestep] = data.loc[timestep-time_lenght: timestep-one_minute, :]
    return output


def from_chunks_to_dataframe(chunks: Dict[str, pd.DataFrame], n_jobs: int = 8):
    """
    Apply calculate_features function to every chunck and turn
    the output features to a pandas dataframe

    # Parameters
    chunks: `Dict[str, pd.DataFrame]`
        A dictionary where each key is a valid timedelta value
        and its value is last 7 days of solar wind data
        before time t

    n_jobs: `int`, optinal(defualt=8)
        The number of jobs to run in parallel

    # Returns
    pd.DataFrame:
        A pandas dataframe with the following shape: (n_chunk, n_features)
        where n_chunk is the number of element in the chunks dictionary
        and n_features is the number of features calculated in the
        calculate_features function
    """
    return pd.DataFrame(Parallel(n_jobs=n_jobs)(delayed(calculate_features)(datastep, timestep)
                        for timestep, datastep in chunks.items()))


def one_chunk_to_dataframe(chunk: pd.DataFrame,
                           timestep=np.nan):
    """
    A function to apply calculate_features function to a single chunk of data.
    this function is used for the testing phase, where we predict only one timestep
    at a time.

    # Parameters
    chuck: `pd.DataFrame`
        7 days worth of solar wind data
    timestep:
        A valid timedelta value

    # Returns
    pd.DataFrame:
        A pandas dataframe with the following shape: (1, n_features)
        where n_features is the number of features generated by the
        calculate_features function
    """
    features = calculate_features(chunk, timestep)
    return pd.DataFrame([features])


def split_into_period(data: pd.DataFrame, features: List[str],
                      n_jobs: int = 8) -> pd.DataFrame:
    """
    This function is used for preprocessing the solar wind data
    for the training phase, where we have multiple periods.
    This function will split the data by period and apply the following steps:
        - create chuncks of 7 seven days using split_data_in_chunks function
        - apply the from_chunks_to_dataframe function
    and finally, we will concatenate the generated features for all periods

    # Params
    data: `pd.DataFrame`
        The solar wind dataset

    features: `List[str]`
        A subset of the column of the input dataframe;
        these are the name of time series we want to process

    n_jobs: `int`, optional(defualt=8)
        The number of jobs to run in parallel

    # Returns
    pd.DataFrame:
        A pandas dataframe with all computed features for all periods
        in the dataset
    """
    output_data = []
    for period, period_data in data.groupby('period'):
        # create chunck using only the chosen time series
        chunks = split_data_in_chunks(period_data.loc[:, features])
        # compute the features
        fe_data = from_chunks_to_dataframe(chunks, n_jobs=n_jobs)
        # add the period column
        fe_data.loc[:, 'period'] = period
        output_data.append(fe_data)
        del chunks
    # concat all periods
    return pd.concat(output_data, ignore_index=True, axis=0)
