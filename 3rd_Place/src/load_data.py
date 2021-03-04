import yaml
import os
from typing import Union, Dict, List, Tuple
import pandas as pd
import numpy as np
path_type = Union[str, os.PathLike]


def read_config_file(path_to_file: path_type):
    """
    A function to read a YAML config file
    # Parameters
    path_to_file: `Union[str, os.PathLike]`
        A path to the YAML config file
    # Returns
    Dict[str, Union[bool, float, int, str]]
    """
    error_message = f'{ path_to_file } yaml file does not exists'
    assert os.path.exists(path_to_file), error_message
    with open(path_to_file, 'r') as file:
        return yaml.safe_load(file)


def read_csv(path_to_csv: path_type, **kargs) -> pd.DataFrame:
    """
    A read_csv function that parse the 'timedelta' column
    of the CSV file automatically
    # Parameters
    path_to_csv: `str`
        A path to the CSV file
    kargs: Any
        Any parameter that the pd.read_csv function may take
    # Returns
    A pandas DataFrame: pd.DataFrame
    """
    return pd.read_csv(path_to_csv, parse_dates=['timedelta'],
                       date_parser=pd.to_timedelta, **kargs)


def read_feather(path_to_feather: path_type, **kargs) -> pd.DataFrame:
    """
    A read_feather function that parse the 'timedelta' column
    of the Feather file automatically
    # Parameters
    path_to_feather: `str`
        A path to the Feather file
    kargs: Any
        Any parameter that the pd.read_feather function may take
    # Returns
    A pandas DataFrame: pd.DataFrame
    """
    data = pd.read_feather(path_to_feather, **kargs)
    data['timedelta'] = pd.to_timedelta(data['timedelta'])
    return data


def split_train_data(data: pd.DataFrame, test_frac: float = 0.2,
                     eval_mode: bool = True) -> Tuple[np.ndarray]:
    """
    A function to split the data into training and validation set.
    Both training and validation set will have data from each
    of the periods availables in the dataset
    # Parameters
    data: `pd.DataFrame`
        The main dataframe to be split
    test_frac: `float`, optional (default=0.2)
        the portion of the main dataframe to be used in the validation set.
        the validation set will be the last {test_frac}% of the data.
    eval_mode: `bool`, optional (default=True)
        if False, the training set will use all data available. Otherwise,
        the training set will have only the first {1-test_frac}% of the data.
    # Returns
    Tuple[np.ndarray, np.ndarray]:
        The training and validation indexes
    """
    # calculate the total size of the training set
    test_size = int(len(data) * test_frac)
    # create an array of indexes
    train_indexes = np.arange(len(data))
    # take the last {test_size}/{unique_period} values of each period
    unique_period = data['period'].nunique()
    test_size //= unique_period
    valid_indexes = data.groupby('period').tail(test_size).index
    # if we are doing inference
    if eval_mode:
        # remove the valid indexes from the train indexes
        train_indexes = train_indexes[~np.isin(train_indexes, valid_indexes)]
    # otherwise, the training set will use all data available
    return train_indexes, valid_indexes


def join_multiple_dict(dict_values: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    A function to join a dictionaries of dictionaries into a single one
    # Parameters
    dict_values: `Dict[str, Dict[str, float]]`
        A dictionary where the keys are strings and the values
        are either floats or another instance of a dictionary
    # Returns
    Dict[str, float]
    # Examples
    >>> main_dict = {'period_1': {'rmse': 0.5, 'mae': 0.25},
                     'period_2': {'rmse': 0.75, 'mae': 0.35}}
    >>> joined_dict = join_multiple_dict(main_dict)
    >>> joined_dict
    {'period_1__rmse': 0.5, 'period_1__mae': 0.25},
     'period_2__rmse': 0.75, 'period_2__mae': 0.35}"""
    # output dictionary
    output = {}
    # for every key and value
    for name, value in dict_values.items():
        # if the value is a dictionary instance
        if isinstance(value, dict):
            # recursively call the function
            value = join_multiple_dict(value)
            # update the key's name
            sub_dict = {f'{name}__{subname}': subvalue
                        for subname, subvalue in value.items()}
            # join to the output dictionary
            output.update(sub_dict)
        # otherwise just add the value under the same key
        # to the output dictionary
        else:
            output[name] = value
    return output


def get_features(data: pd.DataFrame, experiment_path: str,
                 fi_threshold: float = None,
                 ignore_features: List[str] = []) -> List[str]:
    """
    A function to select the features use to train a model.
    if there is a feature importance file available
    we can filter the features so we only use relevant features for training.
    # Parameters
    data: `pd.DataFrame`
        The data with all available predictors features.
    experiment_path: `str`
        the path of the experiment where may or not
        exists a feature importance file
    fi_threshold: `float`, optional (default=None)
        if given, we will use this value for filtering
        the features that has greater importance values than {fi_threshold}.
    ignore_features: List[str], optional (default=[])
        A list of features we dont want to include as predictors in our model.
    # Returns
    List[str]
    """
    # create a path to experiment feature importance
    path_to_fi = experiment_path / 'fi_h0.csv'
    # if this file exists and fi_threshold was given
    if (os.path.exists(path_to_fi) and fi_threshold is not None):
        # read the feature importance file
        fi = pd.read_csv(path_to_fi)
        # get only the features with higher importance values
        # than {fi_threshold}
        features = list(fi['feature'][fi['importance'] > fi_threshold])
    # otherwise
    else:
        # get all the column of the dataframe but the ones
        # in ignore_features list.
        features = sorted([feature for feature in data.columns
                           if feature not in ignore_features])
    return features

