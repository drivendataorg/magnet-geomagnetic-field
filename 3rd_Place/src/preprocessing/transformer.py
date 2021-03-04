from sklearn.base import BaseEstimator, TransformerMixin
from typing import List
from pandas.api.types import is_numeric_dtype
import pandas as pd
import numpy as np
import re
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def get_numeric_features(data: pd.DataFrame) -> List[str]:
    """
    A function to get all numeric features in a pandas dataframe
    # Parameters
    data: `pd.DataFrame`:
        A pandas Dataframe

    # Returns
    List[str]:
        All numeric columns in the input dataframe
    """
    return [feature
            for feature, values in data.items()
            if is_numeric_dtype(values)]


def filter_by_pattern(features: List[str], patterns: List[str]) -> List[str]:
    """
    A function to filter the values of a list using regex
    # Parameters
    features: `List[str]`
        A list of string
    patterns: `List[str]`
        A list of regex pattern

    # Returns
    List[str]:
        All strings from the input list that match
        at least one of the regex patterns

    """
    return [feature
            for feature in features
            if any(re.search(pattern, feature) is not None
                   for pattern in patterns)]


class NoOp(BaseEstimator, TransformerMixin):
    """
    A base sklearn transformer class.
    this class will not change the input
    """
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X


class FeatureFilter(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer class for filtering
    the columns of pandas dataframe
    """
    def fit(self, X: pd.DataFrame, y=None):
        # filter the features of the input dataframe
        self.features = filter_by_pattern(X.columns, self.patterns)
        return self

    def transform(self, X: pd.DataFrame):
        return X[:, self.features]


class Normalize(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer for normalizing pandas DataFrames

    # Parameters
    drop_features: `List[str]`
        A list of columns of the input dataframe that will not be normalized
    """
    def __init__(self, drop_features: List[str] = ['timedelta', 't0',
                                                   't1', 'period']):
        self.drop_features = drop_features
        # init the scaler transformer
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y=None):
        self.features = [f for f in X.columns
                         if f not in self.drop_features]

        self.scaler = self.scaler.fit(X.loc[:, self.features])
        return self

    def transform(self, X: pd.DataFrame):
        X.loc[:, self.features] = self.scaler.transform(X.loc[:, self.features])
        return X


class ToDtype(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer to change the dtype of all numeric columns to float32
    """
    def fit(self, X: pd.DataFrame, y=None):
        self.features = get_numeric_features(X)

    def transform(self, X):
        X.loc[:, self.features] = X.loc[:, self.features].astype(np.float32)
        return X


class FillNaN(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer to fill nan values
    with the median value of each column in a pandas DataFrame
    """
    def fit(self, X: pd.DataFrame, y=None):
        # get all numeric features
        numeric_features = get_numeric_features(X)
        # for each column
        # calculate the median value if there is any nan value
        self.fill_values = {feature: values.median()
                            for feature, values in X[numeric_features].items()
                            if values.isna().any()}
        return self

    def transform(self, X: pd.DataFrame):
        # fill the nan values
        return X.fillna(self.fill_values)


class MakeSureFeatures(BaseEstimator, TransformerMixin):
    """
    A sklearn transformer to make sure all features in the training set
    are in the test set. this is usefull for the testing phase because
    if there is no available data to compute any given feature, this one may
    not be compute at all.
    This transformer will fill the missing columns with NaN values.
    """
    def fit(self, X: pd.DataFrame, y=None):
        # save all features in the input dataframe
        self.features = X.columns
        return self

    def transform(self, X):
        # add missing column with NaN values
        return X.assign(**{f: np.nan
                           for f in self.features
                           if f not in X})
