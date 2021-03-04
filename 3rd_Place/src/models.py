from sklearn.linear_model import LinearRegression, Lasso
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from dplr.model import SimpleDeepNet, Resnet


# to use a model from a config file,
# the instance must be imported in this file
# and add it to the dictionary with a unique key
library = {'LR': LinearRegression,
           'LGBM': LGBMRegressor,
           'RF': RandomForestRegressor,
           'LASSO': Lasso,
           'simple_deep_net': SimpleDeepNet,
           'resnet': Resnet}
