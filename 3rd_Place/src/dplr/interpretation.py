from dplr.data import create_dl
from dplr import predict_dl
import numpy as np
import pandas as pd


def permutation_importance(model, data,
                           features,
                           target,
                           score_func,
                           times: int = 1):

    def _score(data):
        _, dl = create_dl(data, features=features)
        output = predict_dl(model, dl)
        prediction = output['prediction'].numpy()
        error = score_func(data[target], prediction)
        return error

    base_score = _score(data)
    fi = []

    for feature in features:
        permuted_data = data.copy()
        permuted_data[feature] = np.random.permutation(permuted_data[feature])
        feature_score = _score(permuted_data)
        feature_importance = {'feature': feature,
                              'score': feature_score,
                              'importance': feature_score-base_score,
                              }
        fi.append(feature_importance)
    fi = pd.DataFrame(fi)
    fi.sort_values(by='importance', inplace=True, ascending=False)
    fi.reset_index(drop=True, inplace=True)
    return fi
