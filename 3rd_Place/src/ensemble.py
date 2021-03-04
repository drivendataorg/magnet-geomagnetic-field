import pandas as pd
import numpy as np
from pathlib import Path
import load_data
import default
import logging
import os
from typing import List
from metrics import compute_metrics

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def ensemble(data: pd.DataFrame,
             prediction: List[str] = default.yhat):
    """
    A function to ensemble preds by grouping by ['period', 'timedelta']
    # Parameters
    data: `pd.DataFrame`
        A dataframe containing the predictions
    prediction: `List[str]`, optional (defulat=['yhat_t0', 'yhat_t1'])
        the name of the prediction columns
    """
    ensemble_pred = data.groupby(['period', 'timedelta'])
    ensemble_pred = ensemble_pred[prediction].mean()
    ensemble_pred.reset_index(inplace=True)
    return ensemble_pred


def main():
    """Ensemble all models inside the experiments folder"""
    # we assume all the experiments are saved
    # in the experiments folder
    path = Path('experiments')
    # get a list of all experiments name
    experiment_list = os.listdir(path)
    assert len(experiment_list) > 1, \
           'there is not enough experiments to ensemble'
    predictions = []
    # for every experiment
    for experiment in experiment_list:
        # create a path to the valid prediction file
        path_to_pred = path.joinpath(experiment,
                                     'prediction', 'valid.csv')
        if not os.path.exists(path_to_pred):
            continue
        # if this file exists, we read it and
        # set the experiment column to the name of this experiment
        pred_exp = load_data.read_csv(path_to_pred)
        pred_exp = pred_exp.assign(experiment=experiment)
        predictions.append(pred_exp)
    # concat all the predictions
    predictions = pd.concat(predictions)
    # create the target by dropping all duplicates
    target = predictions.drop_duplicates(subset=['period',
                                         'timedelta'])
    target.reset_index(drop=True, inplace=True)
    target.drop(columns=default.yhat, inplace=True)

    # ensemble
    predictions_ensemble = ensemble(predictions)

    target_ensemble = target.merge(predictions_ensemble,
                                   on=['period', 'timedelta'],
                                   how='left')
    # check there is non nan values
    assert target_ensemble[default.yhat].isna().sum().sum() == 0
    # compute the metrics
    ensemble_metrics = compute_metrics(target_ensemble)
    experiment_list = list(predictions['experiment'].unique())
    ensemble_metrics['experiment'] = '__'.join(experiment_list)
    ensemble_metrics['n_model'] = len(experiment_list)
    results = pd.DataFrame([ensemble_metrics])
    # print scores
    print(results.head())
    # save the ensemble results in a CSV file
    results.to_csv(path / 'ensemble_summary.csv', index=False)


if __name__ == '__main__':
    main()
