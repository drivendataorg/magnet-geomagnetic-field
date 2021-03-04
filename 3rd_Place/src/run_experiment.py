from pathlib import Path
import pandas as pd
import load_data
import logging
import click
import joblib
import mlflow
import default
import os
from metrics import compute_metrics, feature_importances
from metrics import compute_metrics_per_period, calculate_error_on_test
from models import library as model_library
from pipelines import build_pipeline

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


@click.command()
@click.argument('experiment_path', type=click.Path(exists=True))
@click.option('--eval_mode', type=click.BOOL, default=True)
@click.option('--use_sample', type=click.BOOL, default=False)
@click.option('--test_frac', type=float, default=0.2)
@click.option('-m', '--message', type=str, default=None)
@click.option('-fthres', '--fi_threshold', type=float, default=None)
def main(experiment_path: str, eval_mode: bool = True,
         use_sample: bool = False, test_frac: float = 0.2,
         message: str = None, fi_threshold: float = None):
    """
    A function to train or validate an Experiment
    # Parameters
    experiment_path: `str`
        A path to the folder's experiment config file.
        the config file must be named config.yml and
        it must contain the following keys:
            model: `str`
                the path to the model config file
            pipeline: `str`
                the path to the pipeline config file

    eval_mode: `bool`, optional (default=True)
        if False, the model will be train using all the data available,
        otherwise, the trained model will be use for inference.
    use_sample: `bool`, optional (default=False)
        if True, we will use only a sample from the dataset.
        to use it, before execute the make_sample.py file
        to create this sample dataset.
    test_frac: `float`, optional (default=0.2)
        if eval_mode is True, the size of the valid dataset
        will be the {test_frac}% of the main dataset.
    message: `str`, optional (default=None)
        we use mlflow to keep track of all parameters and errors of each experiment,
        this parameter will register any string you pass into the experiment record in mlflow.
    fi_threshold `float`, optional (default=None)
        if already exists a feature importance file, this value will be use for filtering
        the features that has greater importance values than {fi_threshold}.
    """
    # getting experiment name
    experiment = os.path.basename(experiment_path)
    logging.info(f'running {experiment}')
    logging.info(f'eval_mode={eval_mode}, use_sample={use_sample}')
    logging.info('reading config file')
    # creating experiment path and loading experiment config file
    experiment_path = Path(experiment_path)
    config = load_data.read_config_file('./config/config.yml')
    experiment_config = load_data.read_config_file(experiment_path / 'config.yml')
    # reading experiment's model and pipeline config file
    pipeline_config = load_data.read_config_file(experiment_config['pipeline'])
    model_config = load_data.read_config_file(experiment_config['model'])

    directories = config['directories']
    # getting the data path
    processed_path = Path(directories['processed'])
    # creating a prediction folder to save prediction after training
    prediction_path = experiment_path / 'prediction'
    prediction_path.mkdir(exist_ok=True, parents=True)
    # creating a model path to save models after training
    model_path = experiment_path / 'models'

    # reading preprocessed data
    filename = ('fe' if not use_sample else 'fe_sample')
    logging.info('reading training data')
    data = load_data.read_feather(processed_path / f'{filename}.feather')

    logging.info('splitting dataset')
    train_idx, valid_idx = load_data.split_train_data(data,
                                                      test_frac=test_frac,
                                                      eval_mode=eval_mode)
    train_data = data.loc[train_idx, :]
    valid_data = data.loc[valid_idx, :]

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    # importing pipeline
    logging.info('building pipeline')
    pipeline = build_pipeline(pipeline_config)
    logging.info(f'{pipeline}')

    # fit pipeline
    logging.info('training pipeline')
    pipeline.fit(train_data)
    # transform both training and valid dataset
    logging.info('transforming datasets')
    train_data = pipeline.transform(train_data)
    valid_data = pipeline.transform(valid_data)

    # loading the features to train our model
    # if exists a feature importance file
    # we can use it to train our model only with revelant features
    features = load_data.get_features(train_data,
                                      experiment_path=experiment_path,
                                      fi_threshold=fi_threshold,
                                      ignore_features=default.ignore_features)
    in_features = len(features)
    logging.info(f'modeling using {len(features)} features')
    logging.info(f'{features[:30]}')

    # importing model instance
    model_instance = model_library[model_config['instance']]
    logging.info('training horizon 0 model')
    # training model for horizon 0
    model_h0 = model_instance(**model_config['parameters'])
    model_h0.fit(train_data.loc[:, features], train_data.loc[:, 't0'])

    logging.info('training horizon 1 model')
    # training model for horizon 1
    model_h1 = model_instance(**model_config['parameters'])
    model_h1.fit(train_data.loc[:, features], train_data.loc[:, 't1'])

    logging.info('prediction h0 and h1 models')
    # predicting
    train_data['yhat_t0'] = model_h0.predict(train_data.loc[:, features])
    train_data['yhat_t1'] = model_h1.predict(train_data.loc[:, features])
    valid_data['yhat_t0'] = model_h0.predict(valid_data.loc[:, features])
    valid_data['yhat_t1'] = model_h1.predict(valid_data.loc[:, features])

    # compute errors
    train_error = compute_metrics(train_data, suffix='_train')
    valid_error = compute_metrics(valid_data, suffix='_valid')

    train_error_period = compute_metrics_per_period(train_data,
                                                    suffix='_train')
    valid_error_period = compute_metrics_per_period(valid_data,
                                                    suffix='_valid')
    logging.info('errors')
    logging.info(f'{train_error}')
    logging.info(f'{valid_error}')
    logging.info('period errors')
    logging.info(f'{train_error_period}')
    logging.info(f'{valid_error_period}')
    if eval_mode:
        with mlflow.start_run(run_name=experiment):
            # saving predictions
            train_prediction = train_data.loc[:, default.keep_columns]
            train_prediction.to_csv(prediction_path / 'train.csv', index=False)
            # saving errors
            train_error_period.to_csv(experiment_path / 'train_erros.csv',
                                      index=False)
            valid_error_period.to_csv(experiment_path / 'valid_erros.csv',
                                      index=False)
            # valid_prediction = valid_data.loc[:, default.keep_columns]
            valid_data.to_csv(prediction_path / 'valid.csv', index=False)
            # saving feature importances if there is aviable
            fi_h0 = feature_importances(model_h0, features)
            fi_h1 = feature_importances(model_h1, features)
            if (fi_h0 is not None) and (fi_h1 is not None) and (fi_threshold is None):
                fi_h0.to_csv(experiment_path / 'fi_h0.csv', index=False)
                fi_h1.to_csv(experiment_path / 'fi_h1.csv', index=False)
            # saving to mlflow
            # saving metrics
            mlflow.log_metrics(train_error)
            mlflow.log_metrics(valid_error)
            mlflow.log_params({'fi_threshold': fi_threshold,
                               'in_features': in_features})
            # saving model parameters
            mlflow.log_params(model_config['parameters'])
            tags = {'use_sample': use_sample,
                    'model_instance': model_config['instance'],
                    'experiment': experiment}
            if message is not None:
                tags['message'] = message
            mlflow.set_tags(tags)
    else:
        # creating model path
        test_error = calculate_error_on_test(train_data)
        test_error = pd.DataFrame([test_error])
        test_error.to_csv(experiment_path / 'check_test_error.csv',
                          index=False)
        model_path.mkdir(exist_ok=True, parents=True)
        joblib.dump(model_h0, model_path / 'model_h0.pkl')
        joblib.dump(model_h1, model_path / 'model_h1.pkl')
        joblib.dump(pipeline, model_path / 'pipeline.pkl')
        joblib.dump(features, model_path / 'features.pkl')


if __name__ == '__main__':
    main()
