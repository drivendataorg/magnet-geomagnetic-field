from pathlib import Path
import load_data
from preprocessing.base import create_target, stl_preprocessing
from preprocessing.base import merge_daily, solar_wind_preprocessing
from preprocessing.base import split_into_period
import time
import logging
import click
import default
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


@click.command()
@click.option('--use_sample', type=click.BOOL, default=False)
@click.option('--n_jobs', type=int, default=1)
def main(use_sample: bool = False,
         n_jobs: int = 1):
    """
    This function will apply all the steps in order to create
    a dataset ready to train models.
    The following steps:
        - read the data
        - compute the solar wind features
        - compute sattelite positions features
        - take the log of smoothed_ssn values
        - create the target for the actual time t and t + 1 hour
        - merge all dataset into a single one
        - save the dataset for future modeling
    # Params
    use_sample: `bool`, optional(defualt=False)
        Whether or not to use the sample dataset
    n_jobs: `in`, optional(defualt=1)
        The number of jobs to run in parallel

    """
    logging.info(f'use_sample={use_sample}, n_jobs={n_jobs}')
    logging.info('reading config file')
    config = load_data.read_config_file('./config/config.yml')
    # directories
    directories = config['directories']
    raw_path = Path(directories['raw'])
    interim_path = Path(directories['interim'])
    processed_path = Path(directories['processed'])
    processed_path.mkdir(exist_ok=True, parents=True)

    # reading gt data
    solar_wind_file = ('sample_solar_wind.feather'
                       if use_sample else 'solar_wind.feather')
    logging.info('reading training data')
    dst_labels = load_data.read_csv(raw_path / 'dst_labels.csv')
    solar_wind = load_data.read_feather(interim_path / solar_wind_file)
    sunspots = load_data.read_csv(raw_path / 'sunspots.csv')
    stl_pos = load_data.read_csv(raw_path / 'satellite_positions.csv')

    logging.info('preprocessing solar wing')
    # preprocessing solar wind
    # setting timedelta as index
    solar_wind.set_index('timedelta', inplace=True)
    # preprocessing solar wind time series
    solar_wind = solar_wind_preprocessing(solar_wind)
    logging.info('computing features')
    start = time.time()
    # computing solar wind features
    data = split_into_period(solar_wind,
                             features=default.init_features,
                             n_jobs=n_jobs)
    elapsed_time = (time.time()-start)/60
    logging.info(f'elapsed time {elapsed_time:.4f}')

    logging.info('merging other datasets')
    # create target
    target = create_target(dst_labels)
    # preprocessing sattelite positions
    stl_pos = stl_preprocessing(stl_pos)
    # taking the log of smoothed_ssn values
    sunspots['smoothed_ssn'] = np.log(sunspots['smoothed_ssn'])
    # merging dataframes to the main dataframe
    data = merge_daily(data, stl_pos)
    data = merge_daily(data, sunspots)
    # merging target dataframe to the main dataframe
    data = data.merge(target, how='left', on=['period', 'timedelta'])
    # droping last values where there is not available data
    data.dropna(subset=['t0', 't1'], inplace=True)
    # reset index
    data.reset_index(inplace=True, drop=True)
    logging.info('saving')
    output_filename = 'fe' if not use_sample else 'fe_sample'
    # saving to feather format
    data.to_feather(processed_path / f'{output_filename}.feather')


if __name__ == '__main__':
    main()
