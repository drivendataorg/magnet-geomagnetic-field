import pandas as pd
from pathlib import Path
import load_data
import logging
import click
from default import default_sample_frac


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


@click.command()
@click.option('--frac', type=click.FLOAT, default=default_sample_frac)
def main(frac: float = default_sample_frac):
    """
    Creates a sample of the solar wind data and saves it.
    # Parameters
    frac: `float`
        should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include on the sample dataset.
    """
    logging.info(f'making sample of {frac}%')
    logging.info('reading config file')
    config = load_data.read_config_file('./config/config.yml')
    directories = config['directories']
    interim_path = Path(directories['interim'])
    # reading gt data
    logging.info('reading training data')
    solar_wind = load_data.read_feather(interim_path / 'solar_wind.feather')

    logging.info('splitting dataset')
    _, valid_idx = load_data.split_train_data(solar_wind, test_frac=frac,
                                              eval_mode=True)

    sample_data = solar_wind.loc[valid_idx, :]
    sample_data.reset_index(drop=True, inplace=True)
    logging.info('saving file..')
    sample_data.to_feather(interim_path / 'sample_solar_wind.feather')


if __name__ == '__main__':
    main()
