from pathlib import Path
import load_data
import logging


logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(format=log_fmt,
                    level=logging.INFO)


def main():
    """
    This function will save the solar wind data
    as a Feather file
    """
    # read the main config file
    config = load_data.read_config_file('./config/config.yml')
    # get the path to the CSV File
    directories = config['directories']
    raw_path = Path(directories['raw'])
    interim_path = Path(directories['interim'])
    interim_path.mkdir(exist_ok=True, parents=True)
    logging.info('reading solar wind data..')
    # reading CSV file
    solar_wind = load_data.read_csv(raw_path / 'solar_wind.csv')
    logging.info('saving to feather..')
    # saving as feather file
    solar_wind.to_feather(interim_path / 'solar_wind.feather')


if __name__ == '__main__':
    main()
