"""M4 Dataset"""
# taken from https://github.com/ServiceNow/N-BEATS/blob/master/datasets/m4.py

import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from glob import glob

import numpy as np
import pandas as pd
import patoolib
from tqdm import tqdm

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

FREQUENCIES = ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Quarterly', 'Yearly']
URL_TEMPLATE = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/{}/{}-{}.csv'

TRAINING_DATASET_URLS = [URL_TEMPLATE.format("Train", freq, "train") for freq in FREQUENCIES]
TEST_DATASET_URLS = [URL_TEMPLATE.format("Test", freq, "test") for freq in FREQUENCIES]
INFO_URL = 'https://github.com/Mcompetitions/M4-methods/raw/master/Dataset/M4-info.csv'

DATASET_PATH = os.path.join(DATASETS_PATH, 'm4')

TRAINING_DATASET_FILE_PATHS = [os.path.join(DATASET_PATH, url_file_name(url)) for url in TRAINING_DATASET_URLS]
TEST_DATASET_FILE_PATHS = [os.path.join(DATASET_PATH, url_file_name(url)) for url in TEST_DATASET_URLS]
INFO_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(INFO_URL))

TRAINING_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npz')
TEST_DATASET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npz')


@dataclass()
class M4Dataset:
    ids: np.ndarray
    groups: np.ndarray
    frequencies: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'M4Dataset':
        """
        Load cached dataset.

        :param training: Load training part if training is True, test part otherwise.
        """
        m4_info = pd.read_csv(INFO_FILE_PATH)
        return M4Dataset(ids=m4_info.M4id.values,
                         groups=m4_info.SP.values,
                         frequencies=m4_info.Frequency.values,
                         horizons=m4_info.Horizon.values,
                         values=np.load(
                             TRAINING_DATASET_CACHE_FILE_PATH if training else TEST_DATASET_CACHE_FILE_PATH,
                             allow_pickle=True))

    @staticmethod
    def download() -> None:
        """
        Download M4 dataset if doesn't exist.
        """
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)

        # Download info file
        download(INFO_URL, INFO_FILE_PATH)
        m4_info = pd.read_csv(INFO_FILE_PATH)
        m4_ids = m4_info.M4id.values

        # Process training data
        all_training_data = []
        for freq, url, path in zip(FREQUENCIES, TRAINING_DATASET_URLS, TRAINING_DATASET_FILE_PATHS):
            logging.info(f'Downloading {freq} training data...')
            download(url, path)
            df = pd.read_csv(path)
            all_training_data.append(df)
        
        # Combine all training data
        combined_train = pd.concat(all_training_data, axis=0)
        combined_train.to_csv(os.path.join(DATASET_PATH, 'Monthly-train.csv'), index=False)
        
        # Process test data
        all_test_data = []
        for freq, url, path in zip(FREQUENCIES, TEST_DATASET_URLS, TEST_DATASET_FILE_PATHS):
            logging.info(f'Downloading {freq} test data...')
            download(url, path)
            df = pd.read_csv(path)
            all_test_data.append(df)
        
        # Combine all test data
        combined_test = pd.concat(all_test_data, axis=0)
        combined_test.to_csv(os.path.join(DATASET_PATH, 'Monthly-test.csv'), index=False)

        # Save numpy arrays
        timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
        
        logging.info('Processing training data...')
        for df in all_training_data:
            df.set_index(df.columns[0], inplace=True)
            for m4id, row in df.iterrows():
                values = pd.to_numeric(row.values, errors='coerce')
                timeseries_dict[m4id] = values[~np.isnan(values)]
        
        np.savez(TRAINING_DATASET_CACHE_FILE_PATH, 
                 data=np.array(list(timeseries_dict.values()), dtype=object),
                 allow_pickle=True)
        
        # Reset for test data
        timeseries_dict = OrderedDict(list(zip(m4_ids, [[]] * len(m4_ids))))
        
        logging.info('Processing test data...')
        for df in all_test_data:
            df.set_index(df.columns[0], inplace=True)
            for m4id, row in df.iterrows():
                values = pd.to_numeric(row.values, errors='coerce')
                timeseries_dict[m4id] = values[~np.isnan(values)]
        
        np.savez(TEST_DATASET_CACHE_FILE_PATH, 
                 data=np.array(list(timeseries_dict.values()), dtype=object),
                 allow_pickle=True)


@dataclass()
class M4Meta:
    seasonal_patterns = ['Yearly', 'Quarterly', 'Monthly', 'Weekly', 'Daily', 'Hourly']
    horizons = [6, 8, 18, 13, 14, 48]
    frequencies = [1, 4, 12, 1, 1, 24]
    horizons_map = {
        'Yearly': 6,
        'Quarterly': 8,
        'Monthly': 18,
        'Weekly': 13,
        'Daily': 14,
        'Hourly': 48
    }
    frequency_map = {
        'Yearly': 1,
        'Quarterly': 4,
        'Monthly': 12,
        'Weekly': 1,
        'Daily': 1,
        'Hourly': 24
    }

def load_m4_info() -> pd.DataFrame:
    """
    Load M4Info file.

    :return: Pandas DataFrame of M4Info.
    """
    return pd.read_csv(INFO_FILE_PATH)
