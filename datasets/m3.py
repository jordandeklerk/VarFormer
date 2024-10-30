"""M3 Dataset"""
# taken from https://github.com/ServiceNow/N-BEATS/blob/master/datasets/m3.py

import logging
import os
from dataclasses import dataclass

import fire
import numpy as np
import pandas as pd

from common.http_utils import download, url_file_name
from common.settings import DATASETS_PATH

DATASET_URL = 'https://forecasters.org/data/m3comp/M3C.xls'
FORECASTS_URL = 'https://forecasters.org/data/m3comp/M3Forecast.xls'

DATASET_PATH = os.path.join(DATASETS_PATH, 'm3')
DATASET_FILE_PATH = os.path.join(DATASET_PATH, url_file_name(DATASET_URL))

TRAINING_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'training.npy')
TEST_SET_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'test.npy')
IDS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'ids.npy')
GROUPS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'groups.npy')
HORIZONS_CACHE_FILE_PATH = os.path.join(DATASET_PATH, 'horizons.npy')


@dataclass()
class M3Meta:
    seasonal_patterns = ['M3Year', 'M3Quart', 'M3Month', 'M3Other']
    horizons = [6, 8, 18, 8]
    frequency = [1, 4, 12, 1]
    horizons_map = {
        'M3Year': 6,
        'M3Quart': 8,
        'M3Month': 18,
        'M3Other': 8
    }
    frequency_map = {
        'M3Year': 1,
        'M3Quart': 4,
        'M3Month': 12,
        'M3Other': 1
    }


@dataclass()
class M3Dataset:
    ids: np.ndarray
    groups: np.ndarray
    horizons: np.ndarray
    values: np.ndarray

    @staticmethod
    def load(training: bool = True) -> 'M3Dataset':
        values_file = TRAINING_SET_CACHE_FILE_PATH if training else TEST_SET_CACHE_FILE_PATH
        return M3Dataset(ids=np.load(IDS_CACHE_FILE_PATH, allow_pickle=True),
                         groups=np.load(GROUPS_CACHE_FILE_PATH, allow_pickle=True),
                         horizons=np.load(HORIZONS_CACHE_FILE_PATH, allow_pickle=True),
                         values=np.load(values_file, allow_pickle=True))

    def to_training_subset(self):
        return M3Dataset(ids=self.ids,
                         groups=self.groups,
                         horizons=self.horizons,
                         values=np.array([v[:-self.horizons[i]] for i, v in enumerate(self.values)]))

    def to_hp_search_training_subset(self):
        return M3Dataset(ids=self.ids,
                         groups=self.groups,
                         horizons=self.horizons,
                         values=np.array([v[:-2 * self.horizons[i]] for i, v in enumerate(self.values)]))

    @staticmethod
    def download() -> None:
        """
        Download M3 dataset if doesn't exist.
        """
        if not os.path.exists(DATASET_PATH):
            os.makedirs(DATASET_PATH)
        
        if os.path.exists(TRAINING_SET_CACHE_FILE_PATH) and os.path.exists(TEST_SET_CACHE_FILE_PATH):
            logging.info(f'skip: {DATASET_PATH} cache files already exist.')
            return

        download(DATASET_URL, DATASET_FILE_PATH)
        logging.info('Load and cache forecasts ...')

        ids = []
        groups = []
        horizons = []
        training_values = []
        test_values = []

        for sp in M3Meta.seasonal_patterns:
            horizon = M3Meta.horizons_map[sp]
            dataset = pd.read_excel(DATASET_FILE_PATH, sheet_name=sp)
            
            # Process each time series
            for idx, row in dataset.iterrows():
                # Convert series to numeric values and handle NaN
                series = pd.to_numeric(row[dataset.columns[6:]], errors='coerce').values
                series = series[pd.notna(series)]  # Remove NaN values
                
                if len(series) > horizon:  # Only include if we have enough data
                    ids.append(row['Series'])
                    horizons.append(row['NF'])
                    groups.append(sp)
                    training_values.append(series[:-horizon])
                    test_values.append(series[-horizon:])

        # Save processed data
        os.makedirs(os.path.dirname(TRAINING_SET_CACHE_FILE_PATH), exist_ok=True)
        
        # Convert lists to arrays and save
        np.save(IDS_CACHE_FILE_PATH, np.array(ids), allow_pickle=True)
        np.save(GROUPS_CACHE_FILE_PATH, np.array(groups), allow_pickle=True)
        np.save(HORIZONS_CACHE_FILE_PATH, np.array(horizons), allow_pickle=True)
        np.save(TRAINING_SET_CACHE_FILE_PATH, np.array(training_values, dtype=object), allow_pickle=True)
        np.save(TEST_SET_CACHE_FILE_PATH, np.array(test_values, dtype=object), allow_pickle=True)

        # Save CSV versions for easier access
        train_df = pd.DataFrame(training_values)
        test_df = pd.DataFrame(test_values)
        train_df.to_csv(os.path.join(DATASET_PATH, 'Monthly-train.csv'), index=False)
        test_df.to_csv(os.path.join(DATASET_PATH, 'Monthly-test.csv'), index=False)


if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    fire.Fire()
