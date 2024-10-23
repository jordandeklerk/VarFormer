"""Common settings"""
# taken from https://github.com/ServiceNow/N-BEATS/blob/master/common/settings.py

import os

STORAGE=os.getenv('STORAGE')
DATASETS_PATH=os.path.join(STORAGE, 'datasets')
EXPERIMENTS_PATH=os.path.join(STORAGE, 'experiments')
TESTS_STORAGE_PATH=os.path.join(STORAGE, 'test')