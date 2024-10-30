import os

# Set the base path to the current directory
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORAGE = BASE_PATH
DATASETS_PATH = os.path.join(STORAGE, 'datasets')
