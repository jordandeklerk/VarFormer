"""
Datasets module
"""
import logging
import ssl
import gc
import sys

from fire import Fire

from datasets.m3 import M3Dataset
from datasets.m4 import M4Dataset

def build(dataset=None):
    """
    Download datasets.
    Args:
        dataset: Optional[str] - Specific dataset to download ('m3' or 'm4'). If None, downloads all.
    """
    if dataset == 'm3' or dataset is None:
        logging.info('M3 Dataset')
        M3Dataset.download()
        gc.collect()  # Force garbage collection

    if dataset == 'm4' or dataset is None:
        logging.info('\n\nM4 Dataset')
        M4Dataset.download()
        gc.collect()  # Force garbage collection

if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()
