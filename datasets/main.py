"""
Datasets module
"""
import logging
import ssl

from fire import Fire

from datasets.m3 import M3Dataset
from datasets.m4 import M4Dataset


def build():
    """
    Download all datasets.
    """

    logging.info('M4 Dataset')
    M4Dataset.download()

    logging.info('\n\nM3 Dataset')
    M3Dataset.download()

if __name__ == '__main__':
    logging.root.setLevel(logging.INFO)
    Fire()