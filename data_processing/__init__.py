"""
Author: Zihan Cao
Email: iamzihan666@gmail.com
Date: 2024/08/17 23:57

                                        Copyright (c) 2024
                                        UESTC, All rights reserved.
"""



"""

import compiled date_processing module based on python version and platform

supported python version: [3.7, 3.8, 3.9, 3.10, 3.12] on linux
supported python version: [3.9] on windows

"""


import sys
from pathlib import Path
from packaging import version
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

py_version = version.parse(sys.version.split()[0])
logger.info(f"Python version: {py_version}")

__all__ = [
    "dataTransforming"
]

# python version
compare_version = lambda major, minor: py_version.major == major and py_version.minor == minor

# platform
platform = sys.platform

logger.info(f"Platform: {platform}")
if platform == 'win32':
    if compare_version(3, 9):
        assert (Path(__file__).parent / 'py39_module').exists(), 'py39_module not found.'
        from .py39_module import data_transforming
    else:
        raise RuntimeError("Python version should be 3.9")
else:
    if compare_version(3, 7):
        assert (Path(__file__).parent / 'py39_module').exists(), 'py37_module not found.'
        from .py37_module import data_transforming
    elif compare_version(3, 8):
        assert (Path(__file__).parent / 'py38_module').exists(), 'py38_module not found.'
        from .py38_module import data_transforming
    elif compare_version(3, 9):
        assert (Path(__file__).parent / 'py39_module').exists(), 'py39_module not found.'
        from .py39_module import data_transforming
    elif compare_version(3, 10):
        assert (Path(__file__).parent / 'py310_module').exists(), 'py310_module not found.'
        from .py310_module import data_transforming
    elif compare_version(3, 12):
        assert (Path(__file__).parent / 'py312_module').exists(), 'py312_module not found.'
        from .py312_module import data_transforming
    else:
        raise RuntimeError("Python version should be [3.7, 3.8, 3.9, 3.10, 3.12]")


def dataTransforming(cfgPath, N, GTPath, dataPath, h5Path, chunkSize):
    """pre-processing the raw data into training and testing data

    Args:
        cfgPath (str): configure data, e.g, Round3CfgData1.txt
        N (str): 500 for training and 20000 for testing
        GTPath (str): anchor data, e.g, Round3InputPos1.txt
        dataPath (str): input data, e.g, Round3InputData1.txt
        h5Path (str): saved h5 file path, e.g, train.h5
        chunkSize (int): chunk size for h5 data, for a fast loading.
    """
    
    data_transforming(cfgPath, N, GTPath, dataPath, h5Path, chunkSize)
