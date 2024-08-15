import sys
from pathlib import Path
from packaging import version
import logging

logging.basicConfig(level='INFO')
logger = logging.getLogger(__name__)

py_version = version.parse(sys.version.split()[0])
logger.info(f"Python version: {py_version}")

__all__ = [
    "data_processing",
    "data_transforming_train",
    "data_transforming_test",
    "data_transforming",
]

compare_version = lambda major, minor: py_version.major == major and py_version.minor == minor

if compare_version(3, 7):
    assert (Path(__file__).parent / 'py39_module').exists(), 'py37_module not found.'
    from .py37_module import data_transforming_train, data_transforming_test
elif compare_version(3, 8):
    assert (Path(__file__).parent / 'py38_module').exists(), 'py38_module not found.'
    from .py38_module import data_transforming_train, data_transforming_test
elif compare_version(3, 9):
    assert (Path(__file__).parent / 'py39_module').exists(), 'py39_module not found.'
    from .py39_module import data_transforming_train, data_transforming_test
elif compare_version(3, 10):
    assert (Path(__file__).parent / 'py310_module').exists(), 'py310_module not found.'
    from .py310_module import data_transforming_train, data_transforming_test
elif compare_version(3, 12):
    assert (Path(__file__).parent / 'py312_module').exists(), 'py312_module not found.'
    from .py312_module import data_transforming_train, data_transforming_test
else:
    raise RuntimeError("Python version should be [3.7, 3.8, 3.9, 3.10, 3.12]")


def data_transforming(cfg_path, slice_num, anchor_path, inputdata_path, save_path, mode='train'):
    if mode == 'train':
        data_transforming_train(cfg_path, slice_num, anchor_path, inputdata_path, save_path)
    elif mode == 'test':
        data_transforming_test(cfg_path, anchor_path, inputdata_path, save_path)
    else:
        raise ValueError(f"Invalid mode: {mode}")
