from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("data_processing.py", language_level="3")
)