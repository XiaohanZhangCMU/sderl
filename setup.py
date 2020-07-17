from os.path import join, dirname, realpath
from setuptools import setup
import sys

assert sys.version_info.major == 3 and sys.version_info.minor >= 6, \
    "The SDERL repo is designed to work with Python 3.6 and greater." \
    + "Please install it before proceeding."

with open(join("sderl", "version.py")) as version_file:
    exec(version_file.read())

setup(
    name='sderl',
    py_modules=['sderl'],
    version=__version__,#'0.1',
    install_requires=[
        'cloudpickle',
        'gym[atari,box2d,classic_control]',
        'ipython',
        'joblib',
        'matplotlib',
        'mpi4py',
        'numpy',
        'pandas',
        'pytest',
        'psutil',
        'scipy',
        'seaborn',
        'torch',
        'tqdm'
    ],
    description="research on stochastic differential equations and deep RL.",
    author="Xiaohan Zhang",
)
