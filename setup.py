from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

REQUIRED_PACKAGES = [
    'tensorflow',
    'tensorflow-probability',
    'gym',
    'trfl',
    'dm-sonnet',
    'wrapt',
]

setup(
    name='pyoneer',
    version='0.0.0',
    url='https://github.com/fomorians/pyoneer',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True)
