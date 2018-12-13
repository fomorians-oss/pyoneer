from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

setup(
    name='pyoneer',
    version='0.0.0',
    description='pyoneer',
    url='https://github.com/fomorians/pyoneer',
    packages=find_packages(),
    install_requires=[
        'tensorflow',
        'tensorflow-probability',
        'gym',
        'trfl',
        'dm-sonnet',
        'wrapt',
    ],
    dependency_links=['git+ssh://git@github.com/deepmind/trfl.git#egg=trfl'])