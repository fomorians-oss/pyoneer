from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from setuptools import setup, find_packages

install_requires = ["gym", "six"]
with open("README.md", "r") as fp:
    long_description = fp.read()

setup(
    name="fomoro-pyoneer",
    version="0.3.0",
    author="Fomoro AI",
    author_email="team@fomoro.com",
    description="Tensor utilities, reinforcement learning, and more!",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fomorians/pyoneer",
    download_url="https://github.com/fomorians/pyoneer/archive/v0.3.tar.gz",
    license="Apache 2.0",
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={"dev": ["pre-commit"]},
    include_package_data=True,
    keywords=[
        "tensorflow",
        "machine learning",
        "reinforcement learning",
        "eager execution",
        "deep learning",
    ],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
)
