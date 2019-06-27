from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from setuptools import setup, find_packages

REQUIRED_PACKAGES = ["gym"]
README_PATH = os.path.join(os.path.dirname(__file__), "README.md")

with open(README_PATH, "r") as fp:
    README = fp.read()

setup(
    name="fomoro-pyoneer",
    version="0.1.0",
    description="""
    Tensor utilities, reinforcement learning, and more!
    Designed to make research easier with low-level abstractions for common operations.
    """,
    author="Fomoro AI",
    author_email="team@fomoro.com",
    long_description=README,
    long_description_content_type="text/markdown",
    license="Apache 2.0",
    url="https://github.com/fomorians/pyoneer",
    install_requires=REQUIRED_PACKAGES,
    extras_require={"dev": ["pre-commit"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
)
