# -*- coding: utf-8 -*-
"""
Created on Wed May  7 15:50:18 2025

@author: Giannis
"""

# setup.py
from setuptools import setup, find_packages

setup(
    name="hosa_toolbox",
    version="1.0.0",
    author="Ioannismk",
    description="Higher-Order Spectral Analysis Toolbox for Python",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Ioannismk/HOSA-toolbox",  # optional
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
