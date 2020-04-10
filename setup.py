"""Setup for the pspso package."""

import setuptools
from setuptools import setup

with open('README.md') as f:
    README = f.read()

setuptools.setup(
    author="Ali Haidar",
    author_email="ali.hdrv@outlook.com",
    name='pspso',
    license="MIT",
    description='pspso is a python package for selecting machine learning algorithms parameters.',
    version='v0.0.9',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/ayhaidar/pspso',
    packages=setuptools.find_packages(),
    python_requires=">=3.5",
    install_requires=['numpy==1.16.1','lightgbm','xgboost','scikit-learn>=0.21.2','keras','pyswarms>=1.0.2','matplotlib>=3.1.1'],
    classifiers=[
        # Trove classifiers
        # (https://pypi.python.org/pypi?%3Aaction=list_classifiers)
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
    ],
)