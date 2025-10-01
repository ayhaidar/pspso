"""Setup script for the pspso package.

This script configures packaging metadata and dependencies for
distribution on PyPI.  The version has been bumped from the original
`v0.1.3` tag to a modern semantic version and the dependency pins have
been updated to target contemporary releases of third‑party libraries.

The install_requires list deliberately omits pinned versions of heavy
scientific dependencies; instead it specifies minimum versions known to
work with the codebase.  Users can install the package with
`pip install pspso` to pull in the dependencies automatically.  For
development or testing on environments without GPU support or when
network access is unavailable, use the `--no-deps` flag to skip
installing dependencies and instead provide them manually.
"""

from pathlib import Path
from setuptools import setup, find_packages

base_dir = Path(__file__).parent
readme_path = base_dir / 'README.md'

# Read long description from README.md
long_description = readme_path.read_text(encoding='utf-8')

setup(
    name='pspso',
    version='0.1.4',
    author='Ali Haidar',
    author_email='ali.hdrv@outlook.com',
    description='PSPSO: Hyper‑parameter optimisation via Particle Swarm Optimisation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    url='https://github.com/ayhaidar/pspso',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.21',
        'lightgbm>=3.3.5',
        'xgboost>=1.6.0',
        'scikit-learn>=1.1.0',
        'tensorflow>=2.6.0',  # includes Keras API
        'pyswarms>=1.3.0',
        'matplotlib>=3.5.0',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)