import os
import subprocess
import sys
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

subprocess.check_call(['make'])

# Read the contents of README.md
with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    readme = f.read()

# Setup configuration
setup(
    name='numba_RKDP',
    version='0.0.1',
    description='A package that integrates the C implementation of Adaptive Doramnd-Prince solver with Numba',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/ihateemoji/numba_RKDP',
    packages=['numba_RKDP'],
    package_data={'numba_RKDP': ['lib/libRKDP.so']},
    include_package_data=True,
    install_requires=[
        'numba',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
