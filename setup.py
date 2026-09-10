import os
import platform
import subprocess
from setuptools import setup

def get_lib_filename():
    sysname = platform.system()
    if sysname == 'Darwin':
        return 'libRKDP.dylib'
    if sysname == 'Windows':
        return 'libRKDP.dll'
    return 'libRKDP.so'

lib_filename = get_lib_filename()
lib_path = os.path.join('numba_RKDP', 'lib', lib_filename)

if not os.path.exists(lib_path):
    subprocess.check_call(['make'])

with open(os.path.join(os.path.dirname(__file__), 'README.md'), encoding='utf-8') as f:
    readme = f.read()

setup(
    name='numba_RKDP',
    version='0.1.0',
    description='A package that integrates the C implementation of Adaptive Doramnd-Prince solver with Numba',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='https://github.com/ihateemoji/numba_RKDP',
    packages=['numba_RKDP'],
    package_data={'numba_RKDP': [os.path.join('lib', lib_filename)]},
    include_package_data=True,
    install_requires=['numba'],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: C',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
