import os
import sys
import bz2
import imp
from setuptools import setup, find_packages

weight_file = 'model.h5'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # include the compressed weights file in sdist
    weight_file = 'model.h5.bz2'
else:
    # in all other cases, decompress the weights file if necessary
    if not os.path.isfile(os.path.join('crepe', 'model.h5')):
        print('Decompressing the model weights ...')
        with bz2.BZ2File(os.path.join('crepe', 'model.h5.bz2'), 'rb') as source:
            with open(os.path.join('crepe', 'model.h5'), 'wb') as target:
                target.write(source.read())
        print('Decompression complete')

version = imp.load_source('crepe.version', os.path.join('crepe', 'version.py'))

with open('README.md') as file:
    long_description = file.read()

setup(
    name='crepe',
    version=version.version,
    description='CREPE pitch tracker',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/marl/crepe',
    author='Jong Wook Kim',
    author_email='jongwook@nyu.edu',
    packages=find_packages(),
    entry_points = {
        'console_scripts': ['crepe=crepe.cli:main'],
    },
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='tfrecord',
    project_urls={
        'Source': 'https://github.com/marl/crepe',
        'Tracker': 'https://github.com/marl/crepe/issues'
    },
    install_requires=[
        'keras==2.1.5',
        'numpy>=1.14.0',
        'scipy>=1.0.0',
        'matplotlib>=2.1.0',
        'resampy>=0.2.0,<0.3.0',
        'h5py>=2.7.0,<3.0.0',
        'hmmlearn>=0.2.0,<0.3.0',
        'imageio>=2.3.0',
        'scikit-learn>=0.16'
    ],
    package_data={
        'crepe': [weight_file]
    },
)
