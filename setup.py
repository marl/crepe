import bz2
import imp
import os
import sys

import pkg_resources
from setuptools import setup, find_packages

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

model_capacities = ['tiny', 'small', 'medium', 'large', 'full']
weight_files = ['model-{}.h5'.format(cap) for cap in model_capacities]
base_url = 'https://github.com/marl/crepe/raw/models/'

if len(sys.argv) > 1 and sys.argv[1] == 'sdist':
    # exclude the weight files in sdist
    weight_files = []
else:
    # in all other cases, decompress the weights file if necessary
    for weight_file in weight_files:
        weight_path = os.path.join('crepe', weight_file)
        if not os.path.isfile(weight_path):
            compressed_file = weight_file + '.bz2'
            compressed_path = os.path.join('crepe', compressed_file)
            if not os.path.isfile(compressed_file):
                print('Downloading weight file {} ...'.format(compressed_file))
                urlretrieve(base_url + compressed_file, compressed_path)
            print('Decompressing ...')
            with bz2.BZ2File(compressed_path, 'rb') as source:
                with open(weight_path, 'wb') as target:
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
    author='Jong Wook Kim and Justin Salamon',
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
        str(requirement)
        for requirement in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    package_data={
        'crepe': weight_files
    },
)
