from setuptools import setup, find_packages

setup(
    name='crepe',
    version='0.0.1',
    description='CREPE pitch tracker',
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
    install_requires=open('requirements.txt').read().split('\n'),
    package_data={
        'crepe': ['model.h5']
    },
)
