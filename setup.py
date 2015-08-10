#!/usr/bin/env python

from setuptools import setup, find_packages


setup(
    name='graphistry',
    version='0.9.1',
    packages = ['graphistry'],
    platforms='any',
    description = 'Visualize node-link graphs using Graphistry\'s cloud',
    #long_description=long_description,
    url='https://github.com/graphistry/pygraphistry',
    download_url= 'https://pypi.python.org/pypi/graphistry/',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    setup_requires=['numpy'],
    install_requires=['numpy', 'pandas', 'requests', 'future'],
    extras_require={
        'igraph': ['python-igraph'],
        'networkx': ['networkx'],
        'pandas-extra': ['numexpr', 'Bottleneck'],
        'all': ['python-igraph', 'networkx', 'numexpr', 'Bottleneck']
    },
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords=['Graph', 'Network', 'Plot', 'Visualization', 'Pandas', 'Igraph']
)
