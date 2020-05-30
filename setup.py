#!/usr/bin/env python

from setuptools import setup, find_packages
import versioneer

long_description = """
**PyGraphistry** is a visual graph analytics library to extract, transform, and
load big graphs into `Graphistry's <http://www.graphistry.com>`_ GPU-cloud-accelerated
explorer.

PyGraphistry is...

- **Fast & Gorgeous**: Cluster, filter, and inspect large amounts of data at
  interactive speed. We layout graphs with a descendant of the gorgeous
  ForceAtlas2 layout algorithm introduced in Gephi. Our data explorer connects
  to Graphistry's GPU cluster to layout and render hundreds of thousand of
  nodes+edges in your browser at unparalleled speeds.

- **Notebook Friendly**: PyGraphistry plays well with interactive notebooks
  like IPython/Juypter, Zeppelin, and Databricks: Process, visualize, and drill
  into with graphs directly within your notebooks.

- **Batteries Included**: PyGraphistry works out-of-the-box with popular data
  science and graph analytics libraries. It is also very easy to use. To create
  the visualization shown above, download this dataset of Facebook communities
  from SNAP and load it with your favorite library


Try It Out!
-----------

Tutorial and API docs are on
`https://github.com/graphistry/pygraphistry <https://github.com/graphistry/pygraphistry>`_
"""

setup(
    name='graphistry',
    version=versioneer.get_version(),
    packages = ['graphistry'],
    platforms='any',
    description = 'Visualize node-link graphs using Graphistry\'s cloud',
    long_description=long_description,
    url='https://github.com/graphistry/pygraphistry',
    download_url= 'https://pypi.python.org/pypi/graphistry/',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    setup_requires=['numpy', 'pytest-runner'],
    install_requires=['numpy', 'pandas >= 0.17.0', 'pyarrow >= 0.15.0', 'requests', 'future >= 0.15.0', 'protobuf >= 2.6.0'],
    extras_require={
        'igraph': ['python-igraph'],
        'networkx': ['networkx'],
        'bolt': ['neo4j', 'neotime'],
        'nodexl': ['openpyxl', 'xlrd'],
        'all': ['python-igraph', 'networkx', 'colorlover', 'neo4j', 'neotime']
    },
    tests_require=
        ['pytest', 'mock', 'ipython', 
        'python-igraph', 'networkx==2.2', 'colorlover', 
        'neo4j', 'neotime',
        'openpyxl', 'xlrd'],
    cmdclass=versioneer.get_cmdclass(),
    license='BSD',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Environment :: Console',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',        
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    keywords=['Graph', 'Network', 'Plot', 'Visualization', 'Pandas', 'Igraph', 'Jupyter', 'Notebook', 'Neo4j', 'Gremlin', 'Tinkerpop', 'RDF', 'GraphX', 'NetworkX', 'Splunk', 'Spark']
)
