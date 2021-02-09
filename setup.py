#!/usr/bin/env python

from setuptools import setup, find_packages
#FIXME: prevents pyproject.toml - same as https://github.com/SciTools/cartopy/issues/1270
import versioneer

long_description = """

## PyGraphistry: Explore Relationships

PyGraphistry is a Python visual graph analytics library to extract, transform, and load big graphs into Graphistry end-to-end GPU visual graph analytics sessions.

Graphistry gets used on problems like visually mapping the behavior of devices and users and for analyzing machine learning results. It provides point-and-click features like timebars, search, filtering, clustering, coloring, sharing, and more. Graphistry is the only tool built ground-up for large graphs. The client's WebGL streaming rendering engine frontend shows up to 8MM nodes + edges at a time, and most older client GPUs smoothly support somewhere between 100K and 1MM elements. The serverside GPU analytics engine supports even bigger graphs and enables sharing and embedding your results.

The PyGraphistry Python client helps several kinds of usage modes:

* **Data scientists:** Go from data to accelerated visual explorations in a couple lines, share live results, build up more advanced views over time, and do it all from notebook environments like Jupyter and Google Colab

* **Developers:** Quickly prototype stunning Python solutions with PyGraphistry, embed in a language-neutral way with the REST APIs, and go deep on customizations like colors, icons, layouts, JavaScript, and more

* **Analysts:** Every Graphistry session is a point-and-click environment with interactive search, filters, timebars, histograms, and more
Dashboarding: Embed into your favorite framework. Additionally, see our sister project Graph-App-Kit for quickly building interactive graph dashboards by launching a stack built on PyGraphistry, StreamLit, Docker, and ready recipes for integrating with common graph libraries.

PyGraphistry is a friendly and optimized PyData-native interface to the language-neutral Graphistry REST APIs. You can use PyGraphistry with traditional Python data sources like CSVs, SQL, Neo4j, Splunk, and more (see below). Wrangle data however you want, and with especially good support for Pandas dataframes, Apache Arrow tables, and Nvidia RAPIDS cuDF dataframes.


Try It Out!
-----------

Tutorial and API docs are on
`https://github.com/graphistry/pygraphistry <https://github.com/graphistry/pygraphistry>`_

Free GPU accounts with `https://hub.graphistry.com <https://hub.graphistry.com`_
"""


def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

core_requires = ['numpy', 'pandas >= 0.17.0', 'pyarrow >= 0.15.0', 'requests', 'protobuf >= 2.6.0']

dev_extras ={
    'test': ['pytest', 'mock', 'flake8'],
    'build': ['build']
}
base_extras = {
    'igraph': ['python-igraph'],
    'networkx': ['networkx==2.2'],
    'bolt': ['neo4j', 'neotime'],
    'nodexl': ['openpyxl', 'xlrd'],
    'jupyter': ['ipython']
}

extras_require = {
  **base_extras,
  **dev_extras,
  'dev': unique_flatten_dict(base_extras) + unique_flatten_dict(dev_extras),
  'all': unique_flatten_dict(base_extras)
}

setup(
    name='graphistry',
    version=versioneer.get_version(),
    packages = ['graphistry'],
    platforms='any',
    description = 'A visual graph analytics library for extracting, transforming, displaying, and sharing big graphs with end-to-end GPU acceleration',
    long_description=long_description,
    url='https://github.com/graphistry/pygraphistry',
    download_url= 'https://pypi.python.org/pypi/graphistry/',
    python_requires='>=3.6',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    install_requires=core_requires,
    extras_require=extras_require,
    cmdclass=versioneer.get_cmdclass(),
    license='BSD',
    classifiers=[
        'Development Status :: 6 - Mature',
        'Environment :: Console',
        'Environment :: GPU :: NVIDIA CUDA',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Information Technology',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Internet :: Log Analysis',
        'Topic :: Multimedia :: Graphics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Visualization',
        'Topic :: Sociology',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: User Interfaces',
        'Topic :: Software Development :: Widget Sets',
        'Topic :: System :: Distributed Computing'
    ],
    keywords=['cugraph', 'cudf', 'dask', 'GPU', 'Graph',  'GraphX', 'Gremlin', 'igraph', 'Jupyter', 'Neo4j', 'Network', 'NetworkX',  'Notebook', 'Pandas', 'Plot', 'Rapids', 'RDF', 'Splunk', 'Spark', 'Tinkerpop', 'Visualization']
)
