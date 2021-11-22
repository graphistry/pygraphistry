#!/usr/bin/env python

from setuptools import setup, find_packages
#FIXME: prevents pyproject.toml - same as https://github.com/SciTools/cartopy/issues/1270
import versioneer

def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

core_requires = ['numpy', 'pandas >= 0.17.0', 'pyarrow >= 0.15.0', 'requests', 'protobuf >= 2.6.0']

stubs = [
  'pandas-stubs', 'types-requests'
]

dev_extras = {
    'docs': ['sphinx==3.4.3', 'docutils==0.16', 'sphinx_autodoc_typehints==1.11.1', 'sphinx-rtd-theme==0.5.1'],
    'test': ['flake8', 'mock', 'mypy', 'pytest'] + stubs,
    'build': ['build']
}

base_extras = {
    'igraph': ['python-igraph'],
    'networkx': ['networkx==2.2'],
    'gremlin': ['gremlinpython'],
    'bolt': ['neo4j', 'neotime'],
    'nodexl': ['openpyxl', 'xlrd'],
    'jupyter': ['ipython']
}

extras_require = {

  **base_extras,
  **dev_extras,

  #kitchen sink for users -- not recommended
  'all': unique_flatten_dict(base_extras),

  #kitchen sink for contributors
  'dev': unique_flatten_dict(base_extras) + unique_flatten_dict(dev_extras),

}

setup(
    name='graphistry',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages = ['graphistry'],
    platforms='any',
    description = 'A visual graph analytics library for extracting, transforming, displaying, and sharing big graphs with end-to-end GPU acceleration',
    long_description=open("./README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/graphistry/pygraphistry',
    download_url= 'https://pypi.python.org/pypi/graphistry/',
    python_requires='>=3.6',
    author='The Graphistry Team',
    author_email='pygraphistry@graphistry.com',
    install_requires=core_requires,
    extras_require=extras_require,
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
