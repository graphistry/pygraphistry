#!/usr/bin/env python

from setuptools import setup, find_packages
#FIXME: prevents pyproject.toml - same as https://github.com/SciTools/cartopy/issues/1270
import versioneer

def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

core_requires = [
  'numpy',
  'palettable >= 3.0',
  'pandas >= 0.17.0',
  'pyarrow >= 0.15.0',
  'requests',
  'squarify',
  'typing-extensions',
  'packaging >= 20.1'
]

stubs = [
  'pandas-stubs', 'types-requests', 'ipython', 'tqdm-stubs'
]

dev_extras = {
    'docs': ['sphinx==3.4.3', 'docutils==0.16', 'sphinx_autodoc_typehints==1.11.1', 'sphinx-rtd-theme==0.5.1', 'Jinja2<3.1'],
    'test': ['flake8', 'mock', 'mypy', 'pytest'] + stubs,
    'build': ['build']
}

base_extras_light = {
    'igraph': ['python-igraph'],
    'networkx': ['networkx>=2.5'],
    'gremlin': ['gremlinpython'],
    'bolt': ['neo4j', 'neotime'],
    'nodexl': ['openpyxl', 'xlrd'],
    'jupyter': ['ipython'],
}

base_extras_heavy = {
  'umap-learn': ['umap-learn', 'dirty-cat==0.2.0', 'scikit-learn>=1.0'],
}
base_extras_heavy['ai'] = base_extras_heavy['umap-learn'] + ['scipy', 'dgl', 'torch', 'sentence-transformers', 'annoy', 'joblib']

base_extras = {**base_extras_light, **base_extras_heavy}

extras_require = {

  **base_extras_light,
  **base_extras_heavy,
  **dev_extras,

  #kitchen sink for users -- not recommended
  'all': unique_flatten_dict(base_extras),

  #kitchen sink for contributors, skips ai
  'dev': unique_flatten_dict(base_extras_light) + unique_flatten_dict(dev_extras),

}

setup(
    name='graphistry',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages = find_packages(),
    platforms='any',
    description = 'A visual graph analytics library for extracting, transforming, displaying, and sharing big graphs with end-to-end GPU acceleration',
    long_description=open("./README.md").read(),
    long_description_content_type='text/markdown',
    url='https://github.com/graphistry/pygraphistry',
    download_url= 'https://pypi.python.org/pypi/graphistry/',
    python_requires='>=3.7',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Internet :: Log Analysis',
        'Topic :: Database :: Front-Ends',
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
    keywords=['cugraph', 'cudf', 'dask', 'GPU', 'Graph',  'GraphX', 'Gremlin', 'igraph', 'Jupyter', 'Neo4j', 'Network', 'NetworkX',  'Notebook', 'Pandas', 'Plot', 'Rapids', 'RDF', 'Splunk', 'Spark', 'Tinkerpop', 'Visualization', 'Torch', 'DGL', 'GNN']
)
