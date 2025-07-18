#!/usr/bin/env python

from setuptools import setup, find_packages
#FIXME: prevents pyproject.toml - same as https://github.com/SciTools/cartopy/issues/1270
import versioneer

def unique_flatten_dict(d):
  return list(set(sum( d.values(), [] )))

core_requires = [
  'numpy',
  'palettable >= 3.0',
  'pandas',
  'pyarrow >= 0.15.0',
  'requests',
  'squarify',
  'typing-extensions',
  'packaging >= 20.1',
  'setuptools',
]

stubs = [
  'pandas-stubs', 'types-requests', 'ipython', 'types-tqdm'
]

test_workarounds = []

dev_extras = {
    'docs': [
      'docutils==0.21.2',
      'ipython==8.28',
      'ipykernel==6.29.5',  # For notebook execution validation
      'Jinja2==3.1.4',
      'myst-parser==4.0.0',
      'nbsphinx==0.9.5',
      'pygments>2.10',
      'sphinx==8.0.2',
      #'sphinx_autodoc_typehints==1.11.1',
      'sphinx-copybutton==0.5.2',
      'sphinx-book-theme==1.1.3',
    ],
    'test': ['flake8>=5.0', 'mock', 'mypy', 'pytest', 'pytest-xdist'] + stubs + test_workarounds,
    'testai': [
      'numba>=0.57.1'  # https://github.com/numba/numba/issues/8615
    ],
    'build': ['build']
}

base_extras_light = {
    'igraph': ['igraph'],
    'networkx': ['networkx>=2.5'],
    'gremlin': ['gremlinpython'],
    'bolt': ['neo4j', 'neotime'],
    'nodexl': ['openpyxl==3.1.0', 'xlrd'],
    'jupyter': ['ipython'],
    'spanner': ['google-cloud-spanner'],
    'kusto': ['azure-kusto-data', 'azure-identity']
}

base_extras_heavy = {
  'umap-learn': ['umap-learn','skrub', 'scikit-learn', 'scipy'],
  'pygraphviz': ['pygraphviz'],  # + apt-get graphviz, graphviz-dev
  'rapids': [
    "cudf-cu12==24.12.*", "dask-cudf-cu12==24.12.*", "cuml-cu12==24.12.*",
    "cugraph-cu12==24.12.*", "nx-cugraph-cu12==24.12.*",
    #"cuspatial-cu12==24.12.*",
    #"cuproj-cu12==24.12.*", "cuxfilter-cu12==24.12.*", "cucim-cu12==24.12.*",
    #"pylibraft-cu12==24.12.*", "raft-dask-cu12==24.12.*", "cuvs-cu12==24.12.*",
  ],
}
# https://github.com/facebookresearch/faiss/issues/1589 for faiss-cpu 1.6.1, #'setuptools==67.4.0' removed
base_extras_heavy['ai'] = base_extras_heavy['umap-learn'] + ['scipy', 'dgl', 'torch', 'sentence-transformers', 'faiss-cpu', 'joblib']

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
    python_requires='>=3.8',
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
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
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
    project_urls = {
        "Homepage": "https://www.graphistry.com",
        "Repository": "https://github.com/graphistry/pygraphistry",
        "Documentation": "https://pygraphistry.readthedocs.io/",
        "Changelog": "https://github.com/graphistry/pygraphistry/blob/master/CHANGELOG.md",
        "Issues": "https://github.com/graphistry/pygraphistry/issues",
        "Funding": "https://www.graphistry.com",
        "Download": "https://pypi.org/project/graphistry/#files",
        "Slack": "https://graphistry-community.slack.com",
        "Twitter": "https://twitter.com/graphistry",
        "LinkedIn": "https://www.linkedin.com/company/graphistry",
        "Contributing": "https://github.com/graphistry/pygraphistry/blob/master/CONTRIBUTING.md",
        "License": "https://github.com/graphistry/pygraphistry/blob/main/LICENSE.txt",
        "Code of Conduct": "https://github.com/graphistry/pygraphistry/blob/main/CODE_OF_CONDUCT.md",
        "Support": "https://www.graphistry.com/support",
    },
    keywords=['cugraph', 'cudf', 'cuml', 'dask', 'Databricks', 'GFQL', 'GPU', 'Graph',  'graphviz', 'GraphX', 'Gremlin', 'igraph', 'Jupyter', 'Neo4j', 'Neptune', 'Network', 'NetworkX',  'Notebook', 'OpenSearch', 'Pandas', 'Plot', 'RAPIDS', 'RDF', 'Splunk', 'Spark', 'SQL', 'Tinkerpop', 'UMAP', 'Visualization', 'Torch', 'DGL', 'GNN']
)
