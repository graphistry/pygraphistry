PyGraphistry: Explore Relationships
========================================
.. only:: html

   .. image:: https://readthedocs.org/projects/pygraphistry/badge/?version=latest
      :target: https://pygraphistry.readthedocs.io/en/latest/?badge=latest
      :alt: Documentation Status


   .. image:: https://github.com/graphistry/pygraphistry/workflows/CI%20Tests/badge.svg
      :target: https://github.com/graphistry/pygraphistry/workflows/CI%20Tests/badge.svg
      :alt: Build Status

   .. image:: https://img.shields.io/pypi/v/graphistry.svg
      :target: https://pypi.python.org/pypi/graphistry
      :alt: PyPi Status

   .. image:: https://img.shields.io/pypi/dm/graphistry
      :target: https://img.shields.io/pypi/dm/graphistry
      :alt: PyPi Downloads

   .. image:: https://img.shields.io/pypi/l/graphistry.svg
      :target: https://pypi.python.org/pypi/graphistry
      :alt: License

   .. .. image:: https://img.shields.io/uptimerobot/status/m787548531-e9c7b7508fc76fea927e2313?label=hub.graphistry.com
   ..    :target: https://img.shields.io/uptimerobot/status/m787548531-e9c7b7508fc76fea927e2313?label=hub.graphistry.com
   ..    :alt: License

   .. .. image:: https://img.shields.io/badge/slack-Graphistry%20chat-orange.svg?logo=slack
   ..       :target: https://join.slack.com/t/graphistry-community/shared_invite/zt-53ik36w2-fpP0Ibjbk7IJuVFIRSnr6g
   ..       :alt: Slack

   .. image:: https://img.shields.io/twitter/follow/graphistry
         :target: https://twitter.com/graphistry
         :alt: Twitter

.. Quickstart:
.. `Read our tutorial <https://github.com/graphistry/pygraphistry/blob/master/README.md>`_

PyGraphistry is a Python visual graph AI library to extract, transform, query, analyze, model, and visualize big graphs. It includes several notable pieces:
* Client to using the optional Graphistry server for GPU compute and visualization
* A variety of efficient dataframe-native tabular and graph methods for loading and transforming graphs
* The GFQL dataframe-native graph query language with optional GPU support
* Graphistry[AI]: Installing optional graphistry[ai] dependencies adds graph autoML, including automatic feature engineering, UMAP, and graph neural net support
* Plugins: Query databases like Neo4j, compute systems like Nvidia RAPIDS, and frameworks like igraph. PyGraphistry enables working with fast and familiar dataframe data representations, and PyGraphistry takes care of efficient conversions.

Combined, PyGraphistry reduces your time to graph for going from raw data to visualizations to AI models in a few lines of code.

The API reference documentation here provides useful packages, modules, and commands for PyGraphistry. In the navbar you can find an overview of all the packages and modules we provided and a few useful highlighted ones as well. You can search for them on our Search page. For a full tutorial, refer to our `PyGraphistry <https://github.com/graphistry/pygraphistry/>`_ repo.



.. .. image:: docs/static/docstring.png
..       :width: 600
..       :alt: PyGraphistry


.. Click to open interactive version! (For server-backed interactive analytics, use an API key)


.. .. raw:: html

..        <iframe width="600" height="400" src="https://hub.graphistry.com/graph/graph.html?dataset=Facebook&splashAfter=true" frameborder="0" allowfullscreen></iframe>

For self-hosting and access to a free API key, refer to our Graphistry `Hub <https://hub.graphistry.com/>`_.



Quickstart
----------

Here's a representative example of how to get started:

.. code-block:: python

    import graphistry
    import pandas as pd

    # Load a simple graph from a dataframe
    df = pd.DataFrame({
      'src': ['A', 'B', 'C'],
      'dst': ['B', 'C', 'A'],
      'nfo': ['X', 'Y', 'Z']
      'abc': [1, 2, 3]
    })
    g1 = graphistry.edges(df, source="src", destination="dst")

    # Optionally convert to GPU if available
    import cudf
    g2 = g1.edges(cudf.from_pandas(g1._edges))

    # Wrangle the graph, using automatic GPU acceleration
    g3 = g2.materialize_nodes().get_degrees()
    print(g3._nodes[['id', 'degree']])

    # Server-accelerated visualization session
    # Use a private server, or a free remote GPU account at graphistry.com/get-started (Graphistry Hub)
    graphistry.register(api=3, username="my_username", password="my_password")
    g3.plot()  # Upload the data and start a visual GPU session

    # Visualize your data as a similarity graph using a UMAP dimensionality reduction
    # umap will feature-encode your data, run UMAP, and connect similarity entities
    # By default, GPUs will be used if available
    umap_g = graphistry.nodes(df).umap()
    umap_g.plot()

.. toctree::
   :maxdepth: 3

   graphistry

.. toctree::
   :hidden:

   graphistry.compute.predicates
   graphistry.plugins
   graphistry.plugins_types
   graphistry.validate


Articles
==================
* `Graphistry: Visual Graph AI Interactive demo <https://github.com/graphistry/pygraphistry/blob/master/README.md#demo-of-friendship-communities-on-facebook/>`_
* `PyGraphistry + Databricks <https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb>`_
* `PyGraphistry + UMAP <https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/umap_learn/umap_learn.ipynb>`_


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

