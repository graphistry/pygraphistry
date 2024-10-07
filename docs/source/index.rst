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

* **`Graphistry Visualization Client <10min-viz>`_**: Convenience layer for using the optional Graphistry GPU server for accelerated compute and visualization
* **Dataframe-native graph manipulation**: Optimized dataframe-native tabular and graph methods for loading, transforming, analyzing, and visualizing data as graphs
* **`GFQL (new!) <10min-gfql>`_ queries**: Home for the GFQL graph dataframe-native query language, and with optional GPU support
* **Graphistry[AI]**: Optional methods and integrations for graph autoML, including automatic feature engineering, UMAP, and graph neural networks
* **`Plugins <plugins>`_**: Optimized and streamlined integrations for enriching your workflows - query databases like Neo4j and Splunk, compute systems like Nvidia RAPIDS, and enrich data with library calls to graph engines like igraph
* **`Louie.AI (new!) <https://www.louie.ai>`_**: Use generative AI to talk to your data, including for GFQL queries and Graphistry visualizations

Combined, PyGraphistry reduces your time to graph for going from raw data to visualizations to AI insights in a few lines of code.

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

1. Install:

.. code-block:: bash

    # Install from PyPI
    pip install graphistry

    # Optionally, get a free GPU account or self-hosted server at https://graphistry.com/get-started

2. Start graphing!

.. code-block:: python

    import graphistry
    import pandas as pd

    # Load data from any Python data science library or database
    df = pd.DataFrame({
      'src': ['A', 'B', 'C'],
      'dst': ['B', 'C', 'A'],
      'nfo': ['X', 'Y', 'Z']
      'abc': [1, 2, 3]
    })
    g1 = graphistry.edges(df, source="src", destination="dst")

    # Server-accelerated GPU visualization
    graphistry.register(api=3, server="hub.graphistry.com", username="A", password="B")
    g1.plot()

    # Use GPUs when available in almost all APIs
    import cudf
    g2 = g1.edges(cudf.from_pandas(g1._edges))
    g2.plot()

    # Many local graph wrangling helpers and easy dataframe-native graph manipulation
    g3 = g2.materialize_nodes().get_degrees()
    print(g3._nodes[['id', 'degree']])

    # ML & AI methods
    umap_g = graphistry.nodes(df).umap()
    umap_g.plot()

    # GFQL graph dataframe-native query language with optional GPU support
    nearest_neighbors_df = umap_g.chain([ n({'id': 'A'}), e(hops=2), n()])._nodes


Articles
==================

We recommend reading the Graphistry blog and github demos. Some useful articles include:

* `Graphistry: Visual Graph AI Interactive demo <https://github.com/graphistry/pygraphistry/blob/master/README.md#demo-of-friendship-communities-on-facebook/>`_
* `PyGraphistry + Databricks <https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/databricks_pyspark/graphistry-notebook-dashboard.ipynb>`_
* `PyGraphistry + UMAP <https://github.com/graphistry/pygraphistry/blob/master/demos/demos_databases_apis/umap_learn/umap_learn.ipynb>`_
* `What is Graph Intelligence? <https://gradientflow.com/what-is-graph-intelligence/>`_


Indices and tables
==================

.. toctree::
   :maxdepth: 3
   :hidden:

   graphistry

.. toctree::
   :maxdepth: 3

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`