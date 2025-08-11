.. _10min:

10 Minutes to PyGraphistry
==========================

Welcome to **PyGraphistry**, the fast and easy platform for graph visualization, querying, analytics, and AI. By the end of this guide, you'll be able to create interactive, GPU-accelerated graph visualizations of your data. If you are already familiar with concepts like dataframes, PyGraphistry will be an easy fit.

PyGraphistry can be used standalone and automatically optimizes for both CPU systems and GPU systems. It is typically used in Python notebooks, dashboards, and web apps. The library includes the GFQL dataframe-native graph query language, official Python bindings for Graphistry GPU visualization & analytics servers, and a variety of graph data science tools.

Why Graph Intelligence?
------------------------

Graphs represent relationships between entities. Whether you're analyzing event logs, social media interactions, security alerts, financial transactions, clickstreams, supply chains, or genomics data, visualizing and analyzing these relationships can reveal patterns and insights that are difficult to detect otherwise.

**Graph visualization and analytics helps you:**

- **Identify Patterns**: Spot clusters, behaviors, progressions, root causes, hubs, and anomalies.
- **Understand Structures**: See how entities are connected and how information flows.
- **Communicate Insights**: Present complex relationships in an understandable way.

As datasets grow larger, traditional tools struggle with performance and complexity, making it challenging for analysts to extract meaningful insights efficiently.

What Makes PyGraphistry Special?
--------------------------------

**PyGraphistry** is a comprehensive Python library that simplifies working with larger graphs. It is known for:

- **GPU Acceleration**: Work with larger datasets visually or programatically
- **Advanced Visualization**: Rich out-of-the-box visual encodings (e.g., color, size, icon, badges), interactive analysis features (e.g., zooming, cross-filtering, drilldowns, timebars), multiple layout algorithms.
- **Seamless Integration**: Works seamlessly with popular Python data science libraries like Pandas, cuDF, and NetworkX, and integrates easily into notebooks, dashboard tools, web apps, databases, and other tools
- **GFQL dataframe-native graph query language**: Run graph queries and analytics directly on dataframes, with optional GPU acceleration, which gives scalable results without the usual infrastructure overhead.
- **Graphistry[AI]**: With native support for GPU feature engineering, UMAP clustering, and embeddings, quickly perform accelerated graph ETL, analytics, ML/AI, and visualization on large datasets.
- **Multiple Interfaces**: In addition to the PyGraphistry Python bindings, Graphistry provides REST APIs, Node.js and React libraries, and **Louie.AI** for conversational analytics, making it accessible from various platforms and languages.

Installation
------------

Install PyGraphistry
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install graphistry

This performs a minimal installation with dependencies limited to mostly just Pandas and PyArrow.

Install cuDF GPU DataFrames (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration with DataFrames, install **cuDF** via the `NVIDIA RAPIDS Installation Guide <https://rapids.ai/>`_.

Register with PyGraphistry (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

While most of PyGraphistry can run locally, use with a GPU visualization server requires an account on your own self-hosted Graphistry server or on Graphistry Hub. If you do not have an account yet, create a free GPU account at `graphistry.com <https://www.graphistry.com/get-started>`_, or launch your own server.

Then, log in your PyGraphistry client:

.. code-block:: python

    import graphistry

    graphistry.register(api=3, server='hub.graphistry.com', username='YOUR_USERNAME', password='YOUR_PASSWORD')

Replace with your actual server and credentials.

Loading Data Efficiently
------------------------

The Python data science ecosystem supports connecting to most databases and file type types

Many users start with CSV, JSON, and SQL database. We often see teams adopt formats like **Parquet** and **Apache Arrow**. Graphistry natively leverages these, so loading data with them can often be 10X+ faster than typical libraries.

**Example: Loading Parquet Data**

.. code-block:: python

    import cudf
    import graphistry

    # Load the dataset using cuDF
    df = cudf.read_parquet('data/honeypot.parquet')

    print(df.head())

Alternatively, if you don't have a GPU or cuDF, you can use Pandas:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Load the dataset using Pandas
    df = pd.read_csv('https://raw.githubusercontent.com/graphistry/pygraphistry/master/demos/data/honeypot.csv')

    print(df.head())

**Sample Data Structure:**

::

    attackerIP       victimIP  victimPort         vulnName  count   time(max)   time(min)
    0   1.235.32.141  172.31.14.66       139.0  MS08067 (NetAPI)      6  1421433577  1421422669
    1  105.157.235.22  172.31.14.66       445.0  MS08067 (NetAPI)      4  1422497735  1422494755
    ...

Creating a Basic Visualization
------------------------------

Let's create a simple graph visualization using the honeypot data:

.. code-block:: python

    g = graphistry.edges(df, 'attackerIP', 'victimIP')
    g.plot()  # Make sure you called graphistry.register() above

This will render an interactive graph where nodes represent IP addresses, and edges represent attacks.

Automatic GPU Acceleration
--------------------------

Note that the ``plot()`` step uploads the data to the Graphistry server for your server-GPU-accelerated visualization session. This results in smoother interactions and faster rendering, even with large datasets.

Other times, PyGraphistry computes over data locally, such as with GFQL queries. GPU acceleration will be automatically used if your environment supports GPU compute.

Adding Visual Encodings
-----------------------

PyGraphistry supports various visual encodings to represent different attributes in your data.


Example: Adding Color Encodings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's add color encodings based on the vulnerability exploited.

.. code-block:: python

    # Plot with color encoding
    g2 = g1.encode_edge_color(
        'vulnName',
        categorical_mapping={
            'MS08067 (NetAPI)': 'red',
            'OtherVuln': 'blue',
        },
        default_mapping='gray')

    g2.plot()

Now, edges are colored based on the type of vulnerability, helping you distinguish different attack types.

Adjusting Sizes, Labels, Icons, Badges, and More
------------------------------------------------

You can adjust further node and edge settings using data. Sample calls include:

- ``bind(point_title=)``: Assign labels to nodes based on a column
- ``encode_point_size()``: Adjust node sizes based on a column
- ``encode_point_icon()``: Assign different icons to nodes based on a column
- ``encode_point_badge()``: Add badges to nodes based on a column
- ``encode_point_weight()``: Adjust node weights based on a column
- Equivalent functions for edges: ``encode_edge_size()``, ``encode_edge_icon()``, ``encode_edge_badge()``

Additional settings, such as background colors and logo watermarks, can also be configured.


Adding an Interactive Timebar
-----------------------------

If your data includes temporal information, you can add a timebar to visualize changes over time.

.. code-block:: python

    # Ensure column has a datetime dtype
    edges['time'] = cudf.to_datetime(df['time(max)'], unit='s')
    g = graphistry.edges(edges)

    # Plot with time encoding: Graphistry automatically detects Arrow/Parquet native types
    g.plot()

The timebar appears as soon as the UI detects datetime values, and enables you to interactively explore the graph as it evolves over time.


Applying Force-Directed Layout
------------------------------

By default, PyGraphistry uses a force-directed layout. You can adjust its parameters:

.. code-block:: python

    # Adjust layout settings
    g2 = g1.settings(url_params={'play': 7000, 'strongGravity': True, 'edgeInfluence': 2})
    g2.plot()

More Layout Algorithms
----------------------

PyGraphistry offers additional layout algorithms of its own, and streamlines using layouts from other libraries, so you can display your graph quickly and meaningfully.

For example, GraphViz layouts is known for its high quality for laying out small trees and directed acyclic graphs (DAGs):

.. code-block:: python

    # pygraphistry handles format conversions behind-the-scenes
    g2 = g1.layout_graphviz('dot')
    g2.plot()

Using UMAP for Dimensionality Reduction
---------------------------------------

For large datasets, you can use UMAP for dimensionality reduction to layout the graph meaningfully. UMAP will identify nodes that are similar across their different attributes.

Special to PyGraphistry, PyGraphistry records and renders the similarity edges between similar entities. We find this to be critical in practice for investigating results and using UMAP in analytical pipelines.

.. code-block:: python

    # Compute UMAP layout by clustering on some subset of columns
    g1 = graphistry.umap(X=['attackerIP', 'victimIP', 'vulnName'])
    print('# similarity edges', len(g1._edges))
    g1.plot()


Query graphs with GFQL
----------------------------------

GFQL, our dataframe-native graph query language, allows you to run optimized graph queries directly on dataframes without the need for a separate graph database system.

Suppose you want to focus on attacks that started with the "MS08067 (NetAPI)" vulnerability at some specific timestamp, and see everything 2 hops after:

.. code-block:: python

    g2 = g1.gfql([
        n(),
        e(edge_query="vulnName == 'MS08067 (NetAPI)' & `time(max)` > 1421430000"),
        n(),
        e(hops=2)
    ])

    g2.plot()

This GFQL query filters the edges based on the vulnerability name and time, then returns the matching nodes and edges for visualization.

Sequencing Programs with Let
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more complex analyses, GFQL's ``let`` feature allows you to create named, reusable graph patterns that can reference each other. This is particularly powerful for multi-step graph algorithms and investigations.

**Example: Finding and visualizing influence zones of high-value nodes**

.. code-block:: python

    from graphistry import let, n, e, ref, call
    
    g_analysis = let('ranked', call('compute_cugraph', {'alg': 'pagerank', 'out_col': 'pagerank'})) \
        .let('influencers', ref('ranked', [n(query='pagerank > 0.02')])) \
        .let('contacts', ref('influencers', [e(), n()])) \
        .let('influence_zone', ref('contacts', [e(), n()])) \
        .run(g1)

    # Visualize the influence zones with PageRank-based sizing
    g_analysis.encode_point_size('pagerank').plot()

This example demonstrates how ``let`` enables you to:

1. **Sequence operations**: Each step builds on previous results
2. **Reuse computations**: Reference earlier results with ``ref()``
3. **Name intermediate results**: Makes complex queries readable
4. **Compose algorithms**: Combine graph algorithms with traversals


Utilizing Hypergraphs
---------------------

PyGraphistry supports hypergraphs, which allow you to quickly visualize complex relationships involving more than two entities.

**Example: Visualizing Attacks as Hyperedges**

.. code-block:: python

    hg = graphistry.hypergraph(df, ['attackerIP', 'victimIP', 'vulnName', 'victimPort'])

    hg['graph'].plot()

This will represent each attack as a hyperedge connecting the attacker IP, victim IP, vulnerability name, and port nodes.

Embedding Visualizations into Web Apps
--------------------------------------

You can embed PyGraphistry visualizations in web applications using additional SDKs like **GraphistryJS**.

The JavaScript client comes in two forms and provides further configuration hooks:

- **Vanilla JavaScript**: Use the GraphistryJS library to embed visualizations directly.
- **React**: Use the Graphistry React components for seamless integration.

Rendering Options
-----------------

Inline Rendering
~~~~~~~~~~~~~~~~

In Jupyter notebooks, you can render the visualization inline.

.. code-block:: python

    g.plot()

URL Rendering
~~~~~~~~~~~~~

Alternatively, you can generate a URL to view the visualization in a separate browser tab.

.. code-block:: python

    url = g.plot(render=False)
    print(f"View your visualization at: {url}")

Next Steps
----------

- :ref:`10 Minutes to Graphistry Visualization <10min-viz>`: Learn how to create more advanced visualizations.
- :ref:`10 Minutes to GFQL <10min-gfql>`: Use GFQL to query and manipulate your graph data before visualization.
- :ref:`Layout guide <layout-guide>`: Explore different layouts for your visualizations.
- :ref:`Plugins <plugins>`: Discover more ways to connect to your data and work with your favorite tools. 
- :ref:`PyGraphistry API Reference <api>`

External Resources
------------------
- `Graphistry UI Guide <https://hub.graphistry.com/docs/ui/index/>`_
- `GraphistryJS <https://github.com/graphistry/graphistry-js>`_: Node, React, and vanilla JS clients
- `Graphistry REST API <https://hub.graphistry.com/docs/api/>`_: Work from any language
- `Graphistry URL settings <https://hub.graphistry.com/docs/api/1/rest/url/#urloptions>`_: Control visualizations via URL parameters`

Happy graphing!