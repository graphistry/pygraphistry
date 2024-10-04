10 Minutes to PyGraphistry Visualization
========================================

Welcome to **PyGraphistry**, the platform for graph visualization, analytics, and AI. By the end of this guide, you'll be able to create interactive, GPU-accelerated graph visualizations of your data. If you are already familiar with ideas like dataframes, PyGraphistry will be an easy fit.

Why Graph Visualization?
------------------------

Graphs represent relationships between entities. Whether you're analyzing event logs, social media, security alerts, financial transactions, clickstreams, supply chains, or genomics, visualizing these relationships can reveal patterns and insights that are difficult to detect otherwise.

**Graph visualization helps you:**

- **Identify Patterns**: Spot clusters, behaviors, progressions, root causes, hubs, and anomalies.
- **Understand Structures**: See how entities are connected and how information flows.
- **Communicate Insights**: Present complex relationships in an understandable way.

As datasets grow larger, traditional tools struggle with performance and complexity, making it challenging for analysts to extract meaningful insights efficiently.

What Makes PyGraphistry Special?
--------------------------------

**PyGraphistry** is a comprehensive Python library that simplifies working with larger graphs by leveraging GPU acceleration. It is most known for:

- **GPU Acceleration**: Enables smooth interaction with large datasets, supporting visualization of 10-100X more data than other tools.
- **Advanced Visualization**: Provides rich visual encodings (e.g., color, size, icon, badges), interactive features (e.g., zooming, cross-filtering, drilldowns, timebars), and multiple layout algorithms.
- **Seamless Integration**: Works seamlessly with popular Python data science libraries like Pandas, cuDF, and NetworkX, and integrates easily into Jupyter notebooks for interactive data exploration.
- **Full Analytics Ecosystem**: Offers a native GFQL engine for graph queries and tools like visual UMAP clustering, allowing you to perform accelerated graph ETL, analytics, ML/AI, and visualization without needing a new database.

Installation
------------

Install PyGraphistry
~~~~~~~~~~~~~~~~~~~~~

::

    pip install graphistry

Install cuDF GPU dataframes (Optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For GPU acceleration with DataFrames, install **cuDF** via the `Nvidia RAPIDS Installation Guide <https://rapids.ai/>`_.

Register with PyGraphistry
~~~~~~~~~~~~~~~~~~~~~~~~~~

While most of PyGraphistry can run locally, the visualization server requires an account on your own self-hosted Graphistry server or on Graphistry Hub. If you do not have an account yet, make a free GPU account at `graphistry.com <https://www.graphistry.com/get-started>`_ , or launch your own server.

Then, in your Python environment, login with PyGraphistry:

.. code-block:: python

    import graphistry

    graphistry.register(api=3, server='hub.graphistry.com', username='YOUR_USERNAME', password='YOUR_PASSWORD')

Replace with your actual credentials.

---

Loading Data Efficiently
------------------------

The Python data science ecosystem supports loading almost any kind of data. Many users start with CSV, JSON, SQL, etc.

Loading Data as Parquet or Arrow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We often see teams adopt formats like **Parquet** and **Apache Arrow** as they are optimized for performance, interoperability, and reliability. Loading data with them can often be 10X+ faster.

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

Let's create a simple graph visualization using the honeypot data.

Step 1: Prepare the Data
~~~~~~~~~~~~~~~~~~~~~~~~

We'll create an edge list where each edge represents an attack from an attacker IP to a victim IP.

.. code-block:: python

    # Create the edge list
    edges = df[['attackerIP', 'victimIP', 'count']].rename(columns={
        'attackerIP': 'src',
        'victimIP': 'dst',
        'count': 'edge_count'
    })

Step 2: Plot the Graph
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Plot the graph
    g = graphistry.edges(edges, 'attackerIP', 'victimIP')
    g.plot()  # Make sure you called graphsitry.register() above

This will render an interactive graph where nodes represent IP addresses, and edges represent attacks.

Automatic GPU Acceleration
--------------------------

Note that the `plot()` step uploads the data to the Graphistry server for your server-GPU-accelerated visualization session. This results in smoother interactions and faster rendering, even with large datasets. 

Other times, PyGraphistry computes over data locally, such as with GFQL queries.  GPU acceleration will be automatically used if your enivornment supports GPU compute.

Adding Visual Encodings
-----------------------

PyGraphistry supports various visual encodings to represent different attributes in your data. You can encode attributes using color, size, icon, and badges.

Adding Color Encodings
----------------------

Let's add color encodings based on the vulnerability exploited.

.. code-block:: python

    # Plot with color encoding
    g2 = g.encode_edge_color('vulnName', categorical_mapping={
        'HTTP Vulnerability': 'red',
        'IIS Vulnerability': 'blue',
    }, default_mapping='gray')
    
    g2.plot()

Now, edges are colored based on the type of vulnerability, helping you distinguish different attack types.

Adjusting Sizes, Icons, Badges, and More
-----------------------------------------

You can adjust further node and edge settings using data. Sample calls include:

* `encode_point_size()`: Adjust node sizes based on a column.
* `encode_point_icon()`: Assign different icons to nodes based on a column.
* `encode_point_badge()`: Add badges to nodes based on a column.
* `encode_point_weight()`: Adjust node weights based on a column.
* Equivalent for edges: `encode_edge_size()`, `encode_edge_icon()`, `encode_edge_badge()`.


Adding a Timebar
----------------

If your data includes temporal information, you can add a timebar to visualize changes over time.

.. code-block:: python

    # Convert timestamps to datetime
    edges['time'] = cudf.to_datetime(df['time(max)'], unit='s')
    g3 = graphistry.edges(edges)

    # Plot with time encoding: Graphistry automatically detected arrow/parquet native types
    g3.plot()

The timebar allows you to interactively explore the graph as it evolves over time.


Applying Force-Directed Layout
------------------------------

By default, PyGraphistry uses a force-directed layout. You can adjust its parameters:

.. code-block:: python

    # Adjust layout settings
    g4 = g.settings(url_params={'play': 7000, 'strongGravity': True, 'edgeInfluence': 2})
    g4.plot()

More Layout Algorithms
-----------------------

PyGraphistry offers many layout algorithms and settings to help you display your graph meaningfully.

For example, graphviz layouts can be used for laying out small trees and directed acyclic graphs (DAGs).

.. code-block:: python

    g5 = g.layout_graphviz('dot')
    g5.plot()

Using UMAP for Dimensionality Reduction
---------------------------------------

For large graphs, you can use UMAP for dimensionality reduction to layout the graph meaningfully. UMAP will identify nodes that are similar across their different attributes and connect them into a similarity graph. 

.. code-block:: python

    # Compute UMAP layout by clustering on some subset of columns
    g6 = plot.umap(X=['attackerIP', 'victimIP', 'vulnName'])
    g6.plot()

Utilizing Hypergraphs
----------------------

PyGraphistry supports hypergraphs, which allow you to visualize complex relationships involving more than two entities.

**Example: Visualizing Attacks as Hyperedges**

.. code-block:: python

    # Generate the hypergraph
    hg = graphistry.hypergraph(df, ['attackerIP', 'victimIP', 'vulnName', 'victimPort'])

    # Plot the hypergraph
    hg['graph'].plot()

This will represent each attack as a hyperedge connecting the attacker IP, victim IP, vulnerability name, and port nodes.

Embedding Visualizations into Web Apps
---------------------------------------

You can embed PyGraphistry visualizations in web applications using additional SDKs like **GraphistryJS**.

The JavaScript client comes in 2 forms, and provide further configuration hooks:

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

- **Graph Queries with GFQL**: Use GFQL to query and manipulate your graph data before visualization.
- **Computational Enrichments**: Integrate graph algorithms like PageRank or community detection to enrich your data.
- **Connectors**: Leverage connectors to import data from various sources like databases, APIs, or logs.
- **Data Loading Best Practices**: Utilize Parquet or Arrow formats for efficient data loading.
- **Explore Layouts and Encodings**: Experiment with different layouts and visual encodings to gain deeper insights.

Resources:

- **GFQL Documentation**: Learn how to perform advanced graph queries.
- **PyGraphistry API Reference**: Explore the full capabilities of PyGraphistry.
- **Graphistry Connectors**: Discover how to load data from different sources.
- **GraphistryJS Documentation**: Learn how to embed visualizations in web applications.

Happy graphing!
