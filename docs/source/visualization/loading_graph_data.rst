.. _loading-graph-data:

Loading Graph Data
==================

PyGraphistry represents graphs using **graph DataFrames** - a pair of DataFrames for nodes and edges. This approach enables seamless integration with the Python data ecosystem while supporting both CPU (pandas) and GPU (cuDF) acceleration.

A graph in PyGraphistry consists of:

- **Nodes DataFrame**: Each row represents a node with its properties
- **Edges DataFrame**: Each row represents an edge with source, destination, and properties

This guide shows how to load your data into PyGraphistry from various sources.

From DataFrames
---------------

The most direct way is to create pandas DataFrames and bind them:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Create sample DataFrames
    nodes = pd.DataFrame({
        'id': ['alice', 'bob', 'charlie'],
        'age': [25, 30, 35],
        'type': ['person', 'person', 'person']
    })

    edges = pd.DataFrame({
        'source': ['alice', 'bob', 'alice'],
        'target': ['bob', 'charlie', 'charlie'],
        'relationship': ['knows', 'knows', 'works_with']
    })

    # Create graph
    g = graphistry.nodes(nodes, 'id').edges(edges, 'source', 'target')
    
    # Visualize
    g.plot()

From CSV Files
--------------

Load graph data from CSV files:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Load nodes and edges from local CSV files
    nodes_df = pd.read_csv('nodes.csv')
    edges_df = pd.read_csv('edges.csv')

    g = graphistry.nodes(nodes_df, 'node_id').edges(edges_df, 'src', 'dst')
    g.plot()

From URLs
---------

Load data directly from URLs:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Example: Load honeypot data
    url = 'https://raw.githubusercontent.com/graphistry/pygraphistry/refs/heads/master/demos/data/honeypot.csv'
    df = pd.read_csv(url)

    # For data with edge list format (source, destination columns)
    g = graphistry.edges(df, 'attackerIP', 'victimIP')
    
    # Add edge attributes
    g = g.encode_edge_color('victimPort')
    g.plot()

GPU Acceleration with cuDF
--------------------------

For larger datasets, use GPU DataFrames:

.. code-block:: python

    import cudf
    import graphistry

    # Load data into GPU memory
    nodes_gdf = cudf.read_csv('large_nodes.csv')
    edges_gdf = cudf.read_csv('large_edges.csv')

    # PyGraphistry automatically handles cuDF DataFrames
    g = graphistry.nodes(nodes_gdf, 'id').edges(edges_gdf, 'src', 'dst')
    g.plot()

Alternative Constructors
------------------------

PyGraphistry offers specialized constructors for different data types:

- **Hypergraphs**: For many-to-many relationships - see :ref:`hyper-api`
- **Remote datasets**: Bind to existing server data using ``graphistry.bind(dataset_id='...')``
- **NetworkX**: Convert from NetworkX - see :ref:`networkx-plugin`
- **Graph databases**: Direct connectors for Neo4j, Neptune, and others

Next Steps
----------

- Explore graph visualization in :ref:`10min-viz`
- Learn about :ref:`layout-guide` options
- Query your graph with :ref:`gfql-index`
- Deep dive into the :ref:`plotter-api` reference
