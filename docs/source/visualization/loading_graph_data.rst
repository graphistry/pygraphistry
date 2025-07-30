.. _loading-graph-data:

Loading Graph Data
==================

PyGraphistry represents graphs using **graph DataFrames** - a pair of DataFrames for nodes and edges. This approach enables seamless integration with the Python data ecosystem while supporting both CPU (pandas) and GPU (cuDF) acceleration.

A graph in PyGraphistry consists of:

- **Nodes DataFrame**: Each row represents a node with its properties
- **Edges DataFrame**: Each row represents an edge with source, destination, and properties

This guide shows how to load your data into PyGraphistry from various sources.

Minimal Graphs (Edges Only)
---------------------------

The simplest way to create a graph requires only an edges DataFrame:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Minimal edge list
    edges = pd.DataFrame({
        'source': ['alice', 'bob', 'carol'],
        'destination': ['bob', 'carol', 'alice']
    })

    # Create graph - nodes are inferred automatically
    g = graphistry.edges(edges, 'source', 'destination')
    g.plot()

    # From CSV
    df = pd.read_csv('edges.csv')
    g = graphistry.edges(df, 'src', 'dst')
    
    # From Parquet
    df = pd.read_parquet('edges.parquet')
    g = graphistry.edges(df, 'from_id', 'to_id')

For more on pandas file formats, see `pandas I/O documentation <https://pandas.pydata.org/docs/user_guide/io.html>`_.

Binding Nodes and Edges
-----------------------

When you have both node and edge data, bind them together for richer visualizations:

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

    # Create graph with both nodes and edges
    g = graphistry.nodes(nodes, 'id').edges(edges, 'source', 'target')
    
    # Alternative: bind first, then add data
    g = graphistry.bind(source='source', destination='target', node='id')
    g = g.nodes(nodes).edges(edges)
    
    # Nodes can have attributes for visual encoding
    g = g.encode_point_color('type').encode_point_size('age')
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

Hypergraphs
-----------

Hypergraphs transform tabular data into graphs by connecting entities that appear in the same row:

.. code-block:: python

    # Transform table where rows represent events with multiple participants
    events = pd.DataFrame({
        'user': ['alice', 'bob', 'alice'],
        'product': ['laptop', 'laptop', 'phone'],
        'store': ['online', 'online', 'retail'],
        'amount': [1000, 1200, 800]
    })

    # Create hypergraph - connects values from same row
    hg = graphistry.hypergraph(events, 
        entity_types=['user', 'product', 'store'])
    g = hg['graph']
    g.plot()

    # Direct mode - connects entities without event nodes
    hg = graphistry.hypergraph(events, direct=True)
    g = hg['graph']
    g.plot()

For more details, see :ref:`hyper-api`.

GPU vs CPU DataFrames
---------------------

PyGraphistry automatically detects and uses the appropriate engine:

.. code-block:: python

    import cudf
    import graphistry

    # CPU (pandas) - default
    edges_df = pd.read_csv('edges.csv')
    g_cpu = graphistry.edges(edges_df, 'src', 'dst')

    # GPU (cuDF) - automatic when using cuDF
    edges_gdf = cudf.read_csv('edges.csv') 
    g_gpu = graphistry.edges(edges_gdf, 'src', 'dst')

    # Convert between CPU and GPU
    edges_gdf = cudf.from_pandas(edges_df)
    edges_df = edges_gdf.to_pandas()

**When to use GPU (cuDF)**:

- Large graphs (millions of edges)
- Complex graph algorithms
- When GPU memory is available
- Real-time streaming analysis

**When to use CPU (pandas)**:

- Small to medium graphs
- When GPU is not available
- Integration with CPU-only libraries
- Development and prototyping

For performance comparisons, see `RAPIDS benchmarks <https://rapids.ai/benchmarks/>`_.

Import from Other Systems
-------------------------

PyGraphistry provides adapters for popular graph libraries and databases:

**NetworkX**

.. code-block:: python

    import networkx as nx
    import graphistry

    # Create NetworkX graph
    G = nx.karate_club_graph()
    
    # Convert to PyGraphistry
    g = graphistry.from_networkx(G)
    g.plot()

See :ref:`networkx-plugin` for details.

**igraph**

.. code-block:: python

    # Use igraph algorithms on PyGraphistry graphs
    g = graphistry.edges(df, 'src', 'dst')
    g2 = g.compute_igraph('pagerank')
    g2.plot()

See :doc:`../plugins` for available algorithms.

**Graph Databases**

.. code-block:: python

    # Neo4j
    g = graphistry.bolt(driver)
    g2 = g.cypher("MATCH (n)-[r]->(m) RETURN n, r, m")
    
    # Amazon Neptune  
    g = graphistry.neptune(endpoint)
    g2 = g.gremlin("g.E()")

See database-specific documentation:

- Neo4j/Bolt: :ref:`bolt-notebook`
- Amazon Neptune: :ref:`neptune-notebook`
- TigerGraph: :ref:`tigergraph-notebook`
- Additional connectors in :doc:`../plugins`

**Spark**

For PySpark DataFrames, convert to pandas:

.. code-block:: python

    # From PySpark DataFrame
    spark_df = spark.read.parquet("hdfs://data.parquet")
    pandas_df = spark_df.toPandas()
    g = graphistry.edges(pandas_df, 'src', 'dst')

Alternative Constructors
------------------------

- **Remote datasets**: Bind to existing server data using ``graphistry.bind(dataset_id='...')``
- **Arrow tables**: Direct support for PyArrow tables

Export Graph Data
-----------------

Access and export your graph data for further processing:

.. code-block:: python

    # Access DataFrames
    nodes_df = g._nodes
    edges_df = g._edges

    # Select specific columns
    edge_list = g._edges[[g._source, g._destination]]
    
    # Add computed properties
    g2 = g.compute_igraph('pagerank')
    ranked_nodes = g2._nodes[['node_id', 'pagerank']]

    # Export to files
    g._nodes.to_csv('nodes_processed.csv', index=False)
    g._edges.to_parquet('edges_processed.parquet')
    
    # Export to other formats
    nodes_json = g._nodes.to_json(orient='records')
    edges_dict = g._edges.to_dict(orient='records')

    # Pipeline example
    enriched = (g
        .compute_igraph('pagerank')
        .compute_igraph('community')
        ._nodes
        .query('pagerank > 0.02')
        .to_csv('influential_nodes.csv'))

Next Steps
----------

- Explore graph visualization in :ref:`10min-viz`
- Learn about :ref:`layout-guide` options
- Query your graph with :ref:`gfql-index`
- Deep dive into the :ref:`plotter-api` reference
