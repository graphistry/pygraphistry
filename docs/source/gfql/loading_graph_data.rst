.. _loading-graph-data:

Loading Graph Data
==================

This guide covers various methods to load graph data into PyGraphistry for use with GFQL queries. Whether you're working with CSV files, databases, or existing graph formats, PyGraphistry provides flexible options for data ingestion.

Basic Graph Creation
--------------------

Creating from DataFrames
~~~~~~~~~~~~~~~~~~~~~~~~~

The most common way to create a graph is from edge and node DataFrames:

.. code-block:: python

    import pandas as pd
    import graphistry

    # Create edge DataFrame
    edges_df = pd.DataFrame({
        'source': ['Alice', 'Bob', 'Charlie', 'Alice'],
        'destination': ['Bob', 'Charlie', 'Alice', 'Charlie'],
        'weight': [1.0, 2.0, 3.0, 4.0],
        'type': ['friend', 'colleague', 'friend', 'colleague']
    })

    # Create node DataFrame (optional)
    nodes_df = pd.DataFrame({
        'id': ['Alice', 'Bob', 'Charlie'],
        'age': [30, 25, 35],
        'department': ['Sales', 'Engineering', 'Marketing']
    })

    # Create graph
    g = graphistry.edges(edges_df, source='source', destination='destination')
    g = g.nodes(nodes_df, node='id')

Column Name Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

You can use any column names for your graph structure:

.. code-block:: python

    # Custom column names
    edges_df = pd.DataFrame({
        'from_user': [...],
        'to_user': [...],
        'interaction_type': [...]
    })

    g = graphistry.edges(edges_df, source='from_user', destination='to_user')

Loading from Files
------------------

CSV Files
~~~~~~~~~

.. code-block:: python

    # Load edges from CSV
    edges_df = pd.read_csv('edges.csv')
    nodes_df = pd.read_csv('nodes.csv')

    g = graphistry.edges(edges_df, 'source', 'target')
    g = g.nodes(nodes_df, 'node_id')

    # Direct loading with custom parsing
    edges_df = pd.read_csv('transactions.csv',
                          parse_dates=['timestamp'],
                          dtype={'amount': float})

    g = graphistry.edges(edges_df, 'sender', 'receiver')

Parquet Files
~~~~~~~~~~~~~

For larger datasets, Parquet files offer better performance:

.. code-block:: python

    # CPU loading
    edges_df = pd.read_parquet('edges.parquet')
    nodes_df = pd.read_parquet('nodes.parquet')

    # GPU loading with cuDF
    import cudf
    edges_gdf = cudf.read_parquet('edges.parquet')
    nodes_gdf = cudf.read_parquet('nodes.parquet')

    g = graphistry.edges(edges_gdf, 'src', 'dst').nodes(nodes_gdf, 'id')

JSON Files
~~~~~~~~~~

.. code-block:: python

    import json

    # Load JSON data
    with open('graph_data.json', 'r') as f:
        data = json.load(f)

    # Convert to DataFrames
    edges_df = pd.DataFrame(data['edges'])
    nodes_df = pd.DataFrame(data['nodes'])

    g = graphistry.edges(edges_df, 'source', 'target').nodes(nodes_df, 'id')

Loading from Databases
----------------------

SQL Databases
~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from sqlalchemy import create_engine

    # Create database connection
    engine = create_engine('postgresql://user:password@host:port/database')

    # Load edges from SQL query
    edge_query = """
        SELECT user_id as source,
               friend_id as destination,
               created_at,
               relationship_type
        FROM friendships
        WHERE created_at > '2023-01-01'
    """
    edges_df = pd.read_sql(edge_query, engine)

    # Load nodes
    node_query = """
        SELECT user_id as id,
               username,
               created_date,
               account_type
        FROM users
    """
    nodes_df = pd.read_sql(node_query, engine)

    g = graphistry.edges(edges_df, 'source', 'destination')
    g = g.nodes(nodes_df, 'id')

NoSQL Databases
~~~~~~~~~~~~~~~

.. code-block:: python

    from pymongo import MongoClient

    # MongoDB example
    client = MongoClient('mongodb://localhost:27017/')
    db = client['graph_database']

    # Load edges
    edges = list(db.edges.find())
    edges_df = pd.DataFrame(edges)

    # Load nodes
    nodes = list(db.nodes.find())
    nodes_df = pd.DataFrame(nodes)

    g = graphistry.edges(edges_df, 'from', 'to').nodes(nodes_df, '_id')

Graph Database Export
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Neo4j export example
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver("neo4j://localhost:7687",
                                 auth=("neo4j", "password"))

    with driver.session() as session:
        # Export edges
        result = session.run("""
            MATCH (n)-[r]->(m)
            RETURN id(n) as source, id(m) as target, type(r) as rel_type
        """)
        edges_df = pd.DataFrame([r.data() for r in result])

        # Export nodes
        result = session.run("""
            MATCH (n)
            RETURN id(n) as id, labels(n) as labels, properties(n) as props
        """)
        nodes_df = pd.DataFrame([r.data() for r in result])

    g = graphistry.edges(edges_df, 'source', 'target').nodes(nodes_df, 'id')

Working with Large Datasets
----------------------------

Chunked Loading
~~~~~~~~~~~~~~~

For very large files, load data in chunks:

.. code-block:: python

    # Process large CSV in chunks
    chunk_size = 100000
    chunks = []

    for chunk in pd.read_csv('large_edges.csv', chunksize=chunk_size):
        # Process/filter each chunk
        filtered = chunk[chunk['weight'] > 0.5]
        chunks.append(filtered)

    edges_df = pd.concat(chunks, ignore_index=True)
    g = graphistry.edges(edges_df, 'source', 'target')

GPU Memory Management
~~~~~~~~~~~~~~~~~~~~~

When using GPU acceleration:

.. code-block:: python

    import cudf

    # Monitor GPU memory
    print(f"GPU memory before: {cudf.cuda.cuda.get_memory_info()}")

    # Load data
    edges_gdf = cudf.read_parquet('edges.parquet')

    # Sample if needed
    if len(edges_gdf) > 10_000_000:
        edges_gdf = edges_gdf.sample(n=10_000_000)

    g = graphistry.edges(edges_gdf, 'src', 'dst')

    print(f"GPU memory after: {cudf.cuda.cuda.get_memory_info()}")

Data Preprocessing
------------------

Type Conversion
~~~~~~~~~~~~~~~

Ensure proper data types for optimal performance:

.. code-block:: python

    # Convert types
    edges_df['weight'] = edges_df['weight'].astype(float)
    edges_df['timestamp'] = pd.to_datetime(edges_df['timestamp'])
    edges_df['category'] = edges_df['category'].astype('category')

    # For GPU, use appropriate types
    if using_gpu:
        edges_gdf['source'] = edges_gdf['source'].astype('int32')
        edges_gdf['destination'] = edges_gdf['destination'].astype('int32')

Handling Missing Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Fill missing values
    edges_df['weight'].fillna(1.0, inplace=True)
    nodes_df['label'].fillna('Unknown', inplace=True)

    # Drop rows with missing critical values
    edges_df = edges_df.dropna(subset=['source', 'destination'])

    # Infer missing nodes from edges
    g = graphistry.edges(edges_df, 'source', 'destination')
    # GFQL automatically infers nodes from edge endpoints

Data Validation
~~~~~~~~~~~~~~~

.. code-block:: python

    # Validate graph connectivity
    def validate_graph(edges_df, nodes_df=None):
        # Check for self-loops if needed
        self_loops = edges_df[edges_df['source'] == edges_df['destination']]
        print(f"Self-loops: {len(self_loops)}")

        # Check for duplicates
        duplicates = edges_df.duplicated(subset=['source', 'destination'])
        print(f"Duplicate edges: {duplicates.sum()}")

        # Verify node coverage
        if nodes_df is not None:
            edge_nodes = set(edges_df['source']) | set(edges_df['destination'])
            defined_nodes = set(nodes_df['id'])
            missing = edge_nodes - defined_nodes
            if missing:
                print(f"Warning: {len(missing)} nodes in edges but not in nodes table")

        return edges_df, nodes_df

    edges_df, nodes_df = validate_graph(edges_df, nodes_df)
    g = graphistry.edges(edges_df, 'source', 'destination').nodes(nodes_df, 'id')

Remote Data Loading
-------------------

Binding to Existing Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Bind to remote dataset
    g = graphistry.bind(dataset_id='my-dataset-id')

    # Query remote data
    result = g.gfql_remote([
        n({'type': 'person'}),
        e_forward()
    ])

Uploading Local Data
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Upload local graph to remote server
    g = graphistry.edges(edges_df, 'source', 'destination')
    g = g.nodes(nodes_df, 'id')

    # Upload and get dataset ID
    g_remote = g.upload()
    print(f"Dataset ID: {g_remote._dataset_id}")

    # Run remote queries
    result = g_remote.gfql_remote([...])

Best Practices
--------------

1. **Schema Consistency**: Ensure consistent column names and types across your data pipeline
2. **Memory Management**: For large datasets, consider sampling or chunking strategies
3. **Index Optimization**: Create appropriate indexes in your database for graph queries
4. **Data Quality**: Validate and clean data before loading to avoid runtime errors
5. **GPU vs CPU**: Choose the appropriate engine based on data size and available resources

Example: Complete Loading Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    import graphistry
    from graphistry import n, e_forward, gt

    def load_and_prepare_graph(edges_path, nodes_path, gpu=False):
        """Load and prepare graph data for GFQL analysis."""

        # Choose appropriate loader
        if gpu:
            import cudf
            edges_df = cudf.read_parquet(edges_path)
            nodes_df = cudf.read_parquet(nodes_path)
            engine = 'cudf'
        else:
            edges_df = pd.read_parquet(edges_path)
            nodes_df = pd.read_parquet(nodes_path)
            engine = 'pandas'

        # Data cleaning
        edges_df = edges_df.dropna(subset=['source', 'destination'])
        edges_df['weight'] = edges_df['weight'].fillna(1.0)

        # Create graph
        g = graphistry.edges(edges_df, 'source', 'destination')
        g = g.nodes(nodes_df, 'node_id')

        # Validate
        print(f"Loaded {len(g._edges)} edges and {len(g._nodes)} nodes")
        print(f"Using engine: {engine}")

        return g, engine

    # Use the pipeline
    g, engine = load_and_prepare_graph('edges.parquet', 'nodes.parquet', gpu=True)

    # Run GFQL query
    result = g.gfql([
        n({'risk_score': gt(75)}),
        e_forward(hops=2)
    ], engine=engine)

See Also
--------

- :ref:`gfql-quick` - Quick reference for GFQL operations
- :ref:`10min-gfql` - Tutorial on using GFQL with loaded data
- :doc:`../api/gfql/index` - API documentation for graph operations