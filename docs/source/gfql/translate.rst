.. _gfql-translate:

Translating Between SQL, Pandas, Cypher, and GFQL
=================================================

This guide provides a comparison between **SQL**, **Pandas**, **Cypher**, and **GFQL**, helping you translate familiar queries into GFQL.

Introduction
------------

GFQL (GraphFrame Query Language) is designed to be intuitive for users familiar with SQL, Cypher, or dataframe like Pandas and Spark. By comparing equivalent queries across these languages, you can quickly grasp GFQL's syntax, benefits, and start utilizing its powerful graph querying capabilities within your workflows.

Who Is This Guide For?
----------------------

- **Data Scientists:** Familiar with Pandas or SQL, exploring graph relationships.
- **Engineers:** Integrating graph queries into applications.
- **DBAs:** Understanding how GFQL complements SQL for graph data.
- **Graph Specialists:** Experienced with Cypher, integrating graph queries into Python.

Common Graph and Query Tasks
----------------------------

We'll cover a range of common graph and query tasks:

.. contents::
   :depth: 2
   :local:

Ex: Finding Nodes with Specific Properties
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all nodes where the `type` is `"person"`.

**SQL**

.. code-block:: sql

    SELECT * FROM nodes
    WHERE type = 'person';

**Pandas**

.. code-block:: python

    people_nodes_df = nodes_df[ nodes_df['type'] == 'person' ]

**Cypher**

.. code-block:: cypher

    MATCH (n {type: 'person'})
    RETURN n;

**GFQL**

.. code-block:: python

    from graphistry import n

    # df[['id', 'type', ...]]
    g.chain([ n({"type": "person"}) ])._nodes

**Explanation**:

- **GFQL**: `n({"type": "person"})` filters nodes where `type` is `"person"`. `g.chain([...])` applies this filter to the graph `g`, and `._nodes` retrieves the resulting nodes. The performance is similar to that of Pandas (CPU) or cuDF (GPU).

---

Ex: Exploring Relationships Between Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all edges connecting nodes of type `"person"` to nodes of type `"company"`.

**SQL**

.. code-block:: sql

    SELECT e.*
    FROM edges e
    JOIN nodes n1 ON e.src = n1.id
    JOIN nodes n2 ON e.dst = n2.id
    WHERE n1.type = 'person' AND n2.type = 'company';

**Pandas**

.. code-block:: python

    merged_df = edges_df.merge(
        nodes_df[['id', 'type']], left_on='src', right_on='id', suffixes=('', '_src')
    ).merge(
        nodes_df[['id', 'type']], left_on='dst', right_on='id', suffixes=('', '_dst')
    )

    result = merged_df[
        (merged_df['type_src'] == 'person') &
        (merged_df['type_dst'] == 'company')
    ]

**Cypher**

.. code-block:: cypher

    MATCH (n1 {type: 'person'})-[e]->(n2 {type: 'company'})
    RETURN e;

**GFQL**

.. code-block:: python

    from graphistry import n, e_forward

    # df[['src', 'dst', ...]]
    chain([
        n({"type": "person"}), e_forward(), n({"type": "company"})
    ])._edges

**Explanation**:

- **GFQL**: Starts from nodes of type `"person"`, traverses forward edges, and reaches nodes of type `"company"`. The resulting edges are stored in `edges_df`. This version starts to gain the legibility and maintainability benefits of graph query syntax for graph tasks, and maintains the performance benefits of automatically vectorized pandas and GPU-accelerated cuDF.

---

Ex: Performing Multi-Hop Traversals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find nodes that are two hops away from node `"Alice"`.

**SQL**

.. code-block:: sql

    WITH first_hop AS (
        SELECT e1.dst AS node_id
        FROM edges e1
        WHERE e1.src = 'Alice'
    ),
    second_hop AS (
        SELECT e2.dst AS node_id
        FROM edges e2
        JOIN first_hop fh ON e2.src = fh.node_id
    )
    SELECT * FROM nodes
    WHERE id IN (SELECT node_id FROM second_hop);

**Pandas**

.. code-block:: python

    first_hop = edges_df[ edges_df['src'] == 'Alice' ]['dst']
    second_hop = edges_df[ edges_df['src'].isin(first_hop) ]['dst']
    result_nodes_df = nodes_df[ nodes_df['id'].isin(second_hop) ]

**Cypher**

.. code-block:: cypher

    MATCH (n {id: 'Alice'})-->()-->(m)
    RETURN m;

**GFQL**

.. code-block:: python

    from graphistry import n, e_forward

    # df[['id', ...]]
    g.chain([
        n({g._node: "Alice"}), e_forward(), e_forward(), n(name='m')
    ])._nodes.query('m')

**Explanation**:

- **GFQL**: Starts at node `"Alice"`, performs two forward hops, and obtains nodes two steps away. Results are in `nodes_df`. Building on the expressive and performance benefits of the previous 1-hop example, it begins adding the parallel path finding benefits of GFQL over Cypher, which benefits both CPU and GPU usage.

---

Ex: Filtering Edges and Nodes with Conditions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all edges where the weight is greater than `0.5`.

**SQL**

.. code-block:: sql

    SELECT * FROM edges
    WHERE weight > 0.5;

**Pandas**

.. code-block:: python

    filtered_edges_df = edges_df[ edges_df['weight'] > 0.5 ]

**Cypher**

.. code-block:: cypher

    MATCH ()-[e]->()
    WHERE e.weight > 0.5
    RETURN e;

**GFQL**

.. code-block:: python

    from graphistry import e_forward

    # df[['src', 'dst', 'weight', ...]]
    g.chain([ e_forward(edge_query='weight > 0.5') ])._edges

**Explanation**:

- **GFQL**: Uses `e_forward(edge_query='weight > 0.5')` to filter edges where `weight > 0.5`. This version introduces the string query form that can be convenient. Underneath, it still benefits from the vectorized execution of Pandas and cuDF.

---

Ex: Aggregations and Grouping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Count the number of outgoing edges for each node.

**SQL**

.. code-block:: sql

    SELECT src, COUNT(*) AS out_degree
    FROM edges
    GROUP BY src;

**Pandas**

.. code-block:: python

    out_degree = edges_df.groupby('src').size().reset_index(name='out_degree')

**Cypher**

.. code-block:: cypher

    MATCH (n)-[e]->()
    RETURN n.id AS node_id, COUNT(e) AS out_degree;

**GFQL**

.. code-block:: python

    # df[['src', 'out_degree']]
    g._edges.groupby('src').size().reset_index(name='out_degree')

**Explanation**:

- **GFQL**: Performs aggregation directly on `g._edges` using standard dataframe operations. Or even shorter, call `g.get_degrees()` to enrich each node with in, out, and total degrees. This version benefits from the hardware-accelerated columnar analytics execution of Pandas and cuDF, and the simplicity of dataframe operations.

---

.. _all-paths:

Ex: All Paths and Connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all paths between nodes ``"Alice"`` and ``"Bob"`` that go through friendships.

**SQL**

.. code-block:: sql

    WITH RECURSIVE path AS (
        -- Base case: Start from "Alice" (no type or edge restrictions)
        SELECT e.src, e.dst, ARRAY[e.src, e.dst] AS full_path, 1 AS hop
        FROM edges e
        WHERE e.src = 'Alice'
        
        UNION ALL

        -- Recursive case: Expand path where intermediate src/dst are 'people' and edge is 'friend'
        SELECT e.src, e.dst, full_path || e.dst, p.hop + 1
        FROM edges e
        JOIN path p ON e.src = p.dst
        JOIN nodes n_src ON e.src = n_src.id  -- Check src type for intermediate nodes
        JOIN nodes n_dst ON e.dst = n_dst.id  -- Check dst type for intermediate nodes
        WHERE n_src.type = 'person' AND n_dst.type = 'person'  -- Intermediate nodes must be 'people'
        AND e.type = 'friend'  -- Intermediate edges must be 'friend'
        AND e.dst != ALL(full_path)  -- Avoid cycles (optional)
    )
    -- Final filter to ensure the path ends with "Bob"
    SELECT *
    FROM path
    WHERE dst = 'Bob';

**Pandas**

.. code-block:: python

    def find_paths_fixed_point(edges_df, nodes_df, start_node, end_node):
        # Initialize paths with base case (start with 'Alice')
        paths = [{'path': [start_node], 'last_node': start_node}]
        all_paths = []
        expanded = True  # Continue loop as long as there are paths to expand

        while expanded:
            new_paths = []
            expanded = False

            # Expand each path
            for path in paths:
                last_node = path['last_node']

                # Find all outgoing 'friend' edges from the last node
                valid_edges = edges_df.merge(nodes_df, left_on='dst', right_on='id') \
                                    .merge(nodes_df, left_on='src', right_on='id') \
                                    [(edges_df['src'] == last_node) & 
                                    (edges_df['type'] == 'friend') &
                                    (nodes_df['type_x'] == 'person') &  # src is 'person'
                                    (nodes_df['type_y'] == 'person')]   # dst is 'person'

                for _, edge in valid_edges.iterrows():
                    new_path = path['path'] + [edge['dst']]

                    # If we reached 'Bob', add to all_paths
                    if edge['dst'] == end_node:
                        all_paths.append(new_path)
                    else:
                        # Otherwise, add to new paths to continue expanding
                        new_paths.append({'path': new_path, 'last_node': edge['dst']})
                        expanded = True  # Mark that we found new paths to expand

            # Stop if no new paths were found (fixed-point behavior)
            paths = new_paths

        return all_paths

    # Run the pathfinding function to fixed point
    paths = find_paths_fixed_point(edges_df, nodes_df, 'Alice', 'Bob')

**Cypher**

.. code-block:: cypher

    MATCH p = (n1 {id: 'Alice'})-[e:friend*]-(n2 {id: 'Bob'})
    WHERE ALL(rel IN relationships(p) WHERE type(rel) = 'friend')
    AND ALL(node IN NODES(p) WHERE node.type = 'person')
    RETURN p;

**GFQL**

.. code-block:: python

    # g._edges: df[['src', 'dst', ...]]
    # g._nodes: df[['id', ...]]
    g.chain([
        n({"id": "Alice"}), 
        e_forward(
            source_node_query='type == "person"',
            edge_query='type == "friend"',
            destination_node_query='type == "person"',
            to_fixed_point=True), 
        n({"id": "Bob"})
    ])

**Explanation**:

- **GFQL**: Uses `e(to_fixed_point=True)` to find edge sequences of arbitrary length between nodes `"Alice"` and `"Bob"`. The SQL and Pandas version suffer from syntactic and semantic imepedance mismatch with graph tasks on this example.

---

Ex: Community Detection and Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Identify communities within the graph using the Louvain algorithm.

**SQL and Pandas**

- Not designed for complex graph algorithms like community detection.

**Cypher**

.. code-block:: cypher

    CALL algo.louvain.stream() YIELD nodeId, communityId

**GFQL**

.. code-block:: python

    # g._nodes: df[['id', 'louvain']]
    g.compute_cugraph('louvain')._nodes

**Explanation**:

- **GFQL**: Enriches with many algorithms such as the GPU-accelerated :func:`graphistry.plugins.cugraph.compute_cugraph` for community detection. Any CPU and GPU library can be used, with top plugins already natively supported out-of-the-box.

---

Ex: Time-Windowed Graph Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all edges between nodes `"Alice"` and `"Bob"` that occurred in the last 7 days.

**SQL**

.. code-block:: sql

    SELECT * FROM edges
    WHERE ((src = 'Alice' AND dst = 'Bob') OR (src = 'Bob' AND dst = 'Alice')) 
      AND timestamp >= NOW() - INTERVAL '7 days';

.. warning::

    This version incorrectly simplifies to a two-hop relationship. For multihop scenarios, refer to :ref:`all-paths` for more advanced techniques.

**Pandas**

.. code-block:: python

    filtered_edges_df = edges_df[
        ((edges_df['src'] == 'Alice') & (edges_df['dst'] == 'Bob')) |
        ((edges_df['src'] == 'Bob') & (edges_df['dst'] == 'Alice')) &
        (edges_df['timestamp'] >= pd.Timestamp.now() - pd.Timedelta(days=7))
    ]

.. warning::

    This version incorrectly simplifies to a two-hop relationship. For multihop scenarios, refer to :ref:`all-paths` for more advanced techniques.

**Cypher**

.. code-block:: cypher

    MATCH path = (a {id: 'Alice'})-[e]-(b {id: 'Bob'})
    WHERE e.timestamp >= datetime().subtract(duration({days: 7}))
    RETURN e;

**GFQL**

.. code-block:: python

    past_week = pd.Timestamp.now() - pd.Timedelta(7)
    g.chain([
        n({"id": {"$in": ["Alice", "Bob"]}}), 
        e_forward(edge_query=f'timestamp >= "{past_week}"'), 
        n({"id": {"$in": ["Alice", "Bob"]}})
    ])._edges

**Explanation**:

- **SQL** and **Pandas**: These versions incorrectly simplify to a two-hop relationships; for multihop scenarios, refer to :ref:`all-paths`.

- **GFQL**: Utilizes the `chain` method to filter edges between `"Alice"` and `"Bob"` based on a timestamp within the last 7 days. This approach allows for multihop relationships as it leverages the graph's structure, and further using cuDF for GPU acceleration when available.


---

Ex: Parallel Pathfinding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Objective**: Find all paths from `"Alice"` to `"Bob"` and `"Charlie"` in parallel. Parallel pathfinding is particularly interesting because it allows for efficient querying of multiple target nodes at the same time, reducing the time and complexity required to compute multiple independent paths, especially in large graphs.

**SQL**

- **Not suitable**: SQL is not designed for pathfinding on graphs.

**Pandas**

- **Not suitable**: Pandas is not designed for pathfinding across graphs.

**Cypher**


.. warning::

    Cypher is **path-oriented** and does not natively support parallel pathfinding. Each path must be processed individually, which can result in performance bottlenecks for large graphs or multiple targets. Neo4j users can utilize the APOC or GDS libraries to add parallelism, but this is a limited external workaround, rather than a native strength.

.. code-block:: cypher

    MATCH (a {id: 'Alice'}), (target)
    WHERE target.id IN ['Bob', 'Charlie']
    CALL algo.shortestPath.stream(a, target)
    YIELD nodeId, cost
    RETURN nodeId, cost;

**GFQL**

.. code-block:: python

    from graphistry import n, e_forward

    # g._nodes: cudf.DataFrame[['src', 'dst', ...]]
    g.chain([
        n({"id": "Alice"}), 
        e_forward(to_fixed_point=False), 
        n({"id": is_in(["Bob", "Charlie"])})
    ], engine='cudf')

**Explanation**:


- **Cypher**: Cypher processes paths individually and does not support native parallelism. Libraries like APOC or GDS offer a way to achieve parallel execution, but this adds complexity.
  
- **GFQL**: GFQL natively supports parallel pathfinding using a bulk wavefront algorithm, processing all paths at once, making it highly efficient in GPU-accelerated environments.

---

Ex: GPU Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Objective**: Execute pathfinding queries on the GPU, computing all paths from `"Alice"` to `"Bob"` and `"Charlie"` simultaneously across hardware resources.

**SQL**

- **Not suitable**: SQL is not designed for parallel execution of graph queries.

**Pandas**

- **Not suitable**: Pandas is not designed for parallel execution across graphs.

**Cypher**

- **Not suitable**: Popular Cypher engines like Neo4j do not natively support GPU execution.

**GFQL**

.. code-block:: python

    from graphistry import n, e_forward

    # Executing pathfinding queries in parallel
    g.chain([
        n({"id": "Alice"}), 
        e_forward(to_fixed_point=False), 
        n({"id": is_in(["Bob", "Charlie"])})
    ], engine='cudf')

**Explanation**:

This example builds on the previous one, showing how **GFQL** handles parallel execution natively. GFQL benefits from **bulk vector processing**, which boosts performance in both CPU and GPU modes:

- **In CPU environments**, the bulk processing model accelerates query execution algorithmically and takes advantage of hardware parallelism, improving efficiency.
  
- **In GPU mode**, GFQL **natively parallelizes** pathfinding, further leveraging hardware acceleration to process multiple paths concurrently and quickly, making it highly efficient for large-scale graph traversals.

---







Common Graph Visualization Tasks
------------------------------------

We'll cover a range of common graph visualization tasks:

.. contents::
   :depth: 2
   :local:

Ex: Simple Plotting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Quickly visualize a graph using the `.plot()` method.

**SQL**

- **Not suitable**: SQL does not provide direct graph visualization capabilities.

**Pandas**

- **Not suitable**: Pandas lacks built-in methods for graph visualization.

**Cypher**

- **Not suitable**: Cypher does not include visualization functions natively.

**GFQL**

.. code-block:: python

    g.plot()

**Explanation**:

This example demonstrates how to create a quick visual representation of GFQL results using the :meth:`graphistry.PlotterBase.plot` method. For more visualization techniques, see :ref:`10min-viz`.

---










Common Data Loading and Shaping Tasks
--------------------------------------

We'll cover a range of common tasks related to data loading and shaping for graph structures:

.. contents::
   :depth: 2
   :local:

Ex: Data Loading with Pandas from CSV
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from CSV files using `pandas` and converting to graph structures.

**GFQL**

.. code-block:: python

    import pandas as pd
    import graphistry

    # pd.DataFrame[['src', 'dst', ...]]
    df = pd.read_csv('data.csv')
    
    # g._edges: df[['src', 'dst', ...]]
    g = graphistry.edges(df, 'src', 'dst')

**Explanation**:

This example illustrates how to load data from a CSV file using `pandas` and bind it into a graph structure via :meth:`graphistry.PlotterBase.PlotterBase.edges` for using **GFQL**. For more information on using `pandas`, refer to the `official pandas documentation <https://pandas.pydata.org/docs/>`__.

---

Ex: GPU Data Loading with cuDF from Parquet
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Demonstrate loading data from Parquet files using GPU-accelerated `cuDF` and converting to graph structures.

**GFQL**

.. code-block:: python

    import cudf
    import graphistry

    # cudf.DataFrame[['src', 'dst', ...]]
    df = cudf.read_parquet('data.parquet')

    # g._edges: df[['src', 'dst', ...]]
    g = graphistry.edges(df, 'src', 'dst')

**Explanation**:

This example showcases how to load data from a Parquet file using `cuDF` and convert it into a graph structure with **GFQL**. For further details on using `cuDF`, refer to the official `cuDF <https://docs.rapids.ai/api/cudf/stable/>`__ documentation.

---

Ex: Nodes and Edges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to convert loaded data into graph structures using `.edges()` and `.nodes()` when both are available.

**GFQL**

.. code-block:: python

    # pd.DataFrame[['n_id', ...]]
    df1 = pd.read_csv('nodes.csv')

    # pd.DataFrame[['src', 'dst', ...]]
    df2 = pd.read_csv('edges.csv')

    # g._edges: df2[['src', 'dst', ...]]
    # g._nodes: df1[['n_id', ...]] <-- optional
    g = graphistry.edges(df2, 'src', 'dst').nodes(df1, 'n_id')


**Explanation**:

This example demonstrates how to bind graph data for nodes and edges using **GFQL**. The :meth:`graphistry.PlotterBase.PlotterBase.edges` method is used to load edge data. Binding nodes data is optional, and via method  `graphistry.PlotterBase.PlotterBase.nodes`.

---

Ex: Shaping Graph Data - Multiple Node Columns with Hypergraphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss how to create creates from rows with multiple columns representing nodes via hypergraphs with the default `direct=False` parameter.

**GFQL**

.. code-block:: python
    
    g = graphistry.hypergraph(df, entity_cols=['a', 'b', 'c'])['graph']
    # g._node == 'nodeID'
    # g._nodes: df[['nodeTitle', 'type', 'category', 'nodeID', 'a', 'b', 'c', 'd', 'e', 'EventID']]
    # g._source == 'attribID'
    # g._destination == 'EventID'
    # g._nodes.type.unique() == ['a', 'b', 'c', 'EventID']
    # g._edges: df[['EventID', 'attribID', 'a', 'd', 'e', 'c', 'edgeType']]

**Explanation**:

This example explains how to shape graph data into a hypergraph format using the default `direct=False` parameter. In this case, all values in columns `a`, `b`, and `c` become nodes. Additionally, as `direct=False`, each row also becomes a node, with edges to its corresponding values in columns `a`, `b`, and `c`. When `direct=True`, the nodes for columns `a`, `b`, and `c` would be directly connected. Refer to :meth:`graphistry.PlotterBase.PlotterBase.hypergraph` for more variants and advanced usage.

---









Common Graph Machine Learning and Graph AI Tasks
---------------------------------------------------

We'll cover a range of common tasks related to graph machine learning and AI you can do on GFQL results:

.. contents::
   :depth: 2
   :local:

Ex: UMAP Cluster & Dimensionality Reduction for Embeddings & Similarity Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Show how to apply UMAP for dimensionality reduction, turning wide data into for clustering, embeddings, and similarity graphs.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(df).umap()

**Explanation**:

This example demonstrates how to utilize :meth:`graphistry.umap_utils.UMAPMixin.umap`. See its reference docs for many optional overrides and usage modes, such as defining `X=['col1', 'col2', ...]` to specify which columns to cluster.

---

Ex: UMAP Fit/Transform for Scaling
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain how to use UMAP's fit/transform capabilities for scaling features across datasets.

**GFQL**

.. code-block:: python

    # Train; feature columns X and label column y are optional
    g1 = graphistry.nodes(df_sample).umap(X=['col_1', ..., 'col_n'], y='col_m')

    # Transform new data
    g2 = g1.transform_umap(new_df, return_graph=True)

    # Visualize new data under initial UMAP embedding
    g2.plot()

**Explanation**:

This example illustrates how to fit a UMAP model on one dataset and then use that model to transform another dataset, enabling consistent scaling of features. For more details on using fit/transform with UMAP, consult the :meth:`graphistry.umap_utils.UMAPMixin.umap` documentation.

---

Ex: Basic RGCN for Graph Neural Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Introduce the basic concepts of RGCNs and how to build and train a simple model.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(ndf).edges(edf, src, dst)
    g.build_gnn(X_nodes=['feature_1', 'feature_2'], y_nodes='label')
    g.train()  # Train the model

**Explanation**:

This example provides an introduction to building and training a basic Relational Graph Convolutional Network (RGCN) using **GFQL**. For further information on GNNs, see the **GNN Models** section of our documentation.

---

Ex: Anomaly Detection Using RGCN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Utilize the trained RGCN model to detect anomalies in graph data based on learned representations.

**GFQL**

.. code-block:: python

    anomalies = g.detect_anomalies()  # Detect anomalies in the graph using the trained model

**Explanation**:

This example demonstrates how to leverage a trained RGCN model to identify anomalies in graph data. For additional details on anomaly detection techniques, refer to the relevant sections in our documentation.

---

Ex: Clustering with DBSCAN
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Discuss using DBSCAN for clustering nodes or edges based on features.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(ndf).edges(edf, src, dst).umap()
    g.dbscan(eps=0.5, min_samples=5)  # Apply DBSCAN clustering

**Explanation**:

This example illustrates how to apply DBSCAN clustering to graph data after reducing dimensionality with UMAP. For more information on clustering techniques, consult the **Clustering** section in our documentation.

---

Ex: Automated Feature Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Illustrate generating features from raw data for AI applications.

**GFQL**

.. code-block:: python

    g = graphistry.nodes(df).featurize(kind='nodes', X=['raw_feature_1', 'raw_feature_2'])

**Explanation**:

This example demonstrates how to automatically generate features from raw data using **GFQL**. For additional insights into feature generation techniques, refer to the **Feature Generation** section in our documentation.

---

Ex: Semantic Search in Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Implement semantic search using graph embeddings and natural language queries.

**GFQL**

.. code-block:: python

    results_df, query_vector = g.search('natural language query')

**Explanation**:

This example showcases how to perform semantic searches within graph data using embeddings. For further details on implementing semantic search, see the **Semantic Search** section in our documentation.

---

Ex: Knowledge Graph Embeddings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Explain training models for knowledge graph embeddings and predicting relationships.

**GFQL**

.. code-block:: python

    g = graphistry.edges(edf, src, dst)
    g2 = g.embed(relation='relationship_column')

**Explanation**:

This example describes how to train models for knowledge graph embeddings with **GFQL** and how to predict relationships between entities. For more information, refer to the **Knowledge Graph Embeddings** section in our documentation.

















GFQL Functions and Equivalents
------------------------------

**Node Matching**

- **SQL**: ``SELECT * FROM nodes WHERE ...``
- **Pandas**: ``nodes_df[ condition ]``
- **Cypher**: ``MATCH (n {property: value})``
- **GFQL**: ``n({ "property": value })``

**Edge Matching**

- **SQL**: ``SELECT * FROM edges WHERE ...``
- **Pandas**: ``edges_df[ condition ]``
- **Cypher**: ``MATCH ()-[e {property: value}]->()``
- **GFQL**: ``e_forward({ "property": value })`` or ``e_reverse({ "property": value })`` or ``e({ "property": value })``

**Traversal**

- **SQL**: Complex joins or recursive queries
- **Pandas**: Multiple merges; not efficient for deep traversals
- **Cypher**: Patterns like ``()-[]->()`` for traversal
- **GFQL**: Chains of ``n()``, ``e_forward()``, ``e_reverse()``, and ``e()`` functions

Tips for Users
--------------

- **Data Scientists and Analysts**: Use your Pandas knowledge. GFQL operates on dataframes, allowing familiar operations.
- **Engineers and Developers**: Integrate GFQL into Python applications without extra infrastructure.
- **Database Administrators**: Complement SQL queries with GFQL for graph data without changing databases.
- **Graph Enthusiasts**: Start with simple queries and explore complex analytics. Visualize results using PyGraphistry.

Additional Resources
--------------------

- :ref:`gfql-quick`
- :ref:`gfql-predicates-quick`: Use predicates for filtering on nodee and edge attributes.
- :ref:`10min`: Visualize GFQL queries with GPU-accelerated tools.

Conclusion
----------

GFQL bridges the gap between traditional querying languages and graph analytics. By translating queries from SQL, Pandas, and Cypher into GFQL, you can leverage powerful graph queries within your Python workflows.

Start exploring GFQL today and unlock new insights from your graph data!
