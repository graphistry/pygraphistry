.. _gfql-translate:

Translate Between SQL, Pandas, Cypher, and GFQL
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

Finding Nodes with Specific Properties
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
    g.gfql([ n({"type": "person"}) ])._nodes

**Explanation**:

- **GFQL**: `n({"type": "person"})` filters nodes where `type` is `"person"`. `g.gfql([...])` applies this filter to the graph `g`, and `._nodes` retrieves the resulting nodes. The performance is similar to that of Pandas (CPU) or cuDF (GPU).

---

Exploring Relationships Between Nodes
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
    g.gfql([
        n({"type": "person"}), e_forward(), n({"type": "company"})
    ])._edges

**Explanation**:

- **GFQL**: Starts from nodes of type `"person"`, traverses forward edges, and reaches nodes of type `"company"`. The resulting edges are stored in `edges_df`. This version starts to gain the legibility and maintainability benefits of graph query syntax for graph tasks, and maintains the performance benefits of automatically vectorized pandas and GPU-accelerated cuDF.

---

Performing Multi-Hop Traversals
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
    g.gfql([
        n({g._node: "Alice"}), e_forward(), e_forward(), n(name='m')
    ])._nodes.query('m')

**Explanation**:

- **GFQL**: The ``gfql([...])`` pattern is GFQL's equivalent to Cypher's MATCH, but executes as bulk vector joins for performance. Starting at node `"Alice"`, it performs two forward hops and obtains nodes two steps away. Results are standard pandas/cuDF DataFrames. Building on the expressive and performance benefits of the previous 1-hop example, it demonstrates the parallel path finding benefits of GFQL over Cypher, which benefits both CPU and GPU usage.

---

Filtering Edges and Nodes with Conditions
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
    g.gfql([ e_forward(edge_query='weight > 0.5') ])._edges

**Explanation**:

- **GFQL**: Uses `e_forward(edge_query='weight > 0.5')` to filter edges where `weight > 0.5`. This version introduces the string query form that can be convenient. Underneath, it still benefits from the vectorized execution of Pandas and cuDF.

---

Aggregations and Grouping
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

All Paths and Connectivity
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
    
    # Manual path tracking with 'p' attribute
    g.gfql([
        n({"id": "Alice", "p": True}), 
        e_forward(
            source_node_query='type == "person"',
            edge_query='type == "friend"',
            destination_node_query='type == "person"',
            to_fixed_point=True,
            name="p"), 
        n({"id": "Bob", "p": True})
    ])
    
    # Filter path elements: result._nodes[result._nodes["p"]] or result._edges[result._edges["p"]]

.. tip::
   
   **Manual Path Tracking in GFQL**: Since GFQL doesn't have automatic path annotation, you can manually tag nodes and edges with a boolean attribute (e.g., ``"p": True``) and use the ``name`` parameter to mark traversed edges. This allows you to filter path elements later using standard DataFrame operations.

**Explanation**:

- **GFQL**: Uses `e(to_fixed_point=True)` to find edge sequences of arbitrary length between nodes `"Alice"` and `"Bob"`. Manual path tracking is achieved by tagging nodes and edges with attributes. The SQL and Pandas versions suffer from syntactic and semantic impedance mismatch with graph tasks on this example.

---

Community Detection and Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Identify communities within the graph using the Louvain algorithm.

**SQL and Pandas**

- Not designed for complex graph algorithms like community detection.

**Cypher**

.. code-block:: cypher

    CALL algo.louvain.stream() YIELD nodeId, communityId

**GFQL**

.. code-block:: python

    # Using compute_cugraph directly
    # g._nodes: df[['id', 'louvain']]
    g.compute_cugraph('louvain')._nodes

    # Or using GFQL's call operation
    from graphistry import Let, call
    
    # g._nodes: df[['id', 'louvain']]
    Let('communities', call('louvain')).run(g)._nodes

**Explanation**:

- **GFQL**: Enriches with many algorithms such as the GPU-accelerated :func:`graphistry.plugins.cugraph.compute_cugraph` for community detection. The :func:`call <graphistry.compute.Call.call>` operation in GFQL provides a unified interface to invoke these algorithms within GFQL queries. Any CPU and GPU library can be used, with top plugins already natively supported out-of-the-box. Unlike Cypher (which uses external APOC/GDS libraries), GFQL integrates GPU-native algorithms directly via ``call(...)``, with support for chaining, filtering, and visualization.

---

Time-Windowed Graph Analytics
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
    g.gfql([
        n({"id": {"$in": ["Alice", "Bob"]}}), 
        e_forward(edge_query=f'timestamp >= "{past_week}"'), 
        n({"id": {"$in": ["Alice", "Bob"]}})
    ])._edges

**Explanation**:

- **SQL** and **Pandas**: These versions incorrectly simplify to a two-hop relationships; for multihop scenarios, refer to :ref:`all-paths`.

- **GFQL**: Utilizes the `chain` method to filter edges between `"Alice"` and `"Bob"` based on a timestamp within the last 7 days. This approach allows for multihop relationships as it leverages the graph's structure, and further using cuDF for GPU acceleration when available.


---

Parallel Pathfinding
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
    g.gfql([
        n({"id": "Alice"}), 
        e_forward(to_fixed_point=False), 
        n({"id": is_in(["Bob", "Charlie"])})
    ], engine='cudf')

**Explanation**:

- **Cypher**: Cannot perform multi-target pathfinding in parallel without APOC or external workarounds. Cypher processes paths individually due to its per-path recursion model, creating performance bottlenecks for multiple targets.
  
- **GFQL**: Natively supports parallel pathfinding via wavefront join execution, processing all paths simultaneously. This bulk vector approach, combined with GPU acceleration, delivers significant performance advantages for multi-target scenarios.

---

GPU Execution
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
    g.gfql([
        n({"id": "Alice"}), 
        e_forward(to_fixed_point=False), 
        n({"id": is_in(["Bob", "Charlie"])})
    ], engine='cudf')

**Explanation**:

This example builds on the previous one, showing how **GFQL** handles parallel execution natively. GFQL benefits from **bulk vector processing**, which boosts performance in both CPU and GPU modes:

- **In CPU environments**, the bulk processing model accelerates query execution algorithmically and takes advantage of hardware parallelism, improving efficiency.
  
- **In GPU mode**, GFQL **natively parallelizes** pathfinding, further leveraging hardware acceleration to process multiple paths concurrently and quickly, making it highly efficient for large-scale graph traversals.

---










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

Graph Algorithms
----------------

GFQL provides built-in graph algorithms through the Call operation, similar to Neo4j's APOC procedures but with GPU acceleration and DataFrame integration.

**Objective**: Run various graph algorithms like PageRank, community detection, and pathfinding.

**Neo4j with APOC Procedures**

.. code-block:: cypher

    // PageRank
    CALL apoc.algo.pageRank(null, null) YIELD node, score
    RETURN node.name, score
    ORDER BY score DESC LIMIT 10;

    // Betweenness Centrality
    CALL apoc.algo.betweenness(null, null, 'BOTH') YIELD node, score

    // Shortest Path
    MATCH (start {name: 'Alice'}), (end {name: 'Bob'})
    CALL apoc.algo.dijkstra(start, end, 'KNOWS', 'weight') YIELD path

**GFQL with Call Operations**

.. code-block:: python

    from graphistry import call, n, e_forward, gt

    # PageRank (GPU-accelerated for large graphs)
    top_pagerank = g.gfql([
        call('compute_cugraph', {
            'alg': 'pagerank',
            'out_col': 'pagerank_score',
            'params': {'alpha': 0.85}
        })
    ])._nodes.nlargest(10, 'pagerank_score')

    # Betweenness Centrality (CPU version for precise results)
    g_centrality = g.gfql([
        call('compute_igraph', {
            'alg': 'betweenness',
            'out_col': 'betweenness_score',
            'directed': True
        })
    ])

    # Shortest Path (using hop with filtering)
    g_path = g.gfql([
        n({'name': 'Alice'}),
        call('hop', {
            'hops': 10,
            'edge_match': {'type': 'KNOWS'},
            'destination_node_match': {'name': 'Bob'}
        })
    ])

**APOC to GFQL Call Mapping**

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - APOC Procedure
     - GFQL Call Equivalent
     - Notes
   * - apoc.algo.pageRank
     - call('compute_cugraph', {'alg': 'pagerank'})
     - GPU-accelerated
   * - apoc.algo.louvain
     - call('compute_cugraph', {'alg': 'louvain'})
     - GPU-accelerated
   * - apoc.algo.betweenness
     - call('compute_igraph', {'alg': 'betweenness'})
     - CPU for accuracy
   * - apoc.path.expand
     - call('hop', {'hops': N})
     - Bulk parallel execution
   * - apoc.create.nodes
     - call('materialize_nodes')
     - From edges to nodes
   * - apoc.algo.community
     - call('compute_cugraph', {'alg': 'leiden'})
     - GPU-accelerated

**Advanced Algorithm Examples**

.. code-block:: python

    # GPU-accelerated layouts
    g_layout = g.gfql([
        call('layout_cugraph', {
            'layout': 'force_atlas2',
            'params': {'iterations': 500}
        })
    ])

    # Combined analysis and visualization (mixing backends)
    g_analyzed = g.gfql([
        # Filter to important nodes (built-in method)
        call('get_degrees', {'col': 'degree'}),
        n({'degree': gt(10)}),
        # Run community detection (GPU for speed)
        call('compute_cugraph', {'alg': 'louvain', 'out_col': 'community'}),
        # Calculate closeness (CPU-only algorithm)
        call('compute_igraph', {'alg': 'closeness', 'out_col': 'closeness'}),
        # Color by community
        call('encode_point_color', {'column': 'community'}),
        # Size by closeness centrality
        call('encode_point_size', {'column': 'closeness'})
    ])

**Performance Comparison**

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Algorithm
     - Neo4j+APOC
     - GFQL CPU
     - GFQL GPU
   * - PageRank (1M edges)
     - ~5s
     - ~2s
     - ~0.1s
   * - Louvain (1M edges)
     - ~8s
     - ~3s
     - ~0.2s
   * - 3-hop traversal
     - ~2s
     - ~0.5s
     - ~0.05s

Feature Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 10 15

   * - Feature
     - GFQL
     - Cypher
     - GSQL
     - SQL
     - Pandas
   * - Pattern match
     - **Yes**
     - **Yes**
     - **Yes**
     - JOIN
     - **No**
   * - Multi-hop
     - **Yes**
     - **Yes**
     - **Yes**
     - CTE
     - **No**
   * - Path return (``MATCH p=...``)
     - **Partial**\ :sup:`1`
     - **Yes**
     - **Yes**
     - **No**
     - **No**
   * - Optional match
     - **No**\ :sup:`2`
     - **Yes**
     - **Yes**
     - LEFT JOIN
     - **No**
   * - GPU execution
     - **Yes**
     - **No**
     - **No**
     - **No**
     - **Yes** (cuDF)
   * - Aggregations
     - **Partial**\ :sup:`3`
     - **Yes**
     - **Yes**
     - **Yes**
     - **Yes**
   * - Procedural logic
     - **Partial**\ :sup:`4`
     - **No**
     - **Yes**
     - **Yes**
     - **Yes**
   * - Visualization
     - **Yes**
     - **No**
     - **No**
     - **No**
     - **Partial**\ :sup:`5`

**Legend**: **Yes** = Native support | **Partial** = Partial/Manual support | **No** = Not supported

**Footnotes**:

:sup:`1` **Path return**: GFQL does not return nested path objects, but users can tag steps (e.g., ``name='p'``, ``path_id``) to simulate ``MATCH p = ... RETURN p``.

:sup:`2` **Optional match**: Not natively supported in GFQL yet, but could be emulated via post-join left merges.

:sup:`3` **Aggregations**: Done outside GFQL using Pandas/cuDF on ``.nodes`` and ``.edges``.

:sup:`4` **Procedural logic**: GFQL core is declarative, but users can compose DAGs via ``Let(...)`` and use embedded Python for loops, filters, and transformation.

:sup:`5` **Visualization**: GFQL includes built-in ``.plot()``/``encode_*()`` methods; Pandas requires external libraries (matplotlib, seaborn, etc.).

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
