.. _gfql-translate:

Translate Between SQL, Pandas, Cypher, and GFQL
=================================================

This guide provides a comparison between **SQL**, **Pandas**, **Cypher**, and **GFQL**, helping you translate familiar queries into GFQL.

Introduction
------------

GFQL (GraphFrame Query Language) is designed to be intuitive for users familiar with SQL, Cypher, or dataframe like Pandas and Spark. By comparing equivalent queries across these languages, you can quickly grasp GFQL's syntax, benefits, and start utilizing its powerful graph querying capabilities within your workflows.

GFQL operates on graph DataFrames - graphs represented as node and edge DataFrames. This DataFrame-native approach enables seamless integration with the PyData ecosystem and natural vectorization for both CPU and GPU processing.

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

**Objective**: Find all nodes where the ``type`` is ``"person"``.

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

- **GFQL**: ``n({"type": "person"})`` filters nodes where ``type`` is ``"person"``. ``g.gfql([...])`` applies this filter to the graph ``g``, and ``._nodes`` retrieves the resulting nodes. The performance is similar to that of Pandas (CPU) or cuDF (GPU).

.. graphviz::

   digraph find_nodes {
       node [shape=ellipse];
       person1 [style=filled, fillcolor=lightgreen, label="person"];
       person2 [style=filled, fillcolor=lightgreen, label="person"];
       company1 [label="company"];
       person1 -> company1 [style=dashed, color=gray];
   }

---

Exploring Relationships Between Nodes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all edges connecting nodes of type ``"person"`` to nodes of type ``"company"``.

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

- **GFQL**: Starts from nodes of type ``"person"``, traverses forward edges, and reaches nodes of type ``"company"``. The resulting edges are stored in ``edges_df``. This version starts to gain the legibility and maintainability benefits of graph query syntax for graph tasks, and maintains the performance benefits of automatically vectorized pandas and GPU-accelerated cuDF.

.. graphviz::

   digraph relationships {
       rankdir=LR;
       person [style=filled, fillcolor=lightblue, label="person"];
       company [style=filled, fillcolor=lightyellow, label="company"];
       person -> company [label="works_at", color=blue, penwidth=2];
   }

---

Performing Multi-Hop Traversals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find nodes that are two hops away from node ``"Alice"``.

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

- **GFQL**: Starts at node ``"Alice"``, performs two forward hops, and obtains nodes two steps away. Results are in ``nodes_df``. Building on the expressive and performance benefits of the previous 1-hop example, it begins adding the parallel path finding benefits of GFQL over Cypher, which benefits both CPU and GPU usage.

.. graphviz::

   digraph multi_hop {
       rankdir=LR;
       Alice [style=filled, fillcolor=lightblue, label="Alice"];
       n1 [label="?"];
       n2 [style=filled, fillcolor=lightgreen, label="m"];
       Alice -> n1 [label="hop 1"];
       n1 -> n2 [label="hop 2"];
   }

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

- **GFQL**: Uses ``e_forward(edge_query='weight > 0.5')`` to filter edges where ``weight > 0.5``. This version introduces the string query form that can be convenient. Underneath, it still benefits from the vectorized execution of Pandas and cuDF.

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

- **GFQL**: Performs aggregation directly on ``g._edges`` using standard dataframe operations. Or even shorter, call ``g.get_degrees()`` to enrich each node with in, out, and total degrees. This version benefits from the hardware-accelerated columnar analytics execution of Pandas and cuDF, and the simplicity of dataframe operations.

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
    g.gfql([
        n({"id": "Alice"}), 
        e_forward(
            source_node_query='type == "person"',
            edge_query='type == "friend"',
            destination_node_query='type == "person"',
            to_fixed_point=True), 
        n({"id": "Bob"})
    ])

**Explanation**:

- **GFQL**: Uses ``e(to_fixed_point=True)`` to find edge sequences of arbitrary length between nodes ``"Alice"`` and ``"Bob"``. The SQL and Pandas version suffer from syntactic and semantic imepedance mismatch with graph tasks on this example.

.. graphviz::

   digraph all_paths {
       rankdir=LR;
       Alice [style=filled, fillcolor=lightblue, label="Alice"];
       Bob [style=filled, fillcolor=lightgreen, label="Bob"];
       m1 [label="person"];
       m2 [label="person"];
       n1 [label="person"];
       Alice -> m1 [label="friend"];
       m1 -> m2 [label="friend"];
       m2 -> Bob [label="friend"];
       Alice -> n1 [label="friend"];
       n1 -> Bob [label="friend"];
   }

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

    # g._nodes: df[['id', 'louvain']]
    g.compute_cugraph('louvain')._nodes

**Explanation**:

- **GFQL**: Enriches with many algorithms such as the GPU-accelerated :func:`graphistry.plugins.cugraph.compute_cugraph` for community detection. Any CPU and GPU library can be used, with top plugins already natively supported out-of-the-box.

---

Time-Windowed Graph Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all edges between nodes ``"Alice"`` and ``"Bob"`` that occurred in the last 7 days.

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

- **GFQL**: Utilizes the ``chain`` method to filter edges between ``"Alice"`` and ``"Bob"`` based on a timestamp within the last 7 days. This approach allows for multihop relationships as it leverages the graph's structure, and further using cuDF for GPU acceleration when available.


---

Parallel Pathfinding
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


**Objective**: Find all paths from ``"Alice"`` to ``"Bob"`` and ``"Charlie"`` in parallel. Parallel pathfinding is particularly interesting because it allows for efficient querying of multiple target nodes at the same time, reducing the time and complexity required to compute multiple independent paths, especially in large graphs.

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


- **Cypher**: Cypher processes paths individually and does not support native parallelism. Libraries like APOC or GDS offer a way to achieve parallel execution, but this adds complexity.
  
- **GFQL**: GFQL natively supports parallel pathfinding using a bulk wavefront algorithm, processing all paths at once, making it highly efficient in GPU-accelerated environments.

---

GPU Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

*Objective**: Execute pathfinding queries on the GPU, computing all paths from ``"Alice"`` to ``"Bob"`` and ``"Charlie"`` simultaneously across hardware resources.

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
