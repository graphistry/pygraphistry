Translating Between SQL, Pandas, Cypher, and GFQL
=================================================

This guide provides a comparison between **SQL**, **Pandas**, **Cypher**, and **GFQL**, helping you translate familiar queries into GFQL.

Introduction
------------

GFQL (GraphFrame Query Language) is designed to be intuitive for users familiar with SQL, Pandas, or Cypher. By comparing equivalent queries across these languages, you can quickly grasp GFQL's syntax and start utilizing its powerful graph querying capabilities within your Python workflows.

Who Is This Guide For?
----------------------

- **Data Scientists and Analysts**: Familiar with Pandas or SQL, looking to explore graph relationships in their data.
- **Graph Engineers and Developers**: Experienced with Cypher or other graph query languages, interested in integrating graph queries into Python applications.
- **Database Administrators**: Looking to understand how GFQL can complement traditional SQL queries for graph data.
- **Anyone Exploring Graph Analytics**: Interested in learning how to perform graph queries using a dataframe-native approach.

Common Graph and Query Tasks
----------------------------

We'll cover a range of common graph and query tasks:

1. **Finding Nodes with Specific Properties**
2. **Exploring Relationships Between Nodes**
3. **Performing Multi-Hop Traversals**
4. **Filtering Edges and Nodes with Conditions**
5. **Aggregations and Grouping**
6. **All Paths and Connectivity**
7. **Community Detection and Clustering**

Translation Examples
--------------------

Example 1: Finding Nodes with Specific Properties
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

    people_nodes_df = g.chain([ n({"type": "person"}) ])._nodes

**Explanation**:

- **GFQL**: `n({"type": "person"})` filters nodes where `type` is `"person"`. `g.chain([...])` applies this filter to the graph `g`, and `._nodes` retrieves the resulting nodes. The performance is similar to that of Pandas (CPU) or cuDF (GPU).

---

Example 2: Exploring Relationships Between Nodes
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

    g_result = g.chain([
        n({"type": "person"}),
        e_forward(),
        n({"type": "company"})
    ])
    edges_df = g_result._edges

**Explanation**:

- **GFQL**: Starts from nodes of type `"person"`, traverses forward edges, and reaches nodes of type `"company"`. The resulting edges are stored in `edges_df`.

---

Example 3: Performing Multi-Hop Traversals
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

    g_2_hops = g.chain([
        n({g._node: "Alice"}),
        e_forward(),
        e_forward()
    ])
    nodes_df = g_2_hops._nodes

**Explanation**:

- **GFQL**: Starts at node `"Alice"`, performs two forward hops, and obtains nodes two steps away. Results are in `nodes_df`.

---

Example 4: Filtering Edges and Nodes with Conditions
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

    filtered_edges_df = g.chain([ e_forward(edge_query='weight > 0.5') ])._edges

**Explanation**:

- **GFQL**: Uses `e_forward({"weight": lambda w: w > 0.5})` to filter edges where `weight > 0.5`.

---

Example 5: Aggregations and Grouping
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

    out_degree = g._edges.groupby('src').size().reset_index(name='out_degree')

**Explanation**:

- **GFQL**: Performs aggregation directly on `g._edges` using standard dataframe operations. Or even shorter, call `g.get_degrees()` to enrich each node with in, out, and total degrees.

---

Example 6: All Paths and Connectivity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Find all paths between nodes ``"Alice"`` and ``"Bob"``.

**SQL and Pandas**

- Not suitable for path calculations in graphs.

**Cypher**

.. code-block:: cypher

    MATCH p = (n {id: 'Alice'})-[*]-(m {id: 'Bob'})
    RETURN p;

**GFQL**

.. code-block:: python

    g.chain([ n({g._node: "Alice"}), e(to_fixed_point=True), n({g._node: "Bob"}) ])

**Explanation**:

- **GFQL**: Uses `e(to_fixed_point=True)` to find edge sequences of arbitrary length between nodes `"Alice"` and `"Bob"`.

---

Example 7: Community Detection and Clustering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Objective**: Identify communities within the graph using the Louvain algorithm.

**SQL and Pandas**

- Not designed for complex graph algorithms like community detection.

**Cypher**

.. code-block:: cypher

    CALL algo.louvain.stream() YIELD nodeId, communityId

**GFQL**

.. code-block:: python

    nodes_df = g.compute_cugraph('louvain')._nodes[['id', 'louvain']]

**Explanation**:

- **GFQL**: Uses GPU-accelerated algorithms via `compute_cugraph()` for community detection.

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

- **GFQL Documentation**: Detailed documentation for advanced usage.
- **GFQL Predicates**: Use predicates for complex filtering conditions.
- **PyGraphistry Integration**: Visualize GFQL queries with GPU-accelerated tools.

Conclusion
----------

GFQL bridges the gap between traditional querying languages and graph analytics. By translating queries from SQL, Pandas, and Cypher into GFQL, you can leverage powerful graph queries within your Python workflows.

Start exploring GFQL today and unlock new insights from your graph data!
