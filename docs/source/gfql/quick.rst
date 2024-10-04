GFQL Quick Reference
====================

This quick reference page provides short examples of various parameters and usage patterns.

Basic Usage
-----------

**Chaining Operations**

.. code-block:: python

    g.chain([...], engine=EngineAbstract.AUTO)

- **`ops`**: Sequence of graph node and edge matchers (`ASTObject` instances).
- **`engine`**: Optional execution engine (`'cudf'` for GPU acceleration).

Node Matchers
-------------

**`n(filter_dict=None, name=None, query=None)`**

- Filter nodes based on attributes.
- **Parameters**:
  - `filter_dict`: `{attribute: value}` or `{attribute: condition_function}`
  - `name`: Optional label; adds a boolean column in the result.
  - `query`: Custom query string (e.g., `"age > 30 and country == 'USA'"`).

**Examples:**

- Match nodes where `type` is `'person'`:

  .. code-block:: python

      n({"type": "person"})

- Match nodes with `age` greater than 30:

  .. code-block:: python

      n({"age": lambda x: x > 30})

- Use a custom query string:

  .. code-block:: python

      n(query="age > 30 and country == 'USA'")

Edge Matchers
-------------

**`e_forward(edge_match=None, hops=1, to_fixed_point=False, source_node_match=None, destination_node_match=None, source_node_query=None, destination_node_query=None, edge_query=None, name=None)`**

- Traverse edges in the forward direction.
- **Parameters**:
  - `edge_match`: `{attribute: value}` or `{attribute: condition_function}`
  - `edge_query`: Custom query string for edge attributes.
  - `hops`: `int`, number of hops to traverse.
  - `to_fixed_point`: `bool`, continue traversal until no more matches.
  - `source_node_match`: Filter for source nodes.
  - `destination_node_match`: Filter for destination nodes.
  - `source_node_query`: Custom query string for source nodes.
  - `destination_node_query`: Custom query string for destination nodes.
  - `name`: Optional label.

**Examples:**

- Traverse 2 hops forward on edges where `status` is `'active'`:

  .. code-block:: python

      e_forward({"status": "active"}, hops=2)

- Use custom edge query strings:

  .. code-block:: python

      e_forward(edge_query="weight > 5 and type == 'connects'")

- Filter source and destination nodes with match dictionaries:

  .. code-block:: python

      e_forward(
          source_node_match={"status": "active"},
          destination_node_match={"age": lambda x: x < 30}
      )

- Filter source and destination nodes with queries:

  .. code-block:: python

      e_forward(
          source_node_query="status == 'active'",
          destination_node_query="age < 30"
      )

- Label matched edges:

  .. code-block:: python

      e_forward(name="active_edges")

**`e_reverse(...)` and `e_undirected(...)`**

- **`e_reverse`**: Same as `e_forward`, but traverses in reverse.
- **`e_undirected`**: Traverses edges regardless of direction.

Utility Functions
-----------------

**`is_in(values)`**

- Matches attribute values within a list.

**Example:**

- Match nodes where `category` is `'A'`, `'B'`, or `'C'`:

  .. code-block:: python

      n({"category": is_in(["A", "B", "C"])})

Combined Examples
-----------------

- **Find people connected to transactions via active relationships:**

  .. code-block:: python

      g.chain([
          n({"type": "person"}),
          e_forward({"status": "active"}),
          n({"type": "transaction"})
      ])

- **Label nodes and edges during traversal:**

  .. code-block:: python

      g.chain([
          n({"id": "start_node"}, name="start"),
          e_forward(name="edge1"),
          n({"level": 2}, name="middle"),
          e_forward(name="edge2"),
          n({"type": "end_type"}, name="end")
      ])

- **Traverse until no more matches (fixed point):**

  .. code-block:: python

      g.chain([
          n({"status": "infected"}),
          e_forward(to_fixed_point=True),
          n(name="reachable")
      ])

- **Filter by multiple conditions:**

  .. code-block:: python

      g.chain([
          n({"type": is_in(["server", "database"])}),
          e_undirected({"protocol": "TCP"}, hops=3),
          n(query="risk_level >= 8")
      ])

- **Use custom queries in matchers:**

  .. code-block:: python

      g.chain([
          n(query="age > 30 and country == 'USA'"),
          e_forward(edge_query="weight > 5"),
          n(query="status == 'active'")
      ])

GPU Acceleration
----------------

- **Enable GPU mode:**

  .. code-block:: python

      g.chain([...], engine='cudf')

- **Example with cuDF DataFrames:**

  .. code-block:: python

      import cudf

      e_gdf = cudf.from_pandas(edge_df)
      n_gdf = cudf.from_pandas(node_df)

      g = graphistry.nodes(n_gdf, 'node_id').edges(e_gdf, 'src', 'dst')
      g.chain([...], engine='cudf')

Advanced Usage
--------------

- **Traversal with source and destination node filters and queries:**

  .. code-block:: python

      e_forward(
          edge_query="type == 'follows' and weight > 2",
          source_node_match={"status": "active"},
          destination_node_query="age < 30",
          hops=2,
          name="social_edges"
      )

- **Node matcher with all parameters:**

  .. code-block:: python

      n(
          filter_dict={"department": "sales"},
          query="age > 25 and tenure > 2",
          name="experienced_sales"
      )

- **Edge matcher with all parameters:**

  .. code-block:: python

      e_reverse(
          edge_match={"transaction_type": "refund"},
          edge_query="amount > 100",
          source_node_match={"status": "inactive"},
          destination_node_match={"region": "EMEA"},
          name="large_refunds"
      )

Parameter Summary
-----------------

- **Common Parameters:**
  - `filter_dict`: Attribute filters (e.g., `{"status": "active"}`)
  - `query`: Custom query string (e.g., `"age > 30"`)
  - `hops`: Number of steps to traverse (`int`, default `1`)
  - `to_fixed_point`: Continue traversal until no more matches (`bool`, default `False`)
  - `name`: Label for matchers (`str`)
  - `source_node_match`, `destination_node_match`: Filters for connected nodes
  - `source_node_query`, `destination_node_query`: Queries for connected nodes
  - `edge_match`: Filters for edges
  - `edge_query`: Query for edges
  - `engine`: Execution engine (`EngineAbstract.AUTO`, `'cudf'`, etc.)

Traversal Directions
--------------------

- **Forward Traversal:** `e_forward(...)`
- **Reverse Traversal:** `e_reverse(...)`
- **Undirected Traversal:** `e_undirected(...)`

Tips and Best Practices
-----------------------

- **Limit hops for performance:** Specify `hops` to control traversal depth.
- **Use naming for analysis:** Apply `name` to label and filter results.
- **Combine filters:** Use `filter_dict` and `query` for precise matching.
- **Leverage GPU acceleration:** Use `engine='cudf'` for large datasets.
- **Avoid infinite loops:** Be cautious with `to_fixed_point=True` in cyclic graphs.

Examples at a Glance
--------------------

- **Find all paths between two nodes:**

  .. code-block:: python

      g.chain([
          n({g._node: "Alice"}),
          e_undirected(hops=3),
          n({g._node: "Bob"})
      ])

- **Match nodes with IDs in a range:**

  .. code-block:: python

      n(query="100 <= id <= 200")

- **Traverse edges with specific labels:**

  .. code-block:: python

      e_forward({"label": is_in(["knows", "likes"])})

- **Identify subgraphs based on attributes:**

  .. code-block:: python

      g.chain([
          n({"community": "A"}),
          e_undirected(hops=2),
          n({"community": "B"}, name="bridge_nodes")
      ])

- **Custom edge and node queries:**

  .. code-block:: python

      g.chain([
          n(query="age >= 18"),
          e_forward(edge_query="interaction == 'message'"),
          n(query="location == 'NYC'")
      ])

