.. _gfql-builtin-calls:

GFQL Built-in Call Reference
============================

The Call operation in GFQL provides access to a curated set of graph algorithms, transformations, and visualization methods. All methods are validated through a safelist to ensure security and stability.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Call operations are invoked using the ``call()`` function within GFQL chains or Let bindings, or using typed builders for better IDE support:

.. code-block:: python

    from graphistry import call, let, ref, n, e_forward, gt

    # Pure call() chains work - filter then enrich
    result = g.gfql([
        call('filter_nodes_by_dict', {'filter_dict': {'type': 'person'}}),
        call('get_degrees', {'col': 'degree'})
    ])

    # For filter->enrich->filter patterns, use let()
    result = g.gfql(let({
        'persons': n({'type': 'person'}),
        'with_degrees': ref('persons', [call('get_degrees', {'col': 'degree'})]),
        'high_degree': ref('with_degrees', [n({'degree': gt(10)})]),
        'connected': ref('high_degree', [e_forward(), n()])
    }))

All Call operations:

- Validate parameters against type and value constraints
- Return a modified graph (immutable - original is unchanged)
- Can add columns to nodes or edges (schema effects)
- Are restricted to methods in the safelist for security

Graph Transformation Methods
----------------------------

hypergraph
~~~~~~~~~~

Transform event data into entity relationships by connecting entities that appear together in events. This is useful for converting event-based data (logs, transactions, activities) into entity-entity graphs.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - entity_types
     - list[string]
     - No
     - Column names to use as entity types. If None, uses all columns
   * - opts
     - dict
     - No
     - Configuration options for hypergraph transformation (see below)
   * - drop_na
     - boolean
     - No
     - Whether to drop rows with NA values in entity columns (default: True)
   * - drop_edge_attrs
     - boolean
     - No
     - Whether to drop non-entity attributes from edges (default: True)
   * - verbose
     - boolean
     - No
     - Whether to print verbose output during transformation (default: False)
   * - direct
     - boolean
     - No
     - If True, creates direct entity-to-entity edges. If False, keeps hypernodes to show event connections (default: True)
   * - engine
     - string
     - No
     - Processing engine - 'pandas', 'cudf' (GPU), 'dask' (streaming), or 'auto' (default: 'auto')
   * - npartitions
     - integer
     - No
     - Number of partitions for Dask processing
   * - chunksize
     - integer
     - No
     - Chunk size for streaming processing
   * - from_edges
     - boolean
     - No
     - If True, use edges dataframe as input instead of nodes dataframe (default: False)
   * - return_as
     - string
     - No
     - What to return from hypergraph result: 'graph' (default), 'all', 'entities', 'events', 'edges', 'nodes'

**The opts Parameter:**

The ``opts`` dictionary configures advanced hypergraph behavior by controlling how entities are identified and connected. All keys are optional and the dictionary structure is validated to ensure type safety:

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Key
     - Type
     - Description
   * - TITLE
     - string
     - Node title field name (default: 'nodeTitle')
   * - DELIM
     - string
     - Delimiter for composite IDs (default: '::')
   * - NODEID
     - string
     - Node ID field name (default: 'nodeID')
   * - ATTRIBID
     - string
     - Attribute ID field name (default: 'attribID')
   * - EVENTID
     - string
     - Event ID field name (default: 'EventID')
   * - EVENTTYPE
     - string
     - Event type field name (default: 'event')
   * - SOURCE
     - string
     - Source node field name for edges (default: 'src')
   * - DESTINATION
     - string
     - Destination node field name for edges (default: 'dst')
   * - CATEGORY
     - string
     - Category field name (default: 'category')
   * - NODETYPE
     - string
     - Node type field name (default: 'type')
   * - EDGETYPE
     - string
     - Edge type field name (default: 'edgeType')
   * - NULLVAL
     - string
     - Value representing null (default: 'null')
   * - SKIP
     - list[string]
     - Column names to exclude from entity extraction. Each item must be a string
   * - CATEGORIES
     - dict[str, list[str]]
     - Maps category names to lists of values for grouping. Keys must be strings, values must be lists of strings
   * - EDGES
     - dict[str, list[str]]
     - Defines which entity types can connect to each other. Keys represent source entity types (strings), values are lists of target entity types (strings) that the source can connect to

**Examples:**

.. code-block:: python

    # Transform user-product interactions into entity graph
    events_df = pd.DataFrame({
        'user': ['alice', 'bob', 'alice'],
        'product': ['laptop', 'phone', 'tablet'],
        'timestamp': [1, 2, 3]
    })
    g = graphistry.nodes(events_df)

    # Simple transformation using typed builder (recommended)
    hg = g.gfql(hypergraph(entity_types=['user', 'product']))

    # Or using call() directly
    hg = g.gfql(call('hypergraph', {'entity_types': ['user', 'product']}))

    # Keep hypernodes to show event connections
    hg = g.gfql(hypergraph(
        entity_types=['user', 'product'],
        direct=False  # Keep hypernodes
    ))

    # Use GPU acceleration
    hg = g.gfql(hypergraph(
        entity_types=['user', 'product'],
        engine='cudf'
    ))

    # Advanced opts configuration with CATEGORIES and EDGES
    hg = g.gfql(hypergraph(
        entity_types=['user', 'product', 'category'],
        opts={
            'TITLE': 'Entity Graph',
            'SKIP': ['timestamp', 'metadata'],  # Exclude these columns
            'CATEGORIES': {
                'user_type': ['premium', 'regular', 'trial'],
                'product_type': ['electronics', 'clothing', 'books']
            },
            'EDGES': {
                'user': ['product', 'category'],  # Users connect to products and categories
                'product': ['user', 'category'],  # Products connect back to users and categories
                'category': ['product']           # Categories only connect to products
            }
        }
    ))

    # In a DAG with other operations
    from graphistry import let, ref, n

    result = g.gfql(let({
        'hg': hypergraph(entity_types=['user', 'product']),
        'filtered': ref('hg', [n({'type': 'user'})])
    }))

    # Use edges dataframe as input
    edges_df = pd.DataFrame({
        'src_user': ['alice', 'bob', 'alice'],
        'dst_item': ['laptop', 'phone', 'tablet']
    })
    g = graphistry.edges(edges_df, 'src_user', 'dst_item')

    hg = g.gfql(hypergraph(
        from_edges=True,
        entity_types=['src_user', 'dst_item']
    ))

    # Extract only entities dataframe (not full graph)
    entities_df = g.gfql(hypergraph(
        entity_types=['user', 'product'],
        return_as='entities'  # Returns DataFrame instead of Plottable
    ))

    # Extract edges only
    edges_df = g.gfql(hypergraph(
        entity_types=['user', 'product'],
        return_as='edges'
    ))

    # Combine both parameters
    entity_nodes = g.gfql(hypergraph(
        from_edges=True,
        entity_types=['src_user', 'dst_item'],
        return_as='entities'
    ))

**Use Cases:**

- **Social Network Analysis**: Transform interaction events (messages, calls) into social graphs
- **Fraud Detection**: Connect accounts, merchants, and devices from transaction events
- **Security Analysis**: Link users, IPs, and resources from access logs
- **Supply Chain**: Connect suppliers, products, and customers from order events

**Schema Effects:**

Creates a new graph structure where:

- Nodes represent unique entities from the specified columns
- Edges connect entities that appeared in the same event
- Edge attributes can include event metadata (if drop_edge_attrs=False)

**Return Value:**

By default (``return_as='graph'``), returns a Plottable graph object for method chaining. The ``return_as`` parameter controls what is returned:

- ``'graph'``: Plottable graph (default) - enables chaining like ``.plot()``
- ``'all'``: Dict with all 5 components (graph, entities, events, edges, nodes) - backward compatible with module-level ``graphistry.hypergraph()``
- ``'entities'``: DataFrame of entity nodes only
- ``'events'``: DataFrame of event/hypernode nodes only
- ``'edges'``: DataFrame of edges only
- ``'nodes'``: DataFrame of all nodes (entities + events)

.. note::
   Hypergraph transformations cannot be mixed with other operations in chains. Use as a single operation or within Let/DAG constructs for complex compositions.

.. note::
   For large datasets, consider using engine='cudf' for GPU acceleration or engine='dask' for streaming processing.

Graph Analysis Methods
----------------------

compute_cugraph
~~~~~~~~~~~~~~~

Run GPU-accelerated graph algorithms using `cuGraph <https://github.com/rapidsai/cugraph>`_, part of the `NVIDIA RAPIDS <https://rapids.ai/>`_ ecosystem.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - alg
     - string
     - Yes
     - Algorithm name (see supported algorithms below)
   * - out_col
     - string
     - No
     - Output column name (defaults to algorithm name)
   * - params
     - dict
     - No
     - Algorithm-specific parameters
   * - kind
     - string
     - No
     - Graph type hints
   * - directed
     - boolean
     - No
     - Whether to treat graph as directed
   * - G
     - None
     - No
     - Reserved (must be None if provided)

**Supported Algorithms:**

- **pagerank**: PageRank centrality
- **louvain**: Community detection
- **betweenness_centrality**: Betweenness centrality
- **eigenvector_centrality**: Eigenvector centrality
- **katz_centrality**: Katz centrality
- **hits**: HITS (hubs and authorities)
- **bfs**: Breadth-first search
- **sssp**: Single-source shortest path
- **connected_components**: Find connected components
- **strongly_connected_components**: Find strongly connected components
- **k_core**: K-core decomposition
- **triangle_count**: Count triangles per node

**Examples:**

.. code-block:: python

    # PageRank with custom parameters
    g.gfql([
        call('compute_cugraph', {
            'alg': 'pagerank',
            'out_col': 'pr_score',
            'params': {'alpha': 0.85, 'max_iter': 100}
        })
    ])
    
    # Community detection
    g.gfql([
        call('compute_cugraph', {
            'alg': 'louvain',
            'out_col': 'community'
        })
    ])
    
    # Betweenness centrality
    g.gfql([
        call('compute_cugraph', {
            'alg': 'betweenness_centrality',
            'out_col': 'betweenness',
            'directed': True
        })
    ])

**Schema Effects:** Adds one column to nodes with the algorithm result.

**Parameter Discovery:** For detailed algorithm parameters, see the `cuGraph documentation <https://docs.rapids.ai/api/cugraph/stable/>`_. Parameters are passed via the ``params`` dictionary.

.. note::
   For workloads taking 5 seconds to 5 hours on CPU, consider using :ref:`gfql-remote` to offload computation to a GPU-enabled server.

compute_igraph
~~~~~~~~~~~~~~

Run CPU-based graph algorithms using `igraph <https://igraph.org/>`_, the comprehensive network analysis library.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - alg
     - string
     - Yes
     - Algorithm name (see supported algorithms below)
   * - out_col
     - string
     - No
     - Output column name (defaults to algorithm name)
   * - params
     - dict
     - No
     - Algorithm-specific parameters
   * - directed
     - boolean
     - No
     - Whether to treat graph as directed
   * - use_vids
     - boolean
     - No
     - Whether to use vertex IDs

**Supported Algorithms:**

Similar to cuGraph but on CPU, including:

- **pagerank**: PageRank centrality
- **community_multilevel**: Louvain community detection
- **betweenness**: Betweenness centrality
- **closeness**: Closeness centrality
- **eigenvector_centrality**: Eigenvector centrality
- **authority_score**: Authority scores (HITS)
- **hub_score**: Hub scores (HITS)
- **coreness**: K-core values
- **clusters**: Connected components
- **maximal_cliques**: Find maximal cliques
- **shortest_paths**: Compute shortest paths

**Examples:**

.. code-block:: python

    # PageRank using igraph
    g.gfql([
        call('compute_igraph', {
            'alg': 'pagerank',
            'out_col': 'pagerank',
            'params': {'damping': 0.85}
        })
    ])
    
    # Community detection
    g.gfql([
        call('compute_igraph', {
            'alg': 'community_multilevel',
            'out_col': 'community'
        })
    ])

**Schema Effects:** Adds one column to nodes with the algorithm result.

**Parameter Discovery:** For detailed algorithm parameters, see the `Python igraph documentation <https://igraph.org/python/>`_. Parameters are passed via the ``params`` dictionary.

.. note::
   For graphs with millions of edges, consider using ``compute_cugraph`` with a GPU for 10-50x speedup, or :ref:`gfql-remote` if no local GPU is available.

get_degrees
~~~~~~~~~~~

Calculate degree centrality for nodes (in-degree, out-degree, and total degree).

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - col
     - string
     - No
     - Column name for total degree
   * - col_in
     - string
     - No
     - Column name for in-degree
   * - col_out
     - string
     - No
     - Column name for out-degree

**Examples:**

.. code-block:: python

    # Calculate all degree types
    g.gfql([
        call('get_degrees', {
            'col': 'total_degree',
            'col_in': 'in_degree',
            'col_out': 'out_degree'
        })
    ])
    
    # Calculate only total degree
    g.gfql([
        call('get_degrees', {'col': 'degree'})
    ])
    
    # Filter by degree using let()
    from graphistry import let, ref, call, n, gt

    g.gfql(let({
        'with_degrees': call('get_degrees', {'col': 'degree'}),
        'filtered': ref('with_degrees', [n({'degree': gt(10)})])
    }))

**Schema Effects:** Adds up to 3 columns to nodes (based on parameters provided).

get_indegrees
~~~~~~~~~~~~~

Calculate only in-degree for nodes.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - col
     - string
     - No
     - Column name for in-degree (default: 'in_degree')

**Example:**

.. code-block:: python

    g.gfql([
        call('get_indegrees', {'col': 'incoming_connections'})
    ])

**Schema Effects:** Adds one column to nodes.

get_outdegrees
~~~~~~~~~~~~~~

Calculate only out-degree for nodes.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - col
     - string
     - No
     - Column name for out-degree (default: 'out_degree')

**Example:**

.. code-block:: python

    g.gfql([
        call('get_outdegrees', {'col': 'outgoing_connections'})
    ])

**Schema Effects:** Adds one column to nodes.

get_topological_levels
~~~~~~~~~~~~~~~~~~~~~~

Compute topological levels for directed acyclic graphs (DAGs).

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - level_col
     - string
     - No
     - Column name for level (default: 'level')
   * - allow_cycles
     - boolean
     - No
     - Whether to allow cycles (default: True)

**Example:**

.. code-block:: python

    # Compute DAG levels
    g.gfql([
        call('get_topological_levels', {
            'level_col': 'topo_level',
            'allow_cycles': False
        })
    ])

**Schema Effects:** Adds one column to nodes.

Layout Methods
--------------

layout_cugraph
~~~~~~~~~~~~~~

Compute GPU-accelerated graph layouts.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - layout
     - string
     - No
     - Layout algorithm (default: 'force_atlas2')
   * - params
     - dict
     - No
     - Layout-specific parameters
   * - kind
     - string
     - No
     - Graph type hints
   * - directed
     - boolean
     - No
     - Whether to treat graph as directed
   * - bind_position
     - boolean
     - No
     - Whether to bind positions to nodes
   * - x_out_col
     - string
     - No
     - X coordinate column name
   * - y_out_col
     - string
     - No
     - Y coordinate column name
   * - play
     - integer
     - No
     - Animation frames

**Supported Layouts:**

- **force_atlas2**: Force-directed layout

**Example:**

.. code-block:: python

    g.gfql([
        call('layout_cugraph', {
            'layout': 'force_atlas2',
            'params': {
                'iterations': 500,
                'outbound_attraction_distribution': True,
                'edge_weight_influence': 1.0
            }
        })
    ])

**Schema Effects:** Modifies node positions or adds position columns.

layout_igraph
~~~~~~~~~~~~~

Compute CPU-based graph layouts using igraph.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - layout
     - string
     - No
     - Layout algorithm name
   * - params
     - dict
     - No
     - Layout-specific parameters
   * - directed
     - boolean
     - No
     - Whether to treat graph as directed
   * - use_vids
     - boolean
     - No
     - Whether to use vertex IDs
   * - bind_position
     - boolean
     - No
     - Whether to bind positions
   * - x_out_col
     - string
     - No
     - X coordinate column name
   * - y_out_col
     - string
     - No
     - Y coordinate column name
   * - play
     - integer
     - No
     - Animation frames

**Supported Layouts:**

- **kamada_kawai**: Kamada-Kawai layout
- **fruchterman_reingold**: Fruchterman-Reingold force-directed
- **circle**: Circular layout
- **grid**: Grid layout
- **random**: Random layout
- **drl**: Distributed Recursive Layout
- **lgl**: Large Graph Layout
- **graphopt**: GraphOpt layout
- Many more...

**Example:**

.. code-block:: python

    g.gfql([
        call('layout_igraph', {
            'layout': 'fruchterman_reingold',
            'params': {'iterations': 500}
        })
    ])

**Schema Effects:** Modifies node positions or adds position columns.

layout_graphviz
~~~~~~~~~~~~~~~

Compute layouts using Graphviz algorithms.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - prog
     - string
     - No
     - Graphviz program (default: 'dot')
   * - args
     - string
     - No
     - Additional Graphviz arguments
   * - directed
     - boolean
     - No
     - Whether graph is directed
   * - bind_position
     - boolean
     - No
     - Whether to bind positions
   * - x_out_col
     - string
     - No
     - X coordinate column name
   * - y_out_col
     - string
     - No
     - Y coordinate column name
   * - play
     - integer
     - No
     - Animation frames

**Supported Programs:**

- **dot**: Hierarchical layout
- **neato**: Spring model layout
- **fdp**: Force-directed layout
- **sfdp**: Scalable force-directed
- **circo**: Circular layout
- **twopi**: Radial layout

**Example:**

.. code-block:: python

    # Hierarchical layout
    g.gfql([
        call('layout_graphviz', {
            'prog': 'dot',
            'directed': True
        })
    ])
    
    # Circular layout
    g.gfql([
        call('layout_graphviz', {'prog': 'circo'})
    ])

**Schema Effects:** Modifies node positions or adds position columns.

fa2_layout
~~~~~~~~~~

Apply ForceAtlas2 layout algorithm (CPU-based implementation).

.. note::
   This is a CPU-based ForceAtlas2 implementation. For GPU acceleration, use ``call('layout_cugraph', {'layout': 'force_atlas2'})`` instead.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - fa2_params
     - dict
     - No
     - ForceAtlas2 parameters

**Example:**

.. code-block:: python

    g.gfql([
        call('fa2_layout', {
            'fa2_params': {
                'iterations': 1000,
                'gravity': 1.0,
                'scaling_ratio': 2.0
            }
        })
    ])

**Schema Effects:** Modifies node positions.

group_in_a_box_layout
~~~~~~~~~~~~~~~~~~~~~

Apply group-in-a-box layout that organizes nodes into rectangular regions by community.

PyGraphistry's implementation is optimized for large graphs on both CPU and GPU.

**References:**
- Paper: `Group-in-a-box Layout for Multi-faceted Analysis of Communities <https://www.cs.umd.edu/users/ben/papers/Rodrigues2011Group.pdf>`_
- Blog post: `GPU Group-In-A-Box Layout for Larger Social Media Investigations <https://www.graphistry.com/blog/gpu-group-in-a-box-layout-for-larger-social-media-investigations>`_

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - partition_alg
     - string
     - No
     - Community detection algorithm (e.g., 'louvain')
   * - partition_params
     - dict
     - No
     - Parameters for partition algorithm
   * - layout_alg
     - string/callable
     - No
     - Layout algorithm for each box
   * - layout_params
     - dict
     - No
     - Parameters for layout algorithm
   * - x
     - number
     - No
     - X coordinate of bounding box
   * - y
     - number
     - No
     - Y coordinate of bounding box
   * - w
     - number
     - No
     - Width of bounding box
   * - h
     - number
     - No
     - Height of bounding box
   * - encode_colors
     - boolean
     - No
     - Whether to encode communities as colors
   * - colors
     - list[string]
     - No
     - List of colors for communities
   * - partition_key
     - string
     - No
     - Existing column to use as partition
   * - engine
     - string
     - No
     - Engine ('auto', 'cpu', 'gpu', 'pandas', 'cudf')

**Examples:**

.. code-block:: python

    # Basic usage - auto-detect communities
    g.gfql([
        call('group_in_a_box_layout')
    ])
    
    # Use specific partition algorithm
    g.gfql([
        call('group_in_a_box_layout', {
            'partition_alg': 'louvain',
            'engine': 'cpu'
        })
    ])
    
    # Use existing partition column
    g.gfql([
        call('group_in_a_box_layout', {
            'partition_key': 'department',
            'encode_colors': True
        })
    ])
    
    # Full control over layout
    g.gfql([
        call('group_in_a_box_layout', {
            'partition_alg': 'louvain',
            'layout_alg': 'force_atlas2',
            'x': 0, 'y': 0, 'w': 1000, 'h': 1000,
            'colors': ['#ff0000', '#00ff00', '#0000ff']
        })
    ])

**Schema Effects:** Modifies node positions and optionally adds color encoding.

Filtering and Transformation Methods
------------------------------------

filter_nodes_by_dict
~~~~~~~~~~~~~~~~~~~~

Filter nodes based on attribute values.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - filter_dict
     - dict
     - Yes
     - Dictionary of attribute: value pairs to match

**Examples:**

.. code-block:: python

    # Filter by single attribute
    g.gfql([
        call('filter_nodes_by_dict', {
            'filter_dict': {'type': 'person'}
        })
    ])
    
    # Filter by multiple attributes
    g.gfql([
        call('filter_nodes_by_dict', {
            'filter_dict': {'type': 'server', 'status': 'active'}
        })
    ])

**Schema Effects:** None (only filters existing data).

filter_edges_by_dict
~~~~~~~~~~~~~~~~~~~~

Filter edges based on attribute values.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - filter_dict
     - dict
     - Yes
     - Dictionary of attribute: value pairs to match

**Example:**

.. code-block:: python

    g.gfql([
        call('filter_edges_by_dict', {
            'filter_dict': {'weight': 1.0, 'type': 'strong'}
        })
    ])

**Schema Effects:** None (only filters existing data).

hop
~~~

Traverse the graph N steps from current nodes.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - hops
     - integer
     - No*
     - Number of hops (required unless to_fixed_point=True)
   * - to_fixed_point
     - boolean
     - No
     - Traverse until no new nodes found
   * - direction
     - string
     - No
     - 'forward', 'reverse', or 'undirected'
   * - edge_match
     - dict
     - No
     - Filter edges during traversal
   * - source_node_match
     - dict
     - No
     - Filter source nodes
   * - destination_node_match
     - dict
     - No
     - Filter destination nodes
   * - source_node_query
     - string
     - No
     - Query string for source nodes
   * - edge_query
     - string
     - No
     - Query string for edges
   * - destination_node_query
     - string
     - No
     - Query string for destination nodes
   * - return_as_wave_front
     - boolean
     - No
     - Return only new nodes from last hop

**Examples:**

.. code-block:: python

    # Simple N-hop traversal
    g.gfql([
        n({'id': 'start'}),
        call('hop', {'hops': 2, 'direction': 'forward'})
    ])
    
    # Traverse to fixed point
    g.gfql([
        n({'infected': True}),
        call('hop', {
            'to_fixed_point': True,
            'direction': 'undirected'
        })
    ])
    
    # Filtered traversal
    g.gfql([
        n({'type': 'server'}),
        call('hop', {
            'hops': 3,
            'edge_match': {'protocol': 'ssh'},
            'destination_node_match': {'status': 'active'}
        })
    ])

**Schema Effects:** None (returns subgraph).

collapse
~~~~~~~~

Merge nodes based on a shared attribute value.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Column to group nodes by
   * - attribute_columns
     - list[string]
     - No
     - Columns to aggregate
   * - col_aggregations
     - dict
     - No
     - Aggregation functions per column
   * - self_edges
     - boolean
     - No
     - Whether to keep self-edges

**Example:**

.. code-block:: python

    # Collapse by department
    g.gfql([
        call('collapse', {
            'column': 'department',
            'self_edges': False
        })
    ])

**Schema Effects:** Modifies node structure based on collapse.

drop_nodes
~~~~~~~~~~

Remove nodes based on a column value.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Boolean column indicating nodes to drop

**Example:**

.. code-block:: python

    # Mark and drop nodes
    g.gfql([
        n({'status': 'inactive'}, name='to_remove'),
        call('drop_nodes', {'column': 'to_remove'})
    ])

**Schema Effects:** None (only removes nodes).

keep_nodes
~~~~~~~~~~

Keep only nodes where a column is True.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Boolean column indicating nodes to keep

**Example:**

.. code-block:: python

    # Mark and keep nodes
    g.gfql([
        n({'importance': gt(0.5)}, name='important'),
        call('keep_nodes', {'column': 'important'})
    ])

**Schema Effects:** None (only filters nodes).

materialize_nodes
~~~~~~~~~~~~~~~~~

Generate a node table from edges when only edges are provided.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - reuse
     - boolean
     - No
     - Whether to reuse existing node table

**Example:**

.. code-block:: python

    # Create nodes from edges
    g_edges_only.gfql([
        call('materialize_nodes')
    ])

**Schema Effects:** Creates node table if missing.

prune_self_edges
~~~~~~~~~~~~~~~~

Remove edges where source equals destination.

**Parameters:** None

**Example:**

.. code-block:: python

    g.gfql([
        call('prune_self_edges')
    ])

**Schema Effects:** None (only removes edges).

Visual Encoding Methods
-----------------------

encode_point_color
~~~~~~~~~~~~~~~~~~

Map node attributes to colors.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Column to encode as color
   * - palette
     - list
     - No
     - Color palette
   * - as_continuous
     - boolean
     - No
     - Treat as continuous scale
   * - as_categorical
     - boolean
     - No
     - Treat as categorical
   * - categorical_mapping
     - dict
     - No
     - Explicit value-to-color mapping
   * - default_mapping
     - string/int
     - No
     - Default color for unmapped values

**Example:**

.. code-block:: python

    # Categorical color mapping
    g.gfql([
        call('encode_point_color', {
            'column': 'department',
            'categorical_mapping': {
                'sales': 'blue',
                'engineering': 'green',
                'marketing': 'red'
            }
        })
    ])
    
    # Continuous color scale
    g.gfql([
        call('encode_point_color', {
            'column': 'risk_score',
            'palette': ['green', 'yellow', 'red'],
            'as_continuous': True
        })
    ])

**Schema Effects:** Adds color encoding column.

encode_edge_color
~~~~~~~~~~~~~~~~~

Map edge attributes to colors.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Column to encode as color
   * - palette
     - list
     - No
     - Color palette
   * - as_continuous
     - boolean
     - No
     - Treat as continuous scale
   * - as_categorical
     - boolean
     - No
     - Treat as categorical
   * - categorical_mapping
     - dict
     - No
     - Explicit value-to-color mapping
   * - default_mapping
     - string/int
     - No
     - Default color for unmapped values

**Example:**

.. code-block:: python

    g.gfql([
        call('encode_edge_color', {
            'column': 'relationship_type',
            'categorical_mapping': {
                'friend': 'blue',
                'colleague': 'green',
                'family': 'purple'
            }
        })
    ])

**Schema Effects:** Adds color encoding column to edges.

encode_point_size
~~~~~~~~~~~~~~~~~

Map node attributes to sizes.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Column to encode as size
   * - categorical_mapping
     - dict
     - No
     - Value-to-size mapping
   * - default_mapping
     - number
     - No
     - Default size

**Example:**

.. code-block:: python

    g.gfql([
        call('encode_point_size', {
            'column': 'importance',
            'categorical_mapping': {
                'low': 10,
                'medium': 20,
                'high': 40
            }
        })
    ])

**Schema Effects:** Adds size encoding column.

encode_point_icon
~~~~~~~~~~~~~~~~~

Map node attributes to icons.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - column
     - string
     - Yes
     - Column to encode as icon
   * - categorical_mapping
     - dict
     - No
     - Value-to-icon mapping
   * - default_mapping
     - string
     - No
     - Default icon

**Example:**

.. code-block:: python

    g.gfql([
        call('encode_point_icon', {
            'column': 'device_type',
            'categorical_mapping': {
                'server': 'server',
                'laptop': 'laptop',
                'phone': 'mobile'
            }
        })
    ])

**Schema Effects:** Adds icon encoding column.

Utility Methods
---------------

name
~~~~

Set the visualization name.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - name
     - string
     - Yes
     - Name for the visualization

**Example:**

.. code-block:: python

    g.gfql([
        call('name', {'name': 'Network Analysis Results'})
    ])

**Schema Effects:** None (sets metadata).

description
~~~~~~~~~~~

Set the visualization description.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - description
     - string
     - Yes
     - Description text

**Example:**

.. code-block:: python

    g.gfql([
        call('description', {
            'description': 'PageRank analysis of social network'
        })
    ])

**Schema Effects:** None (sets metadata).

Error Handling
--------------

Call operations validate all parameters and will raise specific errors:

.. code-block:: python

    from graphistry.compute.exceptions import GFQLTypeError, ErrorCode
    
    try:
        # Wrong: function not in safelist
        g.gfql([call('invalid_function')])
    except GFQLTypeError as e:
        print(f"Error {e.code}: {e.message}")  # E303: Function not in safelist
    
    try:
        # Wrong: missing required parameter
        g.gfql([call('filter_nodes_by_dict')])
    except GFQLTypeError as e:
        print(f"Error {e.code}: {e.message}")  # E105: Missing required parameter
    
    try:
        # Wrong: invalid parameter type
        g.gfql([call('hop', {'hops': 'two'})])
    except GFQLTypeError as e:
        print(f"Error {e.code}: {e.message}")  # E201: Type mismatch

Common Error Codes:

- **E303**: Function not in safelist
- **E105**: Missing required parameter
- **E201**: Parameter type mismatch
- **E303**: Unknown parameter
- **E301**: Required column not found (runtime)

Best Practices
--------------

1. **Use Specific Algorithms**: Instead of generic "pagerank", use the appropriate compute method:

   .. code-block:: python

       # Good: Explicit algorithm selection
       call('compute_cugraph', {'alg': 'pagerank'})  # GPU
       call('compute_igraph', {'alg': 'pagerank'})   # CPU
       
       # Bad: Non-existent generic method
       call('pagerank')  # ERROR: Not in safelist

2. **Filter Early**: Place filtering operations early in chains:

   .. code-block:: python

       # Good: Filter before expensive operations
       g.gfql([
           call('filter_nodes_by_dict', {'filter_dict': {'active': True}}),
           call('compute_cugraph', {'alg': 'pagerank'})
       ])

3. **Name Output Columns**: Use descriptive column names:

   .. code-block:: python

       # Good: Clear column naming
       call('compute_cugraph', {
           'alg': 'louvain',
           'out_col': 'community_id'
       })

4. **Check Schema Effects**: Be aware of columns added by operations:

   .. code-block:: python

       # After get_degrees, these columns exist - use let() for mixed operations:
       from graphistry import let, ref, call, n, gt

       g.gfql(let({
           'enriched': call('get_degrees', {
               'col': 'total',
               'col_in': 'incoming',
               'col_out': 'outgoing'
           }),
           'filtered': ref('enriched', [n({'total': gt(10)})])  # Filter on degree
       }))

See Also
--------

- :ref:`gfql-quick` - GFQL quick reference
- :ref:`gfql-specifications` - Complete GFQL specification  
- :ref:`gfql-predicates-quick` - Predicate reference for filtering
