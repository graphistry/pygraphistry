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
        'connected': ref('with_degrees', [n({'degree': gt(10)}), e_forward(), n()])
    }))

All Call operations:

- Validate parameters against type and value constraints
- Return a modified graph (immutable - original is unchanged)
- Can add columns to nodes or edges (schema effects)
- Are restricted to methods in the safelist for security

Call operations stay in graph state: the result remains a traversable graph with meaningful `_edges`, so you can keep matching or compose additional graph stages with `let()` / `ref()`. If you want row/tabular output, switch into row-pipeline operators such as `rows()`, `with_()`, `select()`, `return_()`, `group_by()`, or use a row-returning local Cypher `CALL ... YIELD ... RETURN ...` query.

Layout-aware consumers can detect layout helper calls without maintaining their
own string lists:

.. code-block:: python

    from graphistry.compute import call
    from graphistry.compute.gfql import (
        LAYOUT_FUNCTION_NAMES,
        RADIAL_LAYOUT_FUNCTION_NAMES,
        RADIAL_LAYOUT_KINDS,
        is_layout_chain,
        is_layout_kind,
    )

    chain = [call('time_ring_layout', {'time_col': 'ts'})]
    kind = is_layout_kind(chain)

    assert is_layout_chain(chain)
    assert kind == 'time_ring'
    assert kind in RADIAL_LAYOUT_KINDS
    assert 'time_ring_layout' in RADIAL_LAYOUT_FUNCTION_NAMES
    assert 'group_in_a_box_layout' in LAYOUT_FUNCTION_NAMES
    assert 'tree_layout' in LAYOUT_FUNCTION_NAMES

The helpers accept materialized GFQL ``Chain`` objects, list/tuple chains, Chain
wire dictionaries, and direct ``Call`` objects/dicts. Opaque strings are not
parsed, so pass native GFQL structures rather than substring-matching query
text. For legitimate pre-parse string diagnostics, use
``LAYOUT_FUNCTION_NAMES`` or ``RADIAL_LAYOUT_FUNCTION_NAMES`` as the canonical
safelisted function-name registries instead of maintaining local copies.
These registries reflect GFQL ``call()`` safelist names. Legacy ``radial_*``
names are not safelisted pygraphistry function names; normalize them before
calling these helpers.

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
     - Event ID field name (default: 'EventID'). When omitted and the source contains that column, it is used as the event key with a warning; rename the source column to retain it as an attribute and generate row IDs instead.
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

The exact procedure names mirror ``graphistry.plugins.cugraph.compute_algs``.
Current categories include:

- **Node-enriching:** ``betweenness_centrality``, ``bfs``, ``bfs_edges``, ``connected_components``, ``core_number``, ``ecg``, ``hits``, ``katz_centrality``, ``leiden``, ``louvain``, ``pagerank``, ``shortest_path``, ``shortest_path_length``, ``spectralBalancedCutClustering``, ``spectralModularityMaximizationClustering``, ``sssp``, ``strongly_connected_components``
- **Edge-enriching:** ``batched_ego_graphs``, ``edge_betweenness_centrality``, ``jaccard``, ``jaccard_w``, ``overlap``, ``overlap_coefficient``, ``overlap_w``, ``sorensen``, ``sorensen_coefficient``, ``sorensen_w``
- **Topology-returning:** ``ego_graph``, ``k_core``, ``minimum_spanning_tree``

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

    # Weakly connected components (WCC) labels
    g.gfql([
        call('compute_cugraph', {
            'alg': 'connected_components',
            'out_col': 'wcc_id',
            'directed': False
        })
    ])

    # Strongly connected components (SCC) labels
    g.gfql([
        call('compute_cugraph', {
            'alg': 'strongly_connected_components',
            'out_col': 'scc_id',
            'directed': True
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

**Schema Effects:** Depends on the algorithm family. Node algorithms add node columns, edge algorithms add edge columns, and topology-returning algorithms return a new graph topology.

**Local Cypher Modes:**

- **Procedure naming:** ``CALL graphistry.cugraph.<alg>()`` and ``CALL graphistry.cugraph.<alg>.write()`` mirror ``compute_cugraph(alg=...)`` for the supported algorithm names above.
- **Row mode for node algorithms:** ``g.gfql("CALL graphistry.cugraph.louvain()")`` returns row state with ``nodeId`` plus the default algorithm output columns in ``_nodes`` and an empty placeholder ``_edges`` frame (for example, ``assert result._edges.empty``).
- **Row mode for edge algorithms:** ``g.gfql("CALL graphistry.cugraph.edge_betweenness_centrality()")`` returns row state with ``source``, ``destination``, and the edge result columns in ``_nodes`` while leaving ``_edges`` empty.
- **Graph mode / topology mode:** ``g.gfql("CALL graphistry.cugraph.edge_betweenness_centrality.write()")`` enriches the graph in place and keeps traversable edges (for example, ``assert not result._edges.empty``). Topology-returning algorithms such as ``k_core`` and ``minimum_spanning_tree`` require ``.write()``.
- **Options map:** Local Cypher procedures accept one optional map argument. ``out_col``, ``directed``, ``kind``, and ``params`` mirror ``compute_cugraph()`` directly, and any extra keys are forwarded into the nested algorithm ``params`` dictionary. For example, ``CALL graphistry.cugraph.louvain({resolution: 1.0})`` maps to ``compute_cugraph('louvain', params={'resolution': 1.0})``.

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

The exact procedure names mirror ``graphistry.plugins.igraph.compute_algs``.
Current supported names include:

- ``articulation_points``, ``authority_score``, ``betweenness``, ``bibcoupling``, ``closeness``, ``clusters``, ``cocitation``
- ``community_edge_betweenness``, ``community_fastgreedy``, ``community_infomap``, ``community_label_propagation``, ``community_leading_eigenvector``, ``community_leiden``, ``community_multilevel``, ``community_optimal_modularity``, ``community_spinglass``, ``community_walktrap``
- ``constraint``, ``coreness``, ``eccentricity``, ``eigenvector_centrality``, ``harmonic_centrality``, ``hub_score``, ``k_core``, ``pagerank``, ``personalized_pagerank``
- Topology-returning procedures: ``gomory_hu_tree`` and ``spanning_tree``

**Examples:**

.. doc-test: xfail

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

    # Weakly connected components (WCC) labels
    g.compute_igraph('clusters', out_col='wcc_id', params={'mode': 'weak'})

    # Strongly connected components (SCC) labels
    g.compute_igraph('clusters', out_col='scc_id', params={'mode': 'strong'})

**Schema Effects:** Most algorithms add one node column. Topology-returning algorithms such as ``gomory_hu_tree`` and ``spanning_tree`` return a new graph topology instead.

**Local Cypher Modes:**

- **Procedure naming:** ``CALL graphistry.igraph.<alg>()`` and ``CALL graphistry.igraph.<alg>.write()`` mirror ``compute_igraph(alg=...)`` for the supported algorithm names above.
- **Row mode:** ``g.gfql("CALL graphistry.igraph.pagerank()")`` returns row state with ``nodeId`` plus the default algorithm output column in ``_nodes`` and an empty placeholder ``_edges`` frame (for example, ``assert result._edges.empty``).
- **Graph mode / topology mode:** ``g.gfql("CALL graphistry.igraph.pagerank.write()")`` keeps the result in graph state with traversable edges (for example, ``assert not result._edges.empty``). Topology-returning algorithms such as ``spanning_tree`` and ``gomory_hu_tree`` require ``.write()``.
- **Options map:** Local Cypher procedures accept one optional map argument. ``out_col``, ``directed``, ``use_vids``, and ``params`` mirror ``compute_igraph()`` directly, and any extra keys are forwarded into the nested algorithm ``params`` dictionary. For example, ``CALL graphistry.igraph.pagerank({damping: 0.9, directed: false})`` maps to ``compute_igraph('pagerank', directed=False, params={'damping': 0.9})``.
- **NetworkX local subset:** The local Cypher compiler also supports a CPU ``graphistry.nx.*`` subset for parity with common cuGraph-style row outputs:

  - Node-enriching calls: ``CALL graphistry.nx.pagerank()`` / ``.write()``, ``CALL graphistry.nx.betweenness_centrality()`` / ``.write()``, ``CALL graphistry.nx.degree_centrality()`` / ``.write()``, ``CALL graphistry.nx.closeness_centrality()`` / ``.write()``, ``CALL graphistry.nx.eigenvector_centrality()`` / ``.write()``, ``CALL graphistry.nx.katz_centrality()`` / ``.write()``, ``CALL graphistry.nx.connected_components()`` / ``.write()``, ``CALL graphistry.nx.strongly_connected_components()`` / ``.write()``, ``CALL graphistry.nx.core_number()`` / ``.write()``, and ``CALL graphistry.nx.hits()`` / ``.write()``
  - Edge-enriching calls: ``CALL graphistry.nx.edge_betweenness_centrality()`` / ``.write()``
  - Topology-returning calls: ``CALL graphistry.nx.k_core.write()``

  They follow the same row-vs-``.write()`` contract as the other backends: node calls use ``nodeId`` + value column rows, edge calls use ``source`` / ``destination`` + value column rows, and topology-returning calls require ``.write()``. Multi-output ``hits`` returns ``nodeId``, ``hubs``, and ``authorities`` and does not accept ``out_col``.

  The same curated NetworkX algorithm subset is available from regular Python
  as ``g.compute_networkx(...)`` for users who do not need the local Cypher
  ``CALL`` path.

**Parameter Discovery:** For detailed algorithm parameters, see the `Python igraph documentation <https://igraph.org/python/>`_. Parameters are passed via the ``params`` dictionary.

.. note::
   Component IDs (for example, ``wcc_id`` / ``scc_id``) are partition labels and
   may differ numerically across backends or runs. Use them for grouping and
   filtering by component membership, not as stable semantic IDs.

.. note::
   For graphs with millions of edges, consider using ``compute_cugraph`` with a GPU, or :ref:`gfql-remote` if no local GPU is available.

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
   * - degree_in
     - string
     - No
     - Column name for in-degree
   * - degree_out
     - string
     - No
     - Column name for out-degree

**Examples:**

.. code-block:: python

    # Calculate all degree types
    g.gfql([
        call('get_degrees', {
            'col': 'total_degree',
            'degree_in': 'in_degree',
            'degree_out': 'out_degree'
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

**Local Cypher Modes:**

- **Row mode:** ``g.gfql("CALL graphistry.degree()")`` returns row state with default ``nodeId``, ``degree``, ``degree_in``, and ``degree_out`` columns in ``_nodes`` and an empty placeholder ``_edges`` frame (for example, ``assert result._edges.empty``). Add ``YIELD ... RETURN ...`` when you want to project or sort those rows explicitly.
- **Graph mode:** ``g.gfql("CALL graphistry.degree.write()")`` materializes ``degree``, ``degree_in``, and ``degree_out`` on nodes while preserving the graph for later matches with traversable edges (for example, ``assert not result._edges.empty``).

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

.. doc-test: xfail

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
     - Yes
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

.. doc-test: xfail

.. code-block:: python

    g.gfql([
        call('layout_igraph', {
            'layout': 'fruchterman_reingold',
            'params': {'niter': 500}
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

.. doc-test: xfail

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

.. doc-test: xfail

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

circle_layout
~~~~~~~~~~~~~

Arrange nodes in a circular layout.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - bounding_box
     - list[number]
     - No
     - ``[cx, cy, width, height]`` bounding box. If omitted, the graph must already have ``x``/``y`` node positions.
   * - ring_spacing
     - number
     - No
     - Spacing between rings
   * - point_spacing
     - number
     - No
     - Spacing between points
   * - partition_by
     - string or list[string]
     - No
     - Node column(s) for partitioned circles
   * - sort_by
     - string or list[string]
     - No
     - Accepted but currently has no effect; circle order is always by node id
   * - ascending
     - boolean or list[boolean]
     - No
     - Accepted but currently has no effect
   * - na_position
     - string
     - No
     - ``'first'`` or ``'last'``; accepted but currently has no effect
   * - ignore_index
     - boolean
     - No
     - Accepted but currently has no effect
   * - engine
     - string
     - No
     - Engine (``'auto'``, ``'pandas'``, ``'cudf'``, ``'dask'``, ``'dask_cudf'``)

**Example:**

.. code-block:: python

    g.gfql([
        call('circle_layout', {
            'bounding_box': [0, 0, 1000, 1000],
            'engine': 'pandas'
        })
    ])

**Schema Effects:** Modifies node positions.

tree_layout
~~~~~~~~~~~

Apply the Sugiyama-style tree layout.

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
     - Output level column
   * - level_sort_values_by
     - string or list[string]
     - No
     - Column(s) for ordering nodes within levels
   * - level_sort_values_by_ascending
     - boolean
     - No
     - Sort direction for level ordering
   * - width
     - number
     - No
     - Layout width multiplier
   * - height
     - number
     - No
     - Layout height multiplier
   * - rotate
     - number
     - No
     - Rotation in degrees
   * - allow_cycles
     - boolean
     - No
     - Whether to tolerate cyclic inputs
   * - root
     - string/number/boolean
     - No
     - Optional root node id

**Example:**

.. code-block:: python

    g.gfql([
        call('tree_layout', {'level_col': 'level'})
    ])

**Schema Effects:** Modifies node positions and adds a level column.

mercator_layout
~~~~~~~~~~~~~~~

Project latitude/longitude node columns into local Mercator coordinates.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - scale_for_graphistry
     - boolean
     - No
     - Use scaled coordinates suitable for Graphistry visualization

**Example:**

.. code-block:: python

    geo_nodes = pd.DataFrame({
        'id': ['nyc', 'sf'],
        'latitude': [40.7128, 37.7749],
        'longitude': [-74.0060, -122.4194],
    })

    graphistry.nodes(geo_nodes, 'id').gfql([
        call('mercator_layout')
    ])

**Schema Effects:** Modifies node positions.

modularity_weighted_layout
~~~~~~~~~~~~~~~~~~~~~~~~~~

Prepare a community-weighted layout by weighting edges within and across communities.

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - community_col
     - string
     - No
     - Existing node community column
   * - community_alg
     - string
     - No
     - Community algorithm when ``community_col`` is omitted
   * - community_params
     - dict
     - No
     - Community algorithm parameters
   * - same_community_weight
     - number
     - No
     - Edge weight for within-community edges
   * - cross_community_weight
     - number
     - No
     - Edge weight for cross-community edges
   * - edge_influence
     - number
     - No
     - Graphistry edge influence setting
   * - engine
     - string
     - No
     - Engine (``'auto'``, ``'pandas'``, ``'cudf'``)

**Example:**

.. code-block:: python

    g.gfql([
        call('modularity_weighted_layout', {'community_col': 'department'})
    ])

**Schema Effects:** Adds edge weight columns and binds edge weight. May add a community column when ``community_col`` is omitted.

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

.. doc-test: xfail

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
   * - nodes
     - list or dict
     - Yes
     - Node IDs to drop (list) or filter specification (dict)

**Example:**

.. code-block:: python

    # Drop specific nodes by ID
    g.gfql([
        call('drop_nodes', {'nodes': ['node_id_1', 'node_id_2']})
    ])

    # Drop nodes matching a filter — use filter_nodes_by_dict first, then drop
    inactive = g._nodes[g._nodes['status'] == 'inactive']['id'].tolist()
    g.gfql([
        call('drop_nodes', {'nodes': inactive})
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
   * - nodes
     - list or dict
     - Yes
     - Node IDs to keep (list) or filter specification (dict)

**Example:**

.. doc-test: xfail

.. code-block:: python

    # Keep specific nodes by ID
    g.gfql([
        call('keep_nodes', {'nodes': ['node_id_1', 'node_id_2']})
    ])

    # Keep nodes matching a filter — use dict form for column-based filtering
    g.gfql([
        call('keep_nodes', {'nodes': {'importance': [True]}})
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
    g_edges_only = graphistry.edges(edges, 's', 'd')
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

.. _gfql-semantic-table-operators:

Semantic Table Operators
------------------------

Narrow a graph with GFQL, then ask a model typed questions about the rows that remain. The answers come back as columns,
next to the data, with calibrated probabilities when the provider can measure them.

Why semantic table operators?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Graph first, model second.** A GFQL chain or Cypher query narrows 100,000 rows to the few hundred that matter before you
  spend one model token. See :doc:`overview` and :doc:`cypher`.
* **Typed, calibrated decisions, not brittle chat prose.** The operators are built around decision models such as the
  `OpenAI Decisions API <https://developers.openai.com/api/docs/guides/decisions>`_. A question is a predicate, a choice or a
  score. The answer has native or token-logprob probabilities, ``min_p`` sends unsure answers to a queue, and the code never makes up
  a confidence. OpenAI-compatible chat hosts, vLLM and SGLang grammars, Anthropic and LangChain work too.
* **Guardrails are built in.** ``dedupe`` asks once for repeated rows. A ``SemanticCache`` answers repeated input between runs.
  ``budget`` and ``deadline_s`` stop a run. ``explain_sem_decide`` plans a run without a provider call, and ``semantic_trace()``
  reports what a run did.

Quickstart: input, config, output
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The callable provider plays the model, so these examples run offline. Replace it with a real provider (see `Providers`_).
All three forms give the same rows.

*A Plottable method on a GFQL result:*

.. code-block:: python

    from graphistry import n
    from graphistry.compute.semantic import CallableDecisionProvider, semantic_summary, semantic_trace

    provider = CallableDecisionProvider(lambda evidence, q: evidence["type"] == "company", fingerprint="rules-v1")
    questions = [{"id": "is_company", "kind": "predicate", "text": "Is this a company?"}]          # config

    usa = g.gfql([n({"country": "USA"})])                                                          # input: a GFQL subgraph
    with semantic_trace() as runs:
        out = usa.sem_decide(questions, provider, task_id="quickstart", evidence_cols=["type"])
    semantic_summary(out._nodes)                                                                   # output: status counts
    runs[-1]["provider_calls"], runs[-1]["billed_cost_usd"]                                        # output: what the run did

*A GFQL* ``call(...)`` *(JSON-serializable: the provider is named by an alias):*

.. code-block:: python

    from graphistry import call, n
    from graphistry.compute.semantic import MemorySemanticCache, bind_providers

    with bind_providers({"rules": provider}, cache=MemorySemanticCache()):
        out = g.gfql([n({"country": "USA"}),
                      call("sem_decide", {"questions": questions, "provider": "rules",
                                          "task_id": "quickstart", "evidence_cols": ["type"]})])
    out._nodes[["id", "is_company", "is_company__status"]]

*A Cypher query, then* ``sem_decide`` *(alias the returned columns so that they are plain names):*

.. code-block:: python

    rows = g.gfql("MATCH (p {country: 'USA'}) RETURN p.id AS id, p.type AS type")
    out = rows.sem_decide(questions, provider, task_id="quickstart", evidence_cols=["type"])
    out._nodes[["id", "is_company"]]

A native Cypher ``CALL`` procedure for the operators is on the roadmap. Until then, use ``g.gfql("MATCH ...").sem_decide(...)``
or ``call("sem_decide", ...)``. These calls run locally only: ``gfql_remote()`` refuses them before any network request.
Every input row is kept, in order and with its index, and a provider failure becomes a per-row ``error`` result, not an exception.

Function contracts
~~~~~~~~~~~~~~~~~~

.. code-block:: text

    g.sem_decide(questions, provider, *, task_id, kind="nodes", evidence_cols=None, template=None, id_col=None,
                 task_version="1", batch_size=None, max_concurrency=None, deadline_s=None, budget=None,
                 max_rows=None, min_p=None, require_probabilities=False, column_max_chars=None,
                 overwrite=False, dedupe=True) -> Plottable            # adds q, q__status, q__p, ... per question
    g.sem_map(instruction, out_col, provider, *, task_id, kind="nodes", evidence_cols=None, template=None,
              id_col=None, task_version="1", max_output_tokens=None, batch_size=None, max_concurrency=None,
              deadline_s=None, budget=None, max_rows=None, column_max_chars=None, overwrite=False,
              dedupe=True) -> Plottable                                 # adds out_col, out_col__status, ...
    explain_sem_decide(df, *, questions, provider, task_id, ...)  -> SemanticPlanReport   # the arguments of sem_decide, minus kind and overwrite; no provider call
    explain_sem_map(df, *, instruction, out_col, provider, task_id, ...) -> SemanticPlanReport
    with bind_providers(providers, *, cache=None): ...                  # alias -> provider; optional SemanticCache
    with semantic_trace() as runs: ...                                  # runs: list of SemanticRunStats
    MemorySemanticCache(max_entries=10_000, max_bytes=64 * 2**20, ttl_s=3600.0)   # .stats(), .clear(), .peek_many()
    SemanticCache                                                       # protocol: get_many(keys), put_many(entries), clear()
    semantic_verdicts(df, question_id, *, positive=None) -> Series      # yes | no | uncertain | error | skipped
    semantic_summary(df, question_ids=None) -> DataFrame                # status counts and an "unanswered" column

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Entry point
     - Contract
   * - ``sem_decide``
     - One typed answer for each row and question. ``kind`` is ``"nodes"`` or ``"edges"``. It never raises for a provider fault.
   * - ``sem_map``
     - One generated string for each row. ``budget`` takes ``max_requests`` only.
   * - ``explain_sem_*``
     - What the run would do. It sends no request and no probe. It raises ``ValueError`` for invalid parameters, but not for a collision with an existing output column.
   * - ``bind_providers``
     - Binds provider aliases and an optional cache for the runs in its block (``contextvars``: bind in the calling thread).
   * - ``semantic_trace``
     - Collects one ``SemanticRunStats`` for each run in its block. It costs nothing outside a block.
   * - ``MemorySemanticCache``
     - A thread-safe LRU cache with an entry cap, a byte cap and a time to live. You own the instance. There is no default cache.
   * - ``semantic_verdicts``, ``semantic_summary``
     - Read the status columns so that an ``error`` or ``skipped`` row is never counted as ``no``.

**All three question kinds.** The callable provider plays the model, so the example runs offline. A runnable offline tour of every
operator, provider and limit is in the :doc:`demo notebook </notebooks/gfql>` ``GFQL Semantic Operators``. Four demos apply the operators to
real public data: a software supply chain, network logs, offshore companies and device recalls.

.. code-block:: python

    from graphistry.compute.semantic import CallableDecisionProvider

    questions = [
        {"id": "is_company", "kind": "predicate", "text": "Is this a company?"},
        {"id": "team", "kind": "choice", "text": "Which team owns it?", "options": ["sales", "eng", "other"]},
        {"id": "risk", "kind": "score", "text": "How risky is it?", "levels": ["low", "medium", "high"]},
    ]

    def rules(evidence, question):
        if question.kind == "predicate":
            return evidence["type"] == "company"
        if question.kind == "choice":
            return evidence["department"] if evidence["department"] in question.options else "other"
        return "high" if evidence["risk_score"] > 6 else "medium" if evidence["risk_score"] > 2 else "low"

    g2 = g.sem_decide(questions, CallableDecisionProvider(rules), task_id="triage",
                      evidence_cols=["type", "department", "risk_score"])
    g2._nodes[["id", "is_company", "team", "risk", "risk__status"]]

sem_decide
~~~~~~~~~~

Answer typed questions per row: ``predicate`` (yes/no), ``choice`` (one of ``options``) or ``score`` (one of
``levels``, ordered lowest to highest).

**Parameters:**

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 50

   * - Parameter
     - Type
     - Required
     - Description
   * - questions
     - list[dict]
     - Yes
     - ``{id, kind, text, options?, levels?}``; ids name the output columns and must be unique
   * - provider
     - string
     - Yes
     - Alias bound with ``bind_providers`` (the Python method also accepts a provider object)
   * - task_id
     - string
     - Yes
     - Caller label for the task
   * - evidence_cols
     - list[string]
     - One of these
     - Columns sent as a ``{column: value}`` mapping
   * - template
     - string
     - One of these
     - Text with bare ``{column}`` placeholders, substituted verbatim (for example ``<untrusted>{text}</untrusted>``)
   * - kind
     - string
     - No
     - ``'nodes'`` (default) or ``'edges'``
   * - id_col
     - string
     - No
     - Row id column; default the node/edge id binding, else the index when its labels are unique (also after ``str()``, and non-empty), else row positions
   * - task_version
     - string
     - No
     - Version label included in each result's input hash (default ``'1'``)
   * - batch_size, max_concurrency
     - integer
     - No
     - Rows per provider call, at most the provider's ``max_rows_per_request`` (default: that limit, else 32), and concurrent calls
   * - deadline_s
     - number
     - No
     - Seconds for the whole call; unfinished rows become ``skipped`` (an abandoned request may still finish and be billed)
   * - budget
     - dict
     - No
     - ``{max_cost_usd?, max_requests?}``: ``max_cost_usd`` caps the summed *estimated* cost of the provider calls
       (billed cost can differ; it needs a provider that can estimate cost, and the callable provider estimates 0);
       ``max_requests`` caps the number of provider calls
   * - max_rows
     - integer
     - No
     - Rows at positions >= ``max_rows`` are ``skipped``, never dropped
   * - min_p
     - number
     - No
     - ``ok`` answers whose probability ``q__p`` is below this (or undefined) become ``uncertain``
   * - require_probabilities
     - boolean or ``'native'``
     - No
     - ``False`` (default) preserves current behavior; ``True`` requires measured native or token-logprob
       probabilities for successful values; ``'native'`` requires native-decision probabilities
   * - column_max_chars
     - integer
     - No
     - Truncate each string evidence value
   * - overwrite
     - boolean
     - No
     - Replace existing output columns instead of raising
   * - dedupe
     - boolean
     - No
     - ``True`` (default) asks once for rows that have the same evidence and copies the answer; ``False`` asks for every
       row (see `Repeated inputs`_)

**Output columns** for each question ``q``:

.. list-table::
   :header-rows: 1
   :widths: 24 22 54

   * - Column
     - Type
     - Meaning
   * - ``q``
     - nullable boolean (predicate), string (choice, score)
     - The answer. Null when the status is ``error`` or ``skipped``, or when an ``uncertain`` result has no value. An ``uncertain`` result that fell below ``min_p`` keeps its value, so read ``q__status`` first.
   * - ``q__status``
     - string
     - ``ok``, ``uncertain``, ``error`` or ``skipped``.
   * - ``q__p``
     - float
     - The probability that the distribution gives the answer. Missing when undefined, so test it with ``isna()``.
   * - ``q__dist``
     - JSON string or null
     - The distribution over the labels. It is never made up.
   * - ``q__probability_kind``
     - string
     - ``native_decision``, ``token_logprob``, ``verbalized`` or ``none``.
   * - ``q__error_code``
     - string or null
     - Set when the status is ``error``.
   * - ``q__meta``
     - JSON string
     - Model, provider, task version, input hash, latency, cost, cache hit, retry count, provider confidence, status reason and extra metadata.

An unparseable answer is an error, never a default value. A row whose evidence columns are all null is ``skipped`` with status
reason ``null_evidence``, and the provider is not called for it. ``sem_map`` adds ``out_col``, ``out_col__status``,
``out_col__error_code`` and ``out_col__meta``.

.. list-table:: ``require_probabilities``
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Effect
   * - ``False`` (default)
     - Value-only and ``verbalized`` answers are accepted, with ``min_p`` as it is.
   * - ``True``
     - Accepts ``native_decision`` and ``token_logprob`` distributions that include the returned value.
   * - ``'native'``
     - Accepts only ``native_decision``.

The check runs on each successful answer after result validation and before ``min_p``. A missing required probability gives an
``error`` with code ``probabilities_unavailable`` and clears the answer, the distribution and the provider confidence. Zero is an
available probability. Refusals, ties, other answerless uncertain results, errors and skipped rows keep their verdicts. An uncertain
result that still carries a value must also meet the requirement.

* A known incompatible shipped chat configuration fails locally before any evidence, probe, cost estimate or provider call. Its eligible
  rows get ``probabilities_unavailable`` with zero cost and retries. Null-evidence, ``max_rows`` and deadline skips remain. This check
  comes before the budget plan and uses none of the budget.
* The Anthropic and LangChain adapters are always value-only. OpenAI chat can give token logprobs but not native-decision probabilities.
* An ``auto`` profile that includes logprobs is checked from its actual answers, and its probes count against the run's budget.
* Custom providers and subclasses are checked from their validated results, not from their type name.
* No path turns a generated confidence or a constrained token into a probability.

The limits and probability checks together, on a provider that reports native probabilities. Rows
``a`` and ``b`` are answered. Row ``c`` is ``uncertain`` because its probability, 0.8, is below ``min_p``. Row ``d`` is
``skipped`` by ``max_rows``.

.. code-block:: python

    from graphistry.compute.semantic import CallableDecisionProvider

    def measured(evidence, question):
        p_true = 0.9 if evidence["type"] == "person" else 0.2
        return p_true > 0.5, {"true": p_true, "false": 1 - p_true}

    provider = CallableDecisionProvider(measured, probability_kind="native_decision")
    g3 = g.sem_decide(
        [{"id": "is_person", "kind": "predicate", "text": "Is this a person?"}], provider,
        task_id="people", evidence_cols=["type"], batch_size=1, max_rows=3,
        budget={"max_requests": 3}, min_p=0.85, require_probabilities=True)
    g3._nodes[["id", "is_person", "is_person__p", "is_person__status"]]

**Engines:** pandas and cuDF run natively. A polars graph run through ``g.gfql([call("sem_decide", ...)])`` uses the
off-engine call bridge: the call runs on pandas and the result returns to polars through Arrow, so GFQL results stay
polars. GFQL warns once for each function in a session (``RuntimeWarning``: "has no native polars implementation").
Called directly as ``g.sem_decide(...)`` on a polars-bound graph, the result tables are pandas.

.. code-block:: python

    import polars as pl
    from graphistry.compute.semantic import CallableDecisionProvider, bind_providers

    g_pl = graphistry.nodes(pl.from_pandas(nodes[["id", "type"]]), "id").edges(pl.from_pandas(edges[["s", "d"]]), "s", "d")
    question = [{"id": "is_company", "kind": "predicate", "text": "Is this a company?"}]
    provider = CallableDecisionProvider(lambda evidence, q: evidence["type"] == "company")

    with bind_providers({"rules": provider}):
        out = g_pl.gfql([call("sem_decide", {"questions": question, "provider": "rules", "task_id": "t",
                                             "evidence_cols": ["type"]})], engine="polars")
    type(out._nodes)        # polars.DataFrame
    direct = g_pl.sem_decide(question, provider, task_id="t", evidence_cols=["type"])
    type(direct._nodes)     # pandas.DataFrame

To refuse the bridge, call ``graphistry.compute.gfql.lazy.set_call_mode('strict')`` or set the environment variable
``GFQL_POLARS_CALL_MODE=strict``. Then ``engine='polars'`` raises ``NotImplementedError`` before any provider call, and
``engine='auto'`` runs the query on pandas. ``set_call_mode(None)`` restores the default.

.. code-block:: python

    from graphistry.compute.gfql.lazy import set_call_mode

    set_call_mode("strict")
    try:
        with bind_providers({"rules": provider}):
            g_pl.gfql([call("sem_decide", {"questions": question, "provider": "rules", "task_id": "t",
                                           "evidence_cols": ["type"]})], engine="polars")
    except NotImplementedError as error:
        print(error)
    finally:
        set_call_mode(None)

Under ``engine='auto'`` (including the polars GPU route for cuDF graphs), a query that already called a provider and
then reaches a step that engine cannot run raises ``GFQLCallExecutionError`` instead of re-running the whole
query on another engine; rerun with ``engine='pandas'`` (or ``'cudf'``).

sem_map
~~~~~~~

Generate one text value per row. Takes ``instruction`` (string, required), ``out_col`` (string, required) and
``max_output_tokens`` (optional; passed to the provider as its bound on each value), plus every other ``sem_decide``
parameter except ``questions``, ``min_p`` and ``require_probabilities`` (``batch_size`` falls back to 16 rather than
32; ``budget`` takes ``max_requests`` only, since generation providers do not estimate cost). Adds ``out_col``,
``{out_col}__status``, ``{out_col}__error_code`` and ``{out_col}__meta`` (no ``confidence`` in the metadata). Engines,
limits, provider-fault handling, remote refusal and the ``engine='auto'`` rule are the same as for ``sem_decide``.

.. code-block:: python

    from graphistry.compute.semantic import CallableGenerationProvider

    writer = CallableGenerationProvider(lambda evidence, instruction: f"{evidence['name']} ({evidence['type']})")
    g4 = g.sem_map("Write a short label.", "label", writer, task_id="labels", evidence_cols=["name", "type"])
    g4._nodes[["id", "label", "label__status"]]

Providers
~~~~~~~~~

.. list-table:: Provider compatibility
   :header-rows: 1
   :widths: 24 12 22 18 12 12

   * - Provider
     - For
     - Probabilities
     - Questions share one call
     - Row packing
     - Extra dependency
   * - ``CallableDecisionProvider``
     - ``sem_decide``
     - What ``fn`` returns (``probability_kind``)
     - No (``fn`` runs for each row and question)
     - No by default (opt in through ``capabilities``)
     - None
   * - ``CallableGenerationProvider``
     - ``sem_map``
     - Not applicable
     - Not applicable
     - No (one row for each call by default)
     - None
   * - ``OpenAIDecisionsProvider``
     - ``sem_decide``
     - ``native_decision``
     - Yes (one request for each row)
     - Opt in: ``rows_per_request``
     - None
   * - ``OpenAIChatProvider``
     - ``sem_decide``
     - ``token_logprob`` when the host returns logprobs, else none
     - No (one question for each request)
     - No
     - None
   * - ``AnthropicMessagesProvider``
     - ``sem_decide``
     - None (value only)
     - No
     - No
     - None
   * - ``LangChainDecisionProvider``
     - ``sem_decide``
     - None (value only)
     - No
     - No
     - None in graphistry (you supply a LangChain chat model, for example from ``langchain-openai``)

Any object that implements the ``DecisionProvider`` or ``GenerationProvider`` protocol in ``graphistry.compute.semantic`` can be bound.
A callable provider runs one row for each call by default, so up to ``concurrency_limit`` rows run at once and the function must be safe to
call from several threads. ``OpenAIDecisionsProvider`` calls the public
`OpenAI Decisions API <https://developers.openai.com/api/docs/guides/decisions>`_ with a caller-supplied model and ``TokenPrices``.

**Chat providers** (``OpenAIChatProvider`` for OpenAI-compatible hosts such as OpenAI, Azure OpenAI, Fireworks and Mistral, and
``AnthropicMessagesProvider``) ask each question with a one-character code for each label.

* With token logprobs, the provider reads a label distribution from the answer token (``probability_kind`` ``token_logprob``,
  conditional on the answer being a code). The share of the answer token's probability that the codes hold is ``extra.label_mass``.
  Without logprobs the answer has no distribution.
* A question whose labels cannot get unambiguous codes is an ``unsupported_question`` error and is not sent.
* The provider also reads a reply that echoes the label of its code (``A. yes``) or that is one label alone (``yes``; a predicate also
  takes ``y`` and ``n``). It ignores a leading ``<think>`` or ``<reasoning>`` block. It reads a reply that reasons first when the last
  line, after a blank line, is such an answer, or when the reply ends with a code after a single ``answer:`` or ``answer is``.
* It does not read a reply that says ``answer:`` or ``answer is`` more than once, whose marker is followed by anything but a code, or that
  gives two different answers. The provider sends such a request again, up to ``reply_retries`` times.
* For a host whose free-text replies cannot be read, ``strategy='json_schema'`` or ``strategy='tool'`` constrains the answer to the codes
  (no distribution; ``extra.answer_strategy`` says which). ``strategy='text'`` sends no logprobs fields.
* ``strategy='auto'`` probes the host once for each endpoint, model and setting set. Each probe request counts against the run's
  budget. The provider keeps the first strategy that works for an hour, and ``graphistry.compute.semantic.forget_probes()`` drops the kept results.
* A reply cut off at ``max_answer_tokens`` before a readable answer is an ``answer_truncated`` error. The default is 256, output tokens are
  not in the cost estimate, so lower the limit to bound them. A reasoning model needs a higher limit, or reasoning turned off through ``extra_body``.
* A request that the host's content filter rejects (HTTP 400, error code ``content_filter``) is ``uncertain`` with reason ``refusal``, like a filtered reply.

**The Decisions provider.** A server that has the OpenAI Decisions API and smaller limits sets ``max_questions_per_request``,
``max_choice_options`` and ``max_levels`` on ``OpenAIDecisionsProvider``. It sends one request for each row by default.
``rows_per_request=k`` opts in to packing up to ``k`` rows into one request. The rows times the questions of one request stay within
``max_questions_per_request``, so each provider call is one HTTP request and counts once in ``budget.max_requests``. Each question then
carries the evidence of its own row, so the answers of a row do not depend on which rows share its request, but they can differ from
one-row answers. Packing cuts the request count and the latency, not the tokens. Packed answers record ``extra.packed_rows``.

Repeated inputs
~~~~~~~~~~~~~~~

Two mechanisms stop the same input from being sent twice.

**Dedupe** works inside one call. After ``max_rows`` and null-evidence skips, and before ``budget`` is planned,
``sem_decide`` and ``sem_map`` group the rows that have the same evidence (the same input hash for every question). Only
the first row of each group goes to the provider. Each other row gets a copy of the answer with ``cost_usd`` 0,
``latency_ms`` 0, ``retry_count`` 0 and ``extra.deduped_from`` set to the id of the row that was asked. The sum of
``cost_usd`` over the output is therefore the money that was spent, and ``budget.max_requests`` counts only real calls.
A failed call fails every row that shares it, so one failure can give an error to many rows. This changes the default:
before ``dedupe``, a table with repeated evidence sent every row. Pass ``dedupe=False`` when you want an answer for every
row, for example to vote over several answers from a model that is not deterministic. A bound result cache (below) still
returns the stored answer for a repeated input, so do not bind a cache for a vote.

**The result cache** works between calls. ``bind_providers(..., cache=cache)`` gives every ``sem_decide`` and
``sem_map`` run inside the block a ``SemanticCache``. There is no default cache and no module-level state: you create the
cache, and an inner block without ``cache`` keeps the outer one. ``MemorySemanticCache`` is a thread-safe LRU cache in
process memory:

.. code-block:: python

    from graphistry.compute.semantic import CallableDecisionProvider, MemorySemanticCache, bind_providers

    cache = MemorySemanticCache(max_entries=50_000, ttl_s=900)       # also max_bytes; defaults 10,000 entries, 64 MiB, 1 hour
    provider = CallableDecisionProvider(lambda evidence, q: "inc" in str(evidence).lower(), fingerprint="company-rules-v1")
    questions = [{"id": "is_company", "kind": "predicate", "text": "Is this a company?"}]
    with bind_providers({"rules": provider}, cache=cache):
        first = g.sem_decide(questions, "rules", task_id="t", evidence_cols=["name"])    # calls the provider
        again = g.sem_decide(questions, "rules", task_id="t", evidence_cols=["name"])    # calls it no more
    cache.stats()          # hits, misses, puts, evictions, expirations, rejected, entries, bytes, hit_rate

A cache key is a hash of the schema version, the provider fingerprint and the input hash. The provider fingerprint comes
from ``provider.cache_fingerprint()``. For the HTTP providers it covers the class, name, model, base URL and the settings
that shape an answer (strategy, ``max_answer_tokens``, ``extra_body``, ``rows_per_request`` and similar). It never covers the
API key or the headers.

- A callable provider has a fingerprint only when you pass ``fingerprint="..."``. Change that text when your function
  changes.
- An ``OpenAIChatProvider`` with ``strategy="auto"`` has none, because a probe settles the strategy after the cache
  lookup. Choose an explicit strategy.
- If a cache is bound and the provider has no fingerprint, the run raises ``ValueError`` before any call.
- The ``sem_map`` key also covers ``max_output_tokens``. The key does not cover ``task_id`` or the row id, so a custom
  provider must not use them to shape an answer.

A key and a stored answer hold no evidence, no API key, no tenant name and no row id. They hold the input hash, which is a
hash of the evidence. A new library version that changes a prompt does not change the key, so clear an external cache
after an upgrade.

What the cache holds, and what a hit looks like:

- It stores each answer after the provider reply is checked and **before** ``min_p``. A later run with another ``min_p``
  still hits. It stores ``ok`` answers and ``uncertain`` answers with status reason ``tie`` or ``refusal``. It never
  stores ``error``, ``skipped``, ``null_answer`` or ``min_p`` results.
- A hit runs through the same checks as a provider reply, and runs before ``budget`` is planned, so it uses none of the
  budget and sends no request. It has ``cost_usd`` 0, ``latency_ms`` 0, ``retry_count`` 0, ``cache_hit`` true and
  ``extra.cache_source`` ``local_cache``. A duplicate row of a hit also has ``extra.deduped_from``. A stored answer that
  fails the checks is a miss.
- ``sem_decide`` serves a row from the cache only when every question of the row hits. A stored answer is assumed to not
  depend on the other questions that share its call.
- A cache that raises, or that returns data of the wrong shape, counts as a miss. A cache never fails a run.

To keep answers in Redis or a database, write a class with ``get_many(keys)`` (a mapping of the keys it holds to JSON
strings), ``put_many(entries)`` and ``clear()``, and bind an instance for each tenant.

Plans, traces and safe verdicts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three tools show what a run will do, what it did, and which rows the model answered.

**A plan** is what a run would do, with no provider call. ``explain_sem_decide`` and ``explain_sem_map`` take the arguments of
``sem_decide`` and ``sem_map`` and return a ``SemanticPlanReport``. The plan is built by the code that the run uses, so it is
what the run does. It sends no answer request and no probe. It reads the provider's limits and cost estimate, and it reads a
bound cache without changing it. It uses ``peek_many`` when the cache has it, as ``MemorySemanticCache`` does. A cache without
``peek_many`` is read with ``get_many``, which can have side effects. A plan does not use ``deadline_s``, and it does not count the
probe requests that ``strategy="auto"`` can send.

**A trace** is what a run did. ``with semantic_trace() as runs:`` gives a list. Each ``sem_decide`` or ``sem_map`` run in the
block, direct or through ``g.gfql([call(...)])``, adds one ``SemanticRunStats`` when the run ends. Blocks nest: a run goes to the
innermost block when it ends, and to each outer block when the inner block ends. The trace follows ``contextvars``
rules, so a thread that you start yourself sees it only when you start it with ``contextvars.copy_context().run``. Outside a block
a run does no tracing work.

**A verdict** says whether the model answered. ``semantic_verdicts`` turns the columns of one question into ``yes``, ``no``,
``uncertain``, ``error`` or ``skipped``. Only an ``ok`` row is ``yes`` or ``no``. ``~out["q"]`` and ``out["q"] == "label"`` both
count an ``error`` or ``skipped`` row as ``no``, and a report that does this shows a row as safe when the model never answered it.
``semantic_summary`` counts the statuses of every question, with an ``unanswered`` column. Both raise ``ValueError`` for a status that
is not one of the four (a null too), so that no row is left out of a count. ``semantic_verdicts`` also raises for a ``positive`` of
the wrong type, and warns when the answers match ``positive`` only if you ignore case and spaces. A cuDF table is read as pandas.

.. code-block:: python

    from graphistry.compute.semantic import (
        CallableDecisionProvider, explain_sem_decide, semantic_summary, semantic_trace, semantic_verdicts)

    provider = CallableDecisionProvider(lambda evidence, q: "inc" in str(evidence).lower(), fingerprint="company-rules-v1")
    questions = [{"id": "is_company", "kind": "predicate", "text": "Is this a company?"}]

    plan = explain_sem_decide(g._nodes, questions=questions, provider=provider, task_id="t", evidence_cols=["name"])
    plan["planned_calls"], plan["dedupe_saved_rows"], plan["estimated_input_cost_usd"]

    with semantic_trace() as runs:
        g2 = g.sem_decide(questions, provider, task_id="t", evidence_cols=["name"])
    runs[0]["provider_calls"], runs[0]["billed_cost_usd"]
    semantic_verdicts(g2._nodes, "is_company").value_counts()
    semantic_summary(g2._nodes)

The fields of ``SemanticPlanReport`` (rows are rows of the table):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``total_rows``, ``null_evidence_rows``, ``max_rows_skipped``, ``active_rows``
     - Rows, rows with no evidence, rows beyond ``max_rows``, and the rest
   * - ``dedupe``, ``unique_rows``, ``dedupe_saved_rows``
     - Whether ``dedupe`` is on, the distinct active rows, and the rows that a copy answers
   * - ``cache_bound``, ``cache_hits``, ``cache_misses``
     - Whether a cache is bound, and the distinct rows that it answers or does not answer (``sem_decide``: every question must hit)
   * - ``provider_rows``
     - The rows that reach the provider
   * - ``effective_batch_size``, ``effective_concurrency``
     - Rows in a call, and the calls that run at once
   * - ``planned_calls``, ``calls_within_budget``, ``planned_questions``
     - Provider calls, the calls that ``budget`` allows, and the answers that are asked for. A provider that is known to lack the
       required probabilities plans no call
   * - ``estimated_input_cost_usd``
     - The estimate of the input cost of the planned calls, or None when the provider cannot estimate (``sem_map`` never does)
   * - ``budget_max_cost_usd``, ``budget_max_requests``, ``would_budget_skip_rows``
     - The limits, and the rows that they would skip. A cost cap on a provider that cannot estimate cost raises ``ValueError``,
       as it does in a run
   * - ``provider``, ``model``, ``strategy``, ``known_probability_gap``
     - Who answers, the configured strategy (None when the provider has none), and whether the provider is known to lack the
       required probabilities

The fields of ``SemanticRunStats``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Meaning
   * - ``total_rows``, ``null_evidence_rows``, ``max_rows_skipped``, ``active_rows``, ``deduped_rows``, ``cache_hits``, ``provider_rows``
     - The funnel, in rows
   * - ``budget_skipped_rows``, ``deadline_skipped_rows``
     - Provider rows that a budget or a deadline stopped
   * - ``status_counts``
     - For each question, the number of rows for each status (``sem_map`` uses the key ``""``)
   * - ``uncertain_reasons``, ``error_codes``, ``probability_kind_counts``
     - Histograms over the answers of all questions
   * - ``provider_calls``, ``probe_calls``, ``retries``
     - Calls of ``decide_batch`` or ``generate_batch`` (a call that was answered locally does not count), setup requests such as host
       probes, and extra requests that the provider made. A provider can send several HTTP requests in one call
   * - ``billed_cost_usd``, ``provider_latency_ms_sum``, ``cached_input_tokens``
     - Summed over the answers that the provider gave. A copy of a duplicate row and a cache hit add nothing, so nothing counts twice.
       ``billed_cost_usd`` is the sum of the ``cost_usd`` of the answers: it does not hold the cost of probe requests
   * - ``estimated_cost_usd``
     - The estimate of the input cost of the planned calls. It is set only when ``budget`` has ``max_cost_usd``, and then the trace calls the
       provider's ``estimate_cost_usd`` once for each planned call (a run without a cost cap makes no estimate, so a trace does not either);
       otherwise None
   * - ``wall_ms``, ``peak_in_flight``, ``strategy_used``
     - Time of the run, the most calls in flight at once, and the strategy that the provider settled on (None when it has none)

Defaults to review
~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 28 20 52

   * - Setting
     - Default
     - Effect
   * - ``max_answer_tokens``
     - 256
     - Output-token limit of a chat answer. A reply cut off before a readable answer is ``answer_truncated``. Output
       tokens are not in the cost estimate, so lower the limit to bound them.
   * - ``reply_retries``
     - 2
     - Extra requests for a reply with no readable answer. Each one is billed. ``0`` sends once.
   * - ``max_retries``, ``timeout_s``
     - 3, 60
     - Transport retries and the timeout of one HTTP request, capped by the run deadline.
   * - ``batch_size``
     - provider limit, else 32 (16 for ``sem_map``)
     - Rows per provider call.
   * - ``rows_per_request``
     - 1
     - Row packing of ``OpenAIDecisionsProvider`` is opt in.
   * - ``dedupe``
     - ``True``
     - Rows with the same evidence are asked once. ``False`` asks for every row.
   * - result cache
     - none
     - Bind one with ``bind_providers(..., cache=...)``. ``MemorySemanticCache`` keeps 10,000 entries or 64 MiB for 1 hour.
   * - Host probe (``strategy='auto'``)
     - once per host, kept for 1 hour
     - Each probe request counts in ``budget.max_requests`` and its estimated cost in ``budget.max_cost_usd``, before
       any row call is planned. A kept result costs nothing.
   * - ``budget.max_cost_usd``
     - none
     - Caps the estimated input cost of the calls. It does not cap output, retry or tracing charges.
   * - ``require_probabilities``
     - ``False``
     - ``True`` accepts only measured probabilities.

Older and self-hosted chat APIs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``OpenAIChatProvider`` also accepts these explicit strategies:

* ``json_mode`` sends ``response_format={"type": "json_object"}`` and validates the answer locally.
* ``function`` sends legacy ``functions`` and a named ``function_call``.
* ``vllm_choice``, ``vllm_regex`` and ``vllm_grammar`` use ``structured_outputs``.
* ``guided_choice``, ``guided_regex`` and ``guided_grammar`` use the legacy vLLM fields.
* ``sglang_regex`` and ``sglang_ebnf`` use SGLang's chat constraint fields.

SGLang and vLLM JSON Schema endpoints can use the existing ``json_schema`` strategy.
The provider compiles each question's answer codes to the selected constraint. Grammar replies
must contain one code. JSON and function replies must contain only the ``answer`` field with
one valid code. Invalid replies become ``invalid_provider_result``; the provider never chooses
a default answer. Some hosts accept a tool, function or JSON request and still answer in plain text:
that reply is an ``unsupported_strategy`` error, so choose a strategy the host implements. A host that
answers HTTP 501 is not asked again: the row returns ``http_501``. These strategies return no probability or distribution, even when the host
includes logprobs. Metadata records ``answer_strategy`` and, for the new strategies, ``semantic_level``.
The semantic level describes the selected mechanism; it does not prove that a host enforced a constraint.
Constraint and logprob fields in ``extra_body`` cannot override the selected strategy.

The OpenAI-compatible, decisions-endpoint and Anthropic providers send a request again when its reply holds
no readable answer or is cut off, up to two more times (``reply_retries=2``; ``0`` turns this off). The last
reply is the result. Every request is billed: ``cost_usd`` and ``latency_ms`` are their sums and ``retry_count``
counts the extra requests. The cost adds the attempts that reported token usage, so it is a lower bound when an attempt
reports none, and it stays on the result when a later attempt fails or the content filter stops it. Like transport
retries, they are not part of the cost estimate or ``budget.max_requests``. A reply that stays unread is an error
result that still reports the billed ``cost_usd``. A refusal, an
``unsupported_strategy`` reply and an HTTP error are not sent again. The LangChain provider never resends; add
retries to the chat model you pass in.

Choose a known mechanism directly to avoid probes. For explicit fallback discovery, configure an
ordered profile. For example:

.. doc-test: skip

.. code-block:: python

    provider = OpenAIChatProvider(
        key, model, base_url="http://localhost:8000/v1",
        strategy="auto", probe_strategies=("vllm_choice", "json_schema", "text"),
        max_tokens_field="max_tokens",
    )

The default profile stays ``logprobs``, ``json_schema``, ``tool``, ``text``. New rungs require an
explicit profile. A profile probes synthetic evidence and caches only a validated reply for its
endpoint, model, settings and ordered profile. Each rung can send up to ``max_retries + 1`` attempts.
Probes use the remaining deadline and the run's budget: in ``sem_decide`` each probe request counts as one of
``budget.max_requests`` and its estimated cost counts toward ``max_cost_usd``, before any row call is planned.
A probe the budget cannot cover is not sent, nothing is kept, and every row is skipped for budget. A kept strategy
costs nothing. A probe that fails on authentication, transport, rate limits or a server error returns that error
for every row of the run without sending user evidence or probing again. HTTP 501 is the exception: like a
rejected request, it moves the probe to the next rung.
A real row's failure is returned without resending it through a lower strategy.
The existing cost estimate includes the selected constraint payload, or the largest profile payload
before discovery. It estimates input cost and is not a bound on billed spend.

Current vLLM uses ``structured_outputs``; ``guided_*`` fields target older servers. Unsupported
fields can be rejected or silently ignored, even with HTTP 200. A probe validates the reply shape;
it does not verify that the server enforced the constraint. Use a deployment-specific profile and
test the chosen mechanism with instructions that conflict with its constraint. Grammar support
depends on the server backend; a model name alone does not establish support.

LangChain chat models
~~~~~~~~~~~~~~~~~~~~~

``LangChainDecisionProvider`` wraps a supplied chat model or Runnable. PyGraphistry does not import
LangChain or add it as a required dependency. Install and configure the model integration separately.

.. doc-test: skip

.. code-block:: python

    from graphistry.compute.semantic import LangChainDecisionProvider, bind_providers

    provider = LangChainDecisionProvider(chat_model, model="deployment", strategy="json_schema")
    with bind_providers({"classifier": provider}):
        result = g.sem_decide(provider="classifier", questions=questions,
                              task_id="classification", evidence_cols=["text"])

Strategies ``json_schema``, ``function_calling`` and ``json_mode`` pass that method to
``with_structured_output(..., include_raw=True)``. ``structured`` uses the model's default structured
method. ``tool`` uses ``bind_tools`` to request one tool named ``answer``; no tool is executed.
``text`` validates a code, its matching label echo or a label alone, as the whole reply or as its last line after a blank line. Each structured answer must contain exactly
one ``answer`` field with a valid code; refusals, unfinished answers and malformed replies do not
become successful values. All strategies return no probabilities, even when the model exposes logprobs.
Metadata records the selected strategy and semantic level.

The default ``auto`` tries structured binding, tool binding, then text locally, before invoking the
model. Only an absent interface or ``NotImplementedError`` permits this selection. Bindings are
cached per provider and answer-code domain. There are no billed probes. A parse or invocation error
never triggers another strategy or resend. Explicit strategies return ``unsupported_strategy`` when
their local binding is unavailable. A method exposed by an integration can still be rejected by its
remote host; select the deployment's supported strategy before starting another run.

The supplied model controls its network timeouts, retries and tracing. Configure these for the run's
budget and data policy. The adapter adds no retries and makes one invocation per question.
``invoke_kwargs`` passes supported invocation settings, including Runnable ``config``. Set
``timeout_kwarg`` only when the integration accepts that timeout keyword; the adapter caps its value
by ``timeout_s`` and the remaining deadline. Generic Runnable config has no portable hard timeout;
GFQL abandons results after its deadline. Each model invocation can contain client-owned HTTP retries.
``budget.max_requests`` counts provider calls, as for other adapters.

Caller-supplied ``TokenPrices`` estimates input and schema cost, or uses ``per_question``. It is not
a bound on output, retry or tracing charges. Actual cost uses normalized ``usage_metadata`` when
available; missing usage or prices leave cost unknown. Unsupported package/provider combinations
must be tested separately. The adapter does not infer capability from a package version or model name.
No-network compatibility smoke checks cover Python 3.12 with ``langchain-core==0.3.49`` /
``langchain-openai==0.3.10`` and ``langchain-core==1.2.15`` / ``langchain-openai==1.1.10``.
They exercise actual structured/tool bindings against an in-memory transport; remote host support
depends on the deployment. The adapter itself and fake-model tests also run on Python 3.8.

Anthropic structured answers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``AnthropicMessagesProvider(strategy='json_schema')`` sends the answer-code schema through
``output_config.format``. ``strategy='tool'`` requests one tool named ``answer`` with the same
schema. Both validate the returned code and return no distribution or probability. Metadata
records ``answer_strategy`` and ``semantic_level``. ``strategy='text'`` remains the default.
These models often write a sentence before the answer code under ``text``. Such a reply is read when it ends
with the code on its own line after a blank line; one cut off at ``max_answer_tokens`` is ``answer_truncated``.
``json_schema`` constrains the reply to the code instead, and the schema adds input tokens to every request.
The selected strategy controls format/tool fields in ``extra_body``; other ``output_config``
settings are retained. The cost estimate includes the selected schema or tool payload.

Forced tool support depends on the model and thinking settings. An unsupported request retains
its HTTP error; the provider does not resend it with another strategy. A host that accepts the request
but answers in plain text, with no tool call or JSON, returns ``unsupported_strategy``. Refusals remain uncertain,
and truncated, paused, malformed or conflicting answers become errors. The provider reads tool
arguments as data and never executes a returned tool call.


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
               'degree_in': 'incoming',
               'degree_out': 'outgoing'
           }),
           'filtered': ref('enriched', [n({'total': gt(10)})])  # Filter on degree
       }))

See Also
--------

- :ref:`gfql-quick` - GFQL quick reference
- :ref:`gfql-specifications` - Complete GFQL specification  
- :ref:`gfql-predicates-quick` - Predicate reference for filtering
