.. _gfql-builtin-calls:

GFQL Built-in Calls
===================

The Call operation in GFQL allows you to invoke methods on the current graph with validated parameters. This provides a safe and extensible way to apply graph algorithms, transformations, and analytics operations within your GFQL queries.

Basic Usage
-----------

**Call Syntax**

.. code-block:: python

    from graphistry import call
    
    # In a GFQL chain
    g.gfql([
        n({"type": "person"}),
        call('pagerank'),
        n(query='pagerank > 0.5')
    ])
    
    # In a Let binding
    g.gfql(let({
        'influential': [n(), call('pagerank')],
        'top_nodes': [ref('influential'), n(query='pagerank > 0.5')]
    }))

Available Methods
-----------------

The following methods are available through the Call operation:

Graph Algorithms
~~~~~~~~~~~~~~~~

**pagerank**
    Compute PageRank scores for nodes.
    
    .. code-block:: python
    
        call('pagerank')
        call('pagerank', {'damping': 0.85, 'iterations': 20})

**get_degrees**
    Calculate node degrees (in-degree, out-degree, or total).
    
    .. code-block:: python
    
        call('get_degrees')
        call('get_degrees', {'col': 'total_degree'})
        call('get_degrees', {'col_in': 'in_deg', 'col_out': 'out_deg'})

Filtering Operations
~~~~~~~~~~~~~~~~~~~~

**filter_nodes_by_dict**
    Filter nodes based on attribute values.
    
    .. code-block:: python
    
        call('filter_nodes_by_dict', {'filter_dict': {'type': 'user', 'active': True}})

**filter_edges_by_dict**
    Filter edges based on attribute values.
    
    .. code-block:: python
    
        call('filter_edges_by_dict', {'filter_dict': {'weight': gt(0.5)}})

Graph Traversal
~~~~~~~~~~~~~~~

**hop**
    Traverse the graph for N steps from current nodes.
    
    .. code-block:: python
    
        call('hop', {'hops': 2})
        call('hop', {'hops': 3, 'direction': 'forward'})
        call('hop', {'to_fixed_point': True, 'direction': 'undirected'})

Layout Algorithms
~~~~~~~~~~~~~~~~~

**umap**
    Apply UMAP dimensionality reduction for graph layout.
    
    .. code-block:: python
    
        call('umap')
        call('umap', {'n_neighbors': 15, 'min_dist': 0.1})

**fa2_layout**
    Apply ForceAtlas2 layout algorithm.
    
    .. code-block:: python
    
        call('fa2_layout')
        call('fa2_layout', {'iterations': 500})

Graph Structure
~~~~~~~~~~~~~~~

**materialize_nodes**
    Generate node table from edges.
    
    .. code-block:: python
    
        call('materialize_nodes')
        call('materialize_nodes', {'reuse': False})

**add_graph**
    Combine with another graph.
    
    .. code-block:: python
    
        call('add_graph', {'g2': other_graph})

**prune_self_edges**
    Remove self-referencing edges.
    
    .. code-block:: python
    
        call('prune_self_edges')

Utility Operations
~~~~~~~~~~~~~~~~~~

**name**
    Tag nodes with a boolean column.
    
    .. code-block:: python
    
        call('name', {'name': 'important_nodes'})

**sample**
    Sample a subset of nodes.
    
    .. code-block:: python
    
        call('sample', {'n': 1000})

Parameter Validation
--------------------

All Call operations have their parameters validated against a safelist to ensure:

- Type safety: Parameters must be of the correct type
- Required parameters: Missing required parameters will raise an error
- Unknown parameters: Extra parameters not in the safelist will be rejected
- Value constraints: Some parameters have specific allowed values

Example error handling:

.. code-block:: python

    # Missing required parameter
    call('filter_nodes_by_dict')  # Error: Missing 'filter_dict'
    
    # Wrong parameter type
    call('hop', {'hops': 'two'})  # Error: 'hops' must be integer
    
    # Unknown parameter
    call('pagerank', {'unknown_param': 123})  # Error: Unknown parameter

Integration with Other GFQL Features
------------------------------------

Calls can be combined with other GFQL operations:

**With Predicates**

.. code-block:: python

    g.gfql([
        n({'type': 'user'}),
        call('pagerank'),
        n({'pagerank': gt(0.1)})
    ])

**With Let Bindings**

.. code-block:: python

    g.gfql(let({
        'users': n({'type': 'user'}),
        'ranked': [ref('users'), call('pagerank')],
        'top': [ref('ranked'), n(query='pagerank > 0.5')]
    }))

**With Remote Execution**

.. code-block:: python

    g.gfql_remote([
        n(),
        call('pagerank'),
        n(query='pagerank > 0.1')
    ])

Best Practices
--------------

1. **Chain Efficiency**: Place filtering calls early in the chain to reduce data volume
2. **Parameter Reuse**: Store common parameter sets in variables
3. **Error Handling**: Wrap calls in try-except blocks when parameters come from user input
4. **Performance**: Some calls like 'pagerank' are computationally intensive - consider using GPU engine

GPU Acceleration
----------------

Many Call operations support GPU acceleration when using the cuDF engine:

.. code-block:: python

    # Force GPU execution
    g.gfql([
        n(),
        call('pagerank'),
        n(query='pagerank > 0.1')
    ], engine='cudf')

GPU-accelerated methods include:
- pagerank
- get_degrees
- hop
- filter operations
- most graph algorithms

See Also
--------

- :ref:`gfql-quick` - Quick reference for all GFQL operations
- :ref:`gfql-spec` - Complete GFQL specification
- :ref:`10min-gfql` - Tutorial introduction to GFQL