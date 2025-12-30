.. _gfql-remote:

GFQL Remote Mode
====================

You can run GFQL queries and GPU Python remotely, such as when data is already remote, gets big, or you would like to use a remote GPU

Basic Usage
-----------

Run chain remotely and fetch results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import n, e
    g2 = g1.gfql_remote([n(), e(), n()])
    assert len(g2._nodes) <= len(g1._nodes)

.. note::
   Collections are visualization URL settings; apply them after GFQL results
   (for example, ``g2.collections(...)``). The GFQL remote/upload APIs do not
   accept collections payloads yet.

Method :meth:`chain_remote <graphistry.compute.ComputeMixin.ComputeMixin.chain_remote>` runs chain remotely and fetched the computed graph

- **chain**: Sequence of graph node and edge matchers (:class:`ASTObject <graphistry.compute.ast.ASTObject>` instances).
- **output_type**: Defaulting to "all", whether to return the nodes (`'nodes'`), edges (`'edges'`), or both. See :meth:`chain_remote_shape <graphistry.compute.ComputeMixin.ComputeMixin.chain_remote_shape>` to return only metadata.
- **node_col_subset**: Optionally limit which node attributes are returned to an allowlist.
- **edge_col_subset**: Optionally limit which edge attributes are returned to an allowlist.
- **engine**: Optional execution engine. Engine is typically not set, defaulting to `'auto'`. Use `'cudf'` for GPU acceleration and `'pandas'` for CPU.
- **validate**: Defaulting to `True`, whether to validate the query and data.


Manual CPU, GPU engine selection
---------------------------------

By default, GFQL will decide which engine to use based on workload characteristics like the dataset size. You can override this default by specifying which engine to use.

GPU
~~~~~

Run on GPU remotely and fetch results

.. code-block:: python

    from graphistry import n, e
    g2 = g1.gfql_remote([n(), e(), n()], engine='cudf')
    assert len(g2._nodes) <= len(g1._nodes)

CPU
~~~~~~~~~

Run on CPU remotely and fetch results

.. code-block:: python

    from graphistry import n, e
    g2 = g1.gfql_remote([n(), e(), n()], engine='pandas')




Explicit uploads
-----------------

Explicit uploads via :meth:`upload <graphistry.PlotterBase.PlotterBase.upload>` will bind the field `Plottable::dataset_id`, so subsequent remote calls know to skip re-uploading. Always using explicit uploads can make code more predictable for larger codebases.


.. code-block:: python

    from graphistry import n, e
    g2 = g1.upload()
    assert g2._dataset_id is not None, "Uploading sets `dataset_id` for subsequent calls"

    g3a = g2.gfql_remote([n()])
    g3b = g2.gfql_remote([n(), e(), n()])
    assert len(g3a._nodes) >= len(g3b._nodes)


Bind to existing remote data
-------------------------------

If data is already uploaded and your user has access to it, such as from a previous session or shared from another user, you can bind it to a local `Plottable` for remote access.

.. code-block:: python

    import graphistry
    from graphistry import  n, e

    g1 = graphistry.bind(dataset_id='abc123')
    assert g1._nodes is None, "Binding does not fetch data"

    connected_graph_g = g1.gfql_remote([n(), e()])
    connected_nodes_df = connected_graph_g._nodes
    print(connected_nodes_df.shape)


Download less
----------------

You may not need to download all -- or any -- of your results, which can  significantly speed up execution


Return only nodes
~~~~~~~~~~~~~~~~~

.. code-block:: python

  g1.gfql_remote([n(), e(), n()], output_type="nodes")

Return only nodes and specific columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  cols = [g1._node, 'time']
  g2b = g1.gfql_remote(
    [n(), e(), n()],
    output_type="nodes",
    node_col_subset=cols)
  assert len(g2b._nodes.columns) == len(cols)


Return only edges
~~~~~~~~~~~~~~~~~

.. code-block:: python

  g2a = g1.gfql_remote([n(), e(), n()], output_type="edges")

Return only edges and specific columns
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  cols = [g1._source, g1._destination, 'time']
  g2b = g1.gfql_remote([n(), e(), n()],
    output_type="edges",
    edge_col_subset=cols)
  assert len(g2b._edges.columns) == len(cols)

Return metadata but not the actual graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from graphistry import n, e
    shape_df = g1.chain_remote_shape([n(), e(), n()])
    assert len(shape_df) == 2
    print(shape_df)

Remote Python
--------------

You can also run full GPU Python tasks remotely, such as for more complicated code, or if you want the server itself to perform fetching such as from a database.

Run remote python on the current graph
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import graphistry
    from graphistry import n, e

    # Fully self-contained so can be transferred
    def my_remote_trim_graph_task(g):

        # Trick: You can also put database fetch calls here instead of using 'g'!
        return (g
            .nodes(g._nodes[:10])
            .edges(g._edges[:10])
        )

    # Upload any local graph data to the remote server
    g2 = g1.upload()

    g3 = g2.python_remote_g(my_remote_trim_graph_task)

    assert len(g3._nodes) == 10
    assert len(g3._edges) == 10


Run Python on an existing graph, return a table
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  import graphistry

  g = graphistry.bind(dataset_id='ds-abc-123')

  def first_n_edges(g):
      return g._edges[:10]

  some_edges_df = g.python_remote_table(first_n_edges)

  assert len(some_edges_df) == 10


Run Python on an existing graph, return JSON
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

  import graphistry

  g = graphistry.bind(dataset_id='ds-abc-123')

  def first_n_edges_shape(g):
      return {'num_edges': len(g._edges[:10])}

  obj = g.python_remote_json(first_n_edges_shape)

  assert obj['num_edges'] == 10
