Compute Modules
---------------

.. toctree::
   :maxdepth: 2

   graphistry.compute.predicates

ComputeMixin module
------------------------------------------------

.. automodule:: graphistry.compute.ComputeMixin
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Chain
---------------


The `Chain` module in Graphistry allows users to run Cypher-style graph queries directly on dataframes. This feature can be used without accessing a database or Java, and it supports optional GPU acceleration for enhanced performance.

Example Usage:

.. code-block:: python

    from graphistry import n, e_undirected, is_in

    # Define a graph query chain
    g2 = g1.chain([
      n({'user': 'Biden'}),
      e_undirected(),
      n(name='bridge'),
      e_undirected(),
      n({'user': is_in(['Trump', 'Obama'])})
    ])

    # Display the result
    print('# bridges', len(g2._nodes[g2._nodes.bridge]))
    g2.plot()

This example demonstrates a graph query that identifies connections between specific users and nodes labeled as 'bridge'. The `chain` function is used to define a series of node and edge patterns that the graph must match.

To enable GPU acceleration for faster processing:

.. code-block:: python

    # Switch to RAPIDS GPU dataframes for performance
    import cudf
    g2 = g1.edges(lambda g: cudf.DataFrame(g._edges))

    # Utilize the chain function with GPU acceleration
    g3 = g2.chain([n(), e(hops=3), n()])
    g3.plot()

In this example, the `chain` function is used with GPU-accelerated dataframes, demonstrating how Graphistry can efficiently process large-scale graph data.

.. automodule:: graphistry.compute.chain
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Cluster
---------------
.. automodule:: graphistry.compute.cluster
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Collapse
---------------
.. automodule:: graphistry.compute.collapse
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Conditional
---------------
.. automodule:: graphistry.compute.conditional
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Filter by Dictionary
--------------------
.. automodule:: graphistry.compute.filter_by_dict
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Hop
---------------
.. automodule:: graphistry.compute.hop
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

predicates
---------------
.. automodule:: graphistry.compute.predicates
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:
