.. _hyper-api:

Hypergraphs
==============

Hypergraphs are graphs where edges may connect more than two nodes, such as an event involving multiple entities.

Graphistry encodes hypergraphs as regular graphs of two forms. One is a bipartite graph between hypernodes and regular nodes connected by hyperedges. The other is regular nodes connected by hyperedges. In both cases, each hyperedge is encoded by multiple regular src/dst edges.

.. toctree::
   :maxdepth: 2


Hypergraph 
-------------


.. autodata:: graphistry.PlotterBase.PlotterBase.hypergraph
    :noindex:

hypergraph
  Primary alias for function :func:`graphistry.hyper_dask.hypergraph`.


.. automodule:: graphistry.hyper.Hypergraph
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:

.. automodule:: graphistry.hyper_dask
    :members:
    :undoc-members:
    :inherited-members:
    :show-inheritance:


