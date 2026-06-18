.. _networkx:


NetworkX
--------

The following methods are provided for converting and managing NetworkX graph
data within PyGraphistry. ``g.compute_networkx(...)`` exposes the same
curated NetworkX algorithm subset as GFQL local Cypher ``CALL graphistry.nx.*``:
``pagerank``, ``betweenness_centrality``, ``degree_centrality``,
``closeness_centrality``, ``eigenvector_centrality``, ``katz_centrality``,
``connected_components``, ``strongly_connected_components``, ``core_number``,
``hits``, ``edge_betweenness_centrality``, and ``k_core``.

Install with ``pygraphistry[networkx]`` or ``pygraphistry[networkx-scipy]``.
For an executable notebook walkthrough, see :doc:`/demos/demos_databases_apis/networkx/networkx`.

Node algorithm example:

.. code-block:: python

    g2 = g.compute_networkx("degree_centrality", out_col="degree_score")
    assert "degree_score" in g2._nodes.columns

Edge algorithm example:

.. code-block:: python

    g2 = g.compute_networkx("edge_betweenness_centrality", out_col="edge_bc", directed=False)
    assert "edge_bc" in g2._edges.columns

Graph-returning algorithm example:

.. code-block:: python

    g2 = g.compute_networkx("k_core", params={"k": 2}, directed=False)

Result shape:

- Node algorithms append one or more columns to ``g._nodes``.
- Edge algorithms append one column to ``g._edges``.
- ``k_core`` returns a projected PyGraphistry graph.
- ``hits`` writes ``hubs`` and ``authorities`` and does not accept ``out_col``.
- ``connected_components`` uses weak components when ``directed=True`` and connected components when ``directed=False``.

.. autofunction:: graphistry.PlotterBase.PlotterBase.compute_networkx
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.from_networkx
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx2pandas
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx_checkoverlap
    :noindex:
