.. _networkx:


NetworkX Methods
----------------

The following methods are provided for converting and managing NetworkX graph
data within PyGraphistry. ``g.compute_networkx(...)`` exposes the same
curated NetworkX algorithm subset as GFQL local Cypher ``CALL graphistry.nx.*``:
``pagerank``, ``betweenness_centrality``, ``degree_centrality``,
``closeness_centrality``, ``eigenvector_centrality``, ``katz_centrality``,
``connected_components``, ``strongly_connected_components``, ``core_number``,
``hits``, ``edge_betweenness_centrality``, and ``k_core``.

Install with ``pygraphistry[networkx]`` or ``pygraphistry[networkx-scipy]``.

Example:

.. code-block:: python

    g2 = g.compute_networkx("degree_centrality", out_col="degree_score")

.. autofunction:: graphistry.PlotterBase.PlotterBase.compute_networkx
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.from_networkx
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx2pandas
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx_checkoverlap
    :noindex:
