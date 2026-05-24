.. _networkx:


NetworkX Methods
----------------

The following methods are provided for converting and managing NetworkX graph
data within PyGraphistry. They are conversion helpers, not a regular
``g.compute_networkx(...)`` algorithm API. GFQL local Cypher has a separate,
curated ``CALL graphistry.nx.*`` algorithm subset; see the GFQL Cypher CALL
documentation for that surface.

.. autofunction:: graphistry.PlotterBase.PlotterBase.from_networkx
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx2pandas
    :noindex:

.. autofunction:: graphistry.PlotterBase.PlotterBase.networkx_checkoverlap
    :noindex:
