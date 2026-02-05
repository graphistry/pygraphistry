.. _api-gfql-hop:

GFQL Hop Matcher
==================

Hop is the core primitive behind a single matcher step in chain.

Calling hop directly has performance benefits over calling chain so may be helpful for larger graphs.

For cross-step constraints, use `Chain(..., where=[...])`; see :doc:`/gfql/where`.

.. automodule:: graphistry.compute.hop
   :members:
   :undoc-members:
   :show-inheritance:
