"""Engine-agnostic graph algorithm kernels, exposed as GFQL `graphistry.std.*`.

These exist because no backend implements the LDBC Graphalytics semantics:
cuGraph has no label propagation and no maximal independent set, igraph's label
propagation is randomized rather than deterministic, and both PageRanks converge
to a tolerance while LDBC fixes the iteration count.

They are `std` in the same sense as a standard library: no optional third-party
dependency, and they run on whatever engine the frames already use (pandas or
cudf, from one implementation).

Validated against independent references -- networkx for WCC, Dijkstra for SSSP
(bitwise exact), a naive Python LDBC implementation for CDLP, a hand-written
power iteration for PageRank, and independence+maximality invariants for MIS.
"""
from .kernels import ConvergenceError, cdlp, make_weights, mis, pagerank, sssp, wcc

__all__ = [
    "ConvergenceError",
    "cdlp",
    "make_weights",
    "mis",
    "pagerank",
    "sssp",
    "wcc",
]
