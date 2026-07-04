"""Polars hop entry point — the single place both ``.hop()`` dispatch and ``chain_polars``
go through (also the #1658 seeded-index hook site).

The unified ``hop_polars`` (hop_eager.py) runs a single bounded hop as ONE lazy
collect-once plan (the GPU path) and everything else (multi-hop / min_hops /
to_fixed_point) on the eager BFS loop — formerly a separate lazy twin lived in this
module; its setup/gates/epilogue sections are now shared inside ``hop_polars``.
"""
from typing import Any, Optional

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.lazy.engine.polars.hop_eager import hop_polars


def hop_lazy_or_eager(self: Plottable, nodes: Optional[Any] = None, hops: Optional[int] = 1, **kwargs: Any) -> Plottable:
    """Run the polars hop: lazy collect-once for a single bounded hop, eager loop otherwise."""
    return hop_polars(self, nodes, hops, **kwargs)
