"""Polars hop entry point — the single place both ``.hop()`` dispatch and ``chain_polars``
go through (also the #1658 seeded-index hook site).
"""
from typing import Any, Optional, Tuple

from graphistry.compute.typing import DataFrameT

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.lazy.engine.polars.dtypes import is_lazy
from graphistry.compute.gfql.lazy.engine.polars.hop_eager import hop_polars


def _with_every_lazy_input_collected(
    self: Plottable, nodes: Optional[DataFrameT],
) -> Tuple[Plottable, Optional[DataFrameT]]:
    """Polars joins refuse to mix a LazyFrame with an eager frame."""
    if self._nodes is not None and is_lazy(self._nodes):
        self = self.nodes(self._nodes.collect(), self._node)
    if self._edges is not None and is_lazy(self._edges):
        self = self.edges(self._edges.collect(), self._source, self._destination)
    if nodes is not None and is_lazy(nodes):
        nodes = nodes.collect()
    return self, nodes


def hop_lazy_or_eager(self: Plottable, nodes: Optional[Any] = None, hops: Optional[int] = 1, **kwargs: Any) -> Plottable:
    """Run the polars hop: lazy collect-once for a single bounded hop, eager loop otherwise."""
    self, nodes = _with_every_lazy_input_collected(self, nodes)
    from graphistry.compute.gfql.index import get_index_policy, get_registry, maybe_index_hop
    _pol = get_index_policy(self)
    if (not get_registry(self).is_empty()) or _pol in ("auto", "force"):
        from graphistry.Engine import Engine
        from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
        _eng = Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else Engine.POLARS
        _indexed = maybe_index_hop(
            self, _eng, nodes=nodes, hops=hops, direction=kwargs.get("direction", "forward"),
            return_as_wave_front=kwargs.get("return_as_wave_front", False),
            to_fixed_point=kwargs.get("to_fixed_point", False), policy=_pol,
            min_hops=kwargs.get("min_hops"), max_hops=kwargs.get("max_hops"),
            output_min_hops=kwargs.get("output_min_hops"), output_max_hops=kwargs.get("output_max_hops"),
            label_node_hops=kwargs.get("label_node_hops"), label_edge_hops=kwargs.get("label_edge_hops"),
            label_seeds=kwargs.get("label_seeds", False), edge_match=kwargs.get("edge_match"),
            source_node_match=kwargs.get("source_node_match"),
            destination_node_match=kwargs.get("destination_node_match"),
            source_node_query=kwargs.get("source_node_query"),
            destination_node_query=kwargs.get("destination_node_query"),
            edge_query=kwargs.get("edge_query"),
            include_zero_hop_seed=kwargs.get("include_zero_hop_seed", False),
            target_wave_front=kwargs.get("target_wave_front"),
        )
        if _indexed is not None:
            return _indexed
    return hop_polars(self, nodes, hops, **kwargs)
