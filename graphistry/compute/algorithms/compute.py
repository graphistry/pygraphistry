"""GFQL adapter for running a std kernel and binding its node result.

The kernels require dense int32 vertex ids 0..V-1 (that is what makes their
frontier expansion a positional gather rather than a hash join). Real graphs
have arbitrary ids, so this layer renumbers, runs, and maps back -- the same
service `compute_cugraph` performs for cuGraph's own renumbering.
"""

from __future__ import annotations

from typing import Mapping, Optional, TYPE_CHECKING

from ._dfops import dense_renumber, to_host_int
from .registry import STD_ALGS, output_column, run as _run_alg

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable

# Algorithms whose output values are themselves vertex ids.
_LABEL_IS_VERTEX_ID = frozenset({"wcc", "cdlp"})


def compute_std(
    self: "Plottable",
    alg: str,
    out_col: Optional[str] = None,
    params: Optional[Mapping[str, object]] = None,
) -> "Plottable":
    """Run `alg` and return a new Plottable with the result bound to nodes."""
    if alg not in STD_ALGS:
        raise ValueError(f"unknown graphistry.std procedure {alg!r}; known: {sorted(STD_ALGS)}")
    g = self.materialize_nodes()
    src, dst, node = g._source, g._destination, g._node
    if src is None or dst is None or node is None:
        raise ValueError("graphistry.std requires source, destination, and node bindings")
    options = dict(params or {})
    edge_columns = [src, dst]
    if alg == "sssp" and options.get("weight") is not None:
        weight = options["weight"]
        if not isinstance(weight, str) or weight not in g._edges.columns:
            raise ValueError(f"SSSP weight must name an edge column, got {weight!r}")
        edge_columns.append(weight)
    edges = g._edges[edge_columns]
    if alg == "mis":
        edges = edges[edges[src] != edges[dst]].reset_index(drop=True)

    dense, ids, v_count = dense_renumber(edges, src, dst, g._nodes[node])
    if alg == "sssp" and "source" in options:
        source = options["source"]
        try:
            dense_source = to_host_int(ids.searchsorted(source))
        except (TypeError, ValueError):
            dense_source = -1
        if dense_source < 0 or dense_source >= v_count or ids.iloc[dense_source] != source:
            raise ValueError(f"SSSP source {source!r} is not a graph node")
        options["source"] = dense_source
    result = _run_alg(dense, src, dst, v_count, alg, options)

    # WCC/CDLP labels are vertex IDs and must return to the caller ID space.
    if alg in _LABEL_IS_VERTEX_ID:
        result = ids.reset_index(drop=True).take(result.reset_index(drop=True))

    col = out_col or output_column(alg)
    # Pair dense-to-original IDs positionally with per-vertex results.
    mapping = type(g._nodes)({node: ids, col: result.reset_index(drop=True)})

    nodes = g._nodes
    if col in nodes.columns:
        nodes = nodes.drop(columns=[col])
    return g.nodes(nodes.merge(mapping, on=node, how="left"))
