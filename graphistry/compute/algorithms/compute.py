"""`g.compute_std(alg)` -- run a std kernel on a Plottable and bind the result.

The kernels require dense int32 vertex ids 0..V-1 (that is what makes their
frontier expansion a positional gather rather than a hash join). Real graphs
have arbitrary ids, so this layer renumbers, runs, and maps back -- the same
service `compute_cugraph` performs for cuGraph's own renumbering.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional

from ._dfops import dense_renumber
from .registry import STD_ALGS, output_column, run as _run_alg

# Algorithms whose output values are themselves vertex ids.
_LABEL_IS_VERTEX_ID = frozenset({"wcc", "cdlp"})


def compute_std(
    self: Any,
    alg: str,
    out_col: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Any:
    """Run `alg` and return a new Plottable with the result bound to nodes."""
    if alg not in STD_ALGS:
        raise ValueError(
            f"unknown graphistry.std procedure {alg!r}; known: {sorted(STD_ALGS)}"
        )
    g = self.materialize_nodes()
    src, dst, node = g._source, g._destination, g._node
    edges = g._edges[[src, dst]]

    dense, ids, v_count = dense_renumber(edges, src, dst)
    result = _run_alg(dense, src, dst, v_count, alg, params)

    # WCC and CDLP results ARE vertex ids (the component/community label is the
    # minimum member id), so they must be mapped back to the caller's id space
    # too -- not just the row order. Monotone renumbering means the dense label
    # and the original label denote the same vertex, but a user comparing a
    # label against their own node ids needs the original.
    if alg in _LABEL_IS_VERTEX_ID:
        result = ids.reset_index(drop=True).take(result.reset_index(drop=True))

    col = out_col or output_column(alg)
    # `ids` maps dense id -> original id, so pairing them positionally maps the
    # per-vertex result back onto the caller's node ids.
    mapping = type(g._nodes)({node: ids, col: result.reset_index(drop=True)})

    nodes = g._nodes
    if col in nodes.columns:
        nodes = nodes.drop(columns=[col])
    return g.nodes(nodes.merge(mapping, on=node, how="left"))
