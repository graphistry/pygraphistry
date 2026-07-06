"""Index-accelerated node degrees (#5 degree-cache / #3 membership).

Resident CSR adjacency indexes already carry per-node degree for free:
degree(key) = ``group_offsets[i+1] - group_offsets[i]``. This turns
``get_degrees`` from an O(E) ``group_by(endpoint)`` + join into an O(N)
``searchsorted`` gather. Bulk-over-all-nodes, so it always beats the scan when a
valid index is resident (no selectivity gate). Engine-polymorphic (numpy host for
pandas/polars, cupy device for cudf), fingerprint-validated like the hop fast path.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from graphistry.Engine import Engine
from .engine_arrays import array_namespace, col_to_array
from .registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, GfqlIndexRegistry


def _degree_for_nodes(adj_index: Any, node_ids: Any, xp: Any) -> Any:
    """Per-node degree via searchsorted into the CSR keys (0 for isolated nodes)."""
    keys = adj_index.keys_sorted
    offs = adj_index.group_offsets
    U = int(keys.shape[0])
    n = int(node_ids.shape[0])
    if U == 0:
        return xp.zeros(n, dtype=xp.int64)
    key_deg = (offs[1:] - offs[:-1]).astype(xp.int64)         # degree per unique key
    # dtype-safe searchsorted (promote, never narrow — mirrors lookup.py)
    f = node_ids
    if getattr(f, "dtype", None) != getattr(keys, "dtype", None):
        common = xp.promote_types(f.dtype, keys.dtype)
        f = f.astype(common)
        keys = keys.astype(common)
    pos = xp.searchsorted(keys, f)
    pos_c = xp.where(pos < U, pos, U - 1)
    hit = keys[pos_c] == f
    return xp.where(hit, key_deg[pos_c], xp.zeros(n, dtype=xp.int64))


def degrees_from_index(
    registry: GfqlIndexRegistry, nodes_df: Any, node_col: str,
    edges_df: Any, cols: Tuple[str, ...], engine: Engine,
) -> Optional[Tuple[Any, Any]]:
    """Return (in_degree, out_degree) arrays aligned to ``nodes_df`` rows, or None
    to fall back to the group_by path (no valid resident adjacency index)."""
    oi = registry.get_valid(EDGE_OUT_ADJ, edges_df, cols, engine)
    ii = registry.get_valid(EDGE_IN_ADJ, edges_df, cols, engine)
    if oi is None or ii is None:
        return None
    xp, _ = array_namespace(engine)
    node_ids = col_to_array(nodes_df, node_col, engine)
    out_deg = _degree_for_nodes(oi, node_ids, xp)   # out = src-keyed adjacency
    in_deg = _degree_for_nodes(ii, node_ids, xp)    # in  = dst-keyed adjacency
    return in_deg, out_deg


def adjacency_membership_keys(
    registry: GfqlIndexRegistry, direction: str, edges_df: Any, cols: Tuple[str, ...], engine: Engine,
) -> Optional[Any]:
    """Backend array of node-ids with >=1 edge in ``direction``
    ('forward' = has out-edge, 'reverse' = has in-edge, 'undirected' = either).
    None if the needed valid index isn't resident. #3 membership (= degree>=1);
    powers EXISTS {(n)--()} prune-isolated without an O(E) traversal.

    NOTE: keys include nodes whose ONLY edge is a self-loop — correct for the bare
    pattern; the drop-self (neq) flavor must NOT use this path.
    """
    from .engine_arrays import union1d
    if direction in ("forward", "undirected"):
        oi = registry.get_valid(EDGE_OUT_ADJ, edges_df, cols, engine)
        if oi is None:
            return None
    if direction in ("reverse", "undirected"):
        ii = registry.get_valid(EDGE_IN_ADJ, edges_df, cols, engine)
        if ii is None:
            return None
    if direction == "forward":
        return oi.keys_sorted
    if direction == "reverse":
        return ii.keys_sorted
    xp, _ = array_namespace(engine)
    return union1d(oi.keys_sorted, ii.keys_sorted, xp)
