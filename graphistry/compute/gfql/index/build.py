"""Build CSR adjacency / node-id indexes from a graph's frames.

Build cost is O(E log E) (one sort), paid once per resident graph. The result is
a set of sidecar arrays over edge **row positions** — the user's ``.edges`` frame
is never reordered.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from graphistry.Engine import Engine
from .engine_arrays import array_namespace, col_to_array
from .registry import AdjacencyIndex, NodeIdIndex, frame_fingerprint


def _csr_from_keys(keys: Any, xp: Any) -> Tuple[Any, Any, Any]:
    """(keys array over E rows) -> (unique_keys, group_offsets[U+1], row_positions[E]).

    row_positions = the original row indices grouped (contiguously) by key value.
    Fully vectorized: one argsort + one boundary scan.
    """
    E = int(keys.shape[0])
    if E == 0:
        empty = keys[:0]
        return empty, xp.zeros(1, dtype=xp.int64), xp.zeros(0, dtype=xp.int64)
    order = xp.argsort(keys)                       # row positions sorted by key
    sorted_keys = keys[order]
    row_positions = order.astype(xp.int64)
    change = xp.ones(E, dtype=bool)
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = xp.nonzero(change)[0].astype(xp.int64)
    unique_keys = sorted_keys[starts]
    group_offsets = xp.concatenate([starts, xp.asarray([E], dtype=xp.int64)])
    return unique_keys, group_offsets, row_positions


def build_adjacency_index(
    edges: Any,
    kind: str,
    key_col: str,
    other_col: str,
    edge_id_col: Optional[str],
    engine: Engine,
    fingerprint_cols: Tuple[str, ...],
) -> AdjacencyIndex:
    xp, backend = array_namespace(engine)
    keys = col_to_array(edges, key_col, engine)
    other_values = col_to_array(edges, other_col, engine)
    unique_keys, group_offsets, row_positions = _csr_from_keys(keys, xp)
    return AdjacencyIndex(
        kind=kind,
        key_col=key_col,
        other_col=other_col,
        edge_id_col=edge_id_col,
        keys_sorted=unique_keys,
        group_offsets=group_offsets,
        row_positions=row_positions,
        other_values=other_values,
        backend=backend,
        engine=engine,
        fingerprint=frame_fingerprint(edges, fingerprint_cols, engine),
        source_ref=edges,
        n_edges=int(keys.shape[0]),
        n_keys=int(unique_keys.shape[0]),
    )


def build_node_id_index(
    nodes: Any,
    node_col: str,
    engine: Engine,
) -> Optional[NodeIdIndex]:
    """Sorted node-id -> first-row index, or None when node ids are NOT unique.

    ``_csr_from_keys`` returns ``row_positions`` of length E (all rows, grouped by
    key), but a node-id lookup indexes it with a *unique-key* searchsorted position
    (0..U-1). Those align ONLY when keys are unique — so we (a) collapse to the FIRST
    row position per unique key (``row_positions[group_offsets[:-1]]``, length U,
    aligned with ``unique_keys``) and (b) REFUSE (return None) when ids aren't unique:
    a unique-key CSR can't reproduce the scan's "all rows per id" semantics, so the
    caller falls back to the correct ``select_by_ids`` isin path. (B2: a non-unique
    node-id index dropped reached nodes / emitted unrelated rows.)"""
    xp, backend = array_namespace(engine)
    keys = col_to_array(nodes, node_col, engine)
    unique_keys, group_offsets, row_positions = _csr_from_keys(keys, xp)
    n_keys = int(unique_keys.shape[0])
    if n_keys != int(keys.shape[0]):
        return None  # duplicate node ids -> not a valid unique index; scan fallback
    first_row_per_key = row_positions[group_offsets[:-1]]  # length U, aligned to keys
    return NodeIdIndex(
        key_col=node_col,
        keys_sorted=unique_keys,
        row_positions=first_row_per_key,
        backend=backend,
        engine=engine,
        fingerprint=frame_fingerprint(nodes, (node_col,), engine),
        source_ref=nodes,
        n_nodes=n_keys,
    )
