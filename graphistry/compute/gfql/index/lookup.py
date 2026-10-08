"""Vectorized CSR lookup — searchsorted membership + range-expansion gather.

Given a frontier of seed ids, return the edge **row positions** of all incident
edges, with no full edge scan and no per-seed Python loop. Works identically on
numpy (pandas/polars) and cupy (cudf) arrays.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

from .registry import AdjacencyIndex, NodeIdIndex, NodePropIndex
from .types import ArrayLike, ArrayNamespace


def lookup_edge_rows(index: AdjacencyIndex, frontier: ArrayLike, xp: ArrayNamespace) -> Tuple[ArrayLike, ArrayLike]:
    """frontier (backend array of seed ids, deduped) -> (edge_rows, matched_ids).

    ``edge_rows``  = row positions of all edges incident to the frontier.
    ``matched_ids`` = the subset of ``frontier`` that has >=1 incident edge
                      (needed to reproduce hop()'s first-hop visited semantics).

    Steps (all vectorized):
      pos   = searchsorted(keys, frontier)         # candidate group per seed
      hit   = keys[pos] == frontier                # membership verify
      [start,end) = group_offsets[pos], [pos+1]    # CSR slice per hit
      flat  = expand each [start,end) range        # cumsum/arange/repeat trick
      rows  = row_positions[flat]
    """
    keys = index.keys_sorted
    empty = index.row_positions[:0]
    U = int(keys.shape[0])
    if U == 0 or int(frontier.shape[0]) == 0:
        return empty, frontier[:0]

    f = frontier
    if f.dtype != keys.dtype:
        # Promote BOTH sides to a common dtype — never narrow the query to the key
        # dtype (an int64 id cast to int32 keys wraps and false-matches). Widening a
        # sorted int array preserves order, so searchsorted stays valid.
        common = xp.promote_types(f.dtype, keys.dtype)
        f = f.astype(common)
        keys = keys.astype(common)

    # A singleton frontier expands one CSR bucket; retain independent result buffers.
    if index.backend == "cupy" and int(f.shape[0]) == 1:
        positions = _csr_hit_positions(keys, f, xp)
        if int(positions.shape[0]) == 0:
            return empty, f[:0]
        bucket_start, bucket_end = _csr_single_group_bounds(index, positions)
        return index.row_positions[bucket_start:bucket_end].copy(), f.copy()

    pos = xp.searchsorted(keys, f)
    pos_clipped = xp.where(pos < U, pos, U - 1)
    hit = keys[pos_clipped] == f
    matched_ids = f[hit]
    pos_hit = pos_clipped[hit]
    if int(pos_hit.shape[0]) == 0:
        return empty, matched_ids

    start = index.group_offsets[pos_hit]
    end = index.group_offsets[pos_hit + 1]
    counts = end - start
    total = int(counts.sum())
    if total == 0:
        return empty, matched_ids

    flat = _expand_ranges(start, counts, total, xp)
    return index.row_positions[flat], matched_ids



def _expand_ranges(start: ArrayLike, counts: ArrayLike, total: int, xp: ArrayNamespace) -> ArrayLike:
    """Vectorized [start, start+count) range concat WITHOUT np.repeat (cupy's
    ``repeat`` rejects array ``repeats``). Builds a per-output segment id via a
    boundary-marker cumsum, then gathers start/offset by segment.

    Precondition: every count >= 1 (CSR groups always have >=1 edge), so group
    start offsets are strictly increasing and the boundary markers don't collide.
    """
    out_off = xp.cumsum(counts) - counts          # output start of each group
    seg = xp.zeros(total, dtype=xp.int64)
    if int(out_off.shape[0]) > 1:
        seg[out_off[1:]] = 1
    seg = xp.cumsum(seg)                            # group index per output position
    pos_in = xp.arange(total, dtype=xp.int64) - out_off[seg]
    return start[seg] + pos_in


def lookup_node_rows(index: NodeIdIndex, ids: ArrayLike, xp: ArrayNamespace) -> ArrayLike:
    """ids (backend array) -> node row positions for those that exist (in id order
    of the index hits). Used to materialize node rows for a result id set."""
    keys = index.keys_sorted
    U = int(keys.shape[0])
    if U == 0 or int(ids.shape[0]) == 0:
        return index.row_positions[:0]
    f = ids
    if f.dtype != keys.dtype:
        common = xp.promote_types(f.dtype, keys.dtype)  # promote, never narrow
        f = f.astype(common)
        keys = keys.astype(common)
    pos = xp.searchsorted(keys, f)
    pos_clipped = xp.where(pos < U, pos, U - 1)
    hit = keys[pos_clipped] == f
    return index.row_positions[pos_clipped[hit]]


def _csr_hit_positions(keys: ArrayLike, values: ArrayLike, xp: ArrayNamespace) -> ArrayLike:
    """Group positions in ``keys`` for the ``values`` that are actually present.

    The shared front half of every CSR probe: promote to a common dtype (never
    narrow — an int64 id cast to int32 keys wraps and false-matches), searchsorted,
    then verify membership. Returns the matched group positions.
    """
    U = int(keys.shape[0])
    if U == 0 or int(values.shape[0]) == 0:
        return xp.zeros(0, dtype=xp.int64)
    if values.dtype != keys.dtype:
        common = xp.promote_types(values.dtype, keys.dtype)
        values = values.astype(common)
        keys = keys.astype(common)
    import numpy as np
    if isinstance(keys, np.ndarray) and isinstance(values, np.ndarray) and values.size == 1:
        value = values[0]
        position = keys.searchsorted(value)
        if position < U and keys[position] == value:
            return xp.asarray([position], dtype=xp.int64)
        return xp.zeros(0, dtype=xp.int64)
    if int(values.shape[0]) == 1:
        import cupy as cp
        # Only a query position crosses the device boundary, never source rows.
        position = int(cp.asnumpy(xp.searchsorted(keys, values))[0])
        if position < U and keys[position] == values[0]:
            return xp.asarray([position], dtype=xp.int64)
        return xp.zeros(0, dtype=xp.int64)
    pos = xp.searchsorted(keys, values)
    clipped = xp.where(pos < U, pos, U - 1)
    return clipped[keys[clipped] == values]


def _csr_group_sizes(index: Any, positions: ArrayLike) -> ArrayLike:
    """Row count of each CSR group named by ``positions``."""
    return index.group_offsets[positions + 1] - index.group_offsets[positions]


def _csr_single_group_bounds(index: Any, positions: ArrayLike) -> Tuple[int, int]:  # hygiene-ok: explicit-any -- shares the existing CSR interface for property and adjacency records
    """Read one CSR bucket's bounds; GPU probes transfer only three integers."""
    if index.backend == "cupy":
        import cupy as cp
        group = int(cp.asnumpy(positions)[0])
        bounds = cp.asnumpy(index.group_offsets[group:group + 2])
        return int(bounds[0]), int(bounds[1])
    import numpy as np
    group = int(np.asarray(positions)[0])
    return int(index.group_offsets[group]), int(index.group_offsets[group + 1])


def csr_match_count(index: Any, values: ArrayLike, xp: ArrayNamespace) -> int:
    """How many rows a CSR gather of ``values`` would return — offsets only, no
    gather. The planner's free selectivity/degree estimate."""
    positions = _csr_hit_positions(index.keys_sorted, values, xp)
    if int(positions.shape[0]) == 0:
        return 0
    if int(positions.shape[0]) == 1:
        bucket_start, bucket_end = _csr_single_group_bounds(index, positions)
        return bucket_end - bucket_start
    return int(_csr_group_sizes(index, positions).sum())


def csr_gather_rows(index: Any, values: ArrayLike, xp: ArrayNamespace, *, group_positions: Optional[ArrayLike] = None, group_sizes: Optional[ArrayLike] = None, match_count: Optional[int] = None) -> ArrayLike:
    """Row positions of every row whose key is in ``values`` (CSR range expansion)."""
    positions = group_positions if group_positions is not None else _csr_hit_positions(index.keys_sorted, values, xp)
    empty = index.row_positions[:0]
    if int(positions.shape[0]) == 0:
        return empty
    if int(positions.shape[0]) == 1:
        bucket_start, bucket_end = _csr_single_group_bounds(index, positions)
        return index.row_positions[bucket_start:bucket_end]
    start = index.group_offsets[positions]
    counts = group_sizes if group_sizes is not None else _csr_group_sizes(index, positions)
    total = match_count if match_count is not None else int(counts.sum())
    if total == 0:
        return empty
    return index.row_positions[_expand_ranges(start, counts, total, xp)]


def lookup_prop_rows(index: NodePropIndex, values: ArrayLike, xp: ArrayNamespace, *, group_positions: Optional[ArrayLike] = None, group_sizes: Optional[ArrayLike] = None, match_count: Optional[int] = None) -> ArrayLike:
    """values -> node row positions of every row holding one of them.

    Order is unspecified here; callers that need frame order sort.
    """
    return csr_gather_rows(index, values, xp, group_positions=group_positions, group_sizes=group_sizes, match_count=match_count)


def prop_match_count(index: NodePropIndex, values: ArrayLike, xp: ArrayNamespace) -> int:
    """Rows ``lookup_prop_rows`` would return — the free selectivity estimate."""
    return csr_match_count(index, values, xp)


def lookup_degree(index: AdjacencyIndex, frontier: ArrayLike, xp: ArrayNamespace) -> int:
    """Total incident-edge count for a frontier — the hop's fanout estimate,
    computed from CSR offsets before any range expansion."""
    return csr_match_count(index, frontier, xp)
