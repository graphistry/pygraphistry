"""Vectorized CSR lookup — searchsorted membership + range-expansion gather.

Given a frontier of seed ids, return the edge **row positions** of all incident
edges, with no full edge scan and no per-seed Python loop. Works identically on
numpy (pandas/polars) and cupy (cudf) arrays.
"""
from __future__ import annotations

from typing import Any

from .registry import AdjacencyIndex, NodeIdIndex


def lookup_edge_rows(index: AdjacencyIndex, frontier: Any, xp: Any):
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


def _expand_ranges(start: Any, counts: Any, total: int, xp: Any) -> Any:
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


def lookup_node_rows(index: NodeIdIndex, ids: Any, xp: Any) -> Any:
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
