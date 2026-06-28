"""Index-driven seeded traversal — the O(degree) fast path.

Replaces hop()'s O(E) ``edges[edges[src].isin(frontier)]`` scan with a CSR
searchsorted gather. Returns a subgraph Plottable parity-matched to the eager
hop() for the covered cases, or ``None`` when a feature isn't covered (caller
falls back to the scan/join path — correctness is never traded for speed).

Covered (v1): seeded (nodes given), integer ``hops`` >= 1 or ``to_fixed_point``,
direction forward/reverse/undirected, ``return_as_wave_front``. Not covered
(returns None): edge/source/destination match or query, target_wave_front,
min_hops>1, output_min/max_hops, labeling, missing node table.
"""
from __future__ import annotations

from typing import Any, List, Optional

from graphistry.Engine import Engine
from graphistry.Plottable import Plottable
from .engine_arrays import (
    array_namespace, col_to_array, ids_to_array, take_rows, select_by_ids,
    set_difference, union1d,
)
from .lookup import lookup_edge_rows, lookup_node_rows
from .registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, GfqlIndexRegistry


def _indices_for_direction(
    registry: GfqlIndexRegistry, direction: str, edges: Any, cols, engine: Engine
) -> Optional[List[Any]]:
    out_idx = registry.get_valid(EDGE_OUT_ADJ, edges, cols, engine)
    in_idx = registry.get_valid(EDGE_IN_ADJ, edges, cols, engine)
    if direction == "forward":
        chosen = [out_idx]
    elif direction == "reverse":
        chosen = [in_idx]
    else:  # undirected
        chosen = [out_idx, in_idx]
    if any(ix is None for ix in chosen):
        return None
    return chosen


def index_seeded_hop(
    g: Plottable,
    registry: GfqlIndexRegistry,
    *,
    nodes: Any,
    node_col: str,
    src: str,
    dst: str,
    engine: Engine,
    hops: Optional[int],
    to_fixed_point: bool,
    direction: str,
    return_as_wave_front: bool,
) -> Optional[Plottable]:
    if nodes is None or g._edges is None or g._nodes is None:
        return None
    if not to_fixed_point and (not isinstance(hops, int) or hops < 1):
        return None

    # Normalize the seed frame to the engine: the hop hooks can pass a pandas seeds
    # frame even on engine='polars'/'cudf' (conversion happens later in the scan path),
    # but col_to_array assumes engine-native frames. Convert here (seeds are small).
    from graphistry.Engine import df_to_engine
    seed_engine = Engine.POLARS if engine == Engine.POLARS_GPU else engine
    nodes = df_to_engine(nodes, seed_engine)

    edges = g._edges
    indices = _indices_for_direction(registry, direction, edges, (src, dst), engine)
    if indices is None:
        return None

    xp, _backend = array_namespace(engine)

    # I6: do NOT narrow the seed to the index key dtype (a node-id int64 seed cast to
    # an int32 edge-endpoint key wraps large ids → false match). lookup promotes both
    # sides to a common dtype; numpy/cupy set ops promote on concat. So we keep ids at
    # their natural width throughout and only ever widen.
    seed = xp.unique(ids_to_array(nodes, node_col, engine))

    frontier = seed
    visited = seed[:0]
    edge_rows_parts: List[Any] = []
    first = True
    hop_count = 0

    while True:
        if not to_fixed_point and hop_count >= hops:  # type: ignore[operator]
            break
        if int(frontier.shape[0]) == 0:
            break
        hop_count += 1

        matched_parts: List[Any] = []
        neigh_parts: List[Any] = []
        for ix in indices:
            rows, matched = lookup_edge_rows(ix, frontier, xp)
            edge_rows_parts.append(rows)
            neigh_parts.append(ix.other_values[rows])
            matched_parts.append(matched)

        neighbors = neigh_parts[0] if len(neigh_parts) == 1 else xp.concatenate(neigh_parts)
        if first and not return_as_wave_front:
            matched_all = (
                matched_parts[0] if len(matched_parts) == 1 else xp.concatenate(matched_parts)
            )
            visited = xp.unique(matched_all)
        first = False

        cand = xp.unique(neighbors)
        new_frontier = set_difference(cand, visited, xp)
        visited = union1d(visited, new_frontier, xp)
        frontier = new_frontier

    if edge_rows_parts:
        concat_rows = (
            edge_rows_parts[0]
            if len(edge_rows_parts) == 1
            else xp.concatenate(edge_rows_parts)
        )
        all_rows = xp.unique(concat_rows)
    else:
        all_rows = indices[0].row_positions[:0]

    out_edges = take_rows(edges, all_rows, engine)

    needed = visited
    materialize_endpoints = not return_as_wave_front  # nodes is non-None here (guarded above)
    if materialize_endpoints and int(all_rows.shape[0]) > 0:
        src_vals = col_to_array(out_edges, src, engine)
        dst_vals = col_to_array(out_edges, dst, engine)
        endpoints = xp.unique(xp.concatenate([src_vals, dst_vals]))  # natural dtype (I6)
        needed = union1d(needed, endpoints, xp)

    # Materialize node rows. Prefer the node_id index (O(result·log N) searchsorted
    # gather) over an O(N) isin scan — this keeps warm seeded latency flat in N.
    node_idx = registry.get_valid(NODE_ID, g._nodes, (node_col,), engine)
    if node_idx is not None:
        node_rows = lookup_node_rows(node_idx, needed, xp)
        # I4: lookup returns rows in id-hit order; sort ascending so out_nodes keep
        # the original .nodes table order (the index must never reorder .nodes).
        node_rows = xp.sort(node_rows)
        out_nodes = take_rows(g._nodes, node_rows, engine)
    else:
        out_nodes = select_by_ids(g._nodes, node_col, needed, engine)

    # B3: the scan synthesizes a node row for EVERY edge endpoint, including ids
    # absent from the node table (compute/hop.py "Ensure all edge endpoints are
    # present in nodes"); the index only materializes existing rows. If any needed id
    # is missing from the materialized nodes, fall back to scan (return None) rather
    # than silently drop it (a wrong-answer divergence). No-op when nodes are complete.
    present = col_to_array(out_nodes, node_col, engine)
    present_unique = xp.unique(present)
    if int(set_difference(needed, present_unique, xp).shape[0]) > 0:
        return None
    # B2: the scan dedups output nodes by id (hop.py drop_duplicates(subset=[node])).
    # The select_by_ids path returns ALL rows per id, so a node table with DUPLICATE
    # ids would emit extra rows here. Fall back to scan (O(result) check) rather than
    # diverge. (Unique-id tables — the norm — never trip this; node_id index unused
    # for dup ids by construction, so this only guards the isin path.)
    if int(present.shape[0]) != int(present_unique.shape[0]):
        return None
    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
