"""Index-driven seeded traversal — the O(degree) fast path.

Replaces hop()'s O(E) ``edges[edges[src].isin(frontier)]`` scan with a CSR
searchsorted gather. Returns a subgraph Plottable parity-matched to the eager
hop() for the covered cases, or ``None`` when a feature isn't covered (caller
falls back to the scan/join path — correctness is never traded for speed).

Covered (v1): seeded (nodes given), integer ``hops`` >= 1 or ``to_fixed_point``,
direction forward/reverse/undirected, ``return_as_wave_front``, and a simple
scalar-equality ``edge_match`` (typed edges, e.g. Cypher ``-[:KNOWS]->``) applied on
the wavefront path. Not covered (returns None): predicate/membership edge_match,
source/destination match or query, edge_query, target_wave_front, min_hops>1,
output_min/max_hops, labeling, missing node table.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, cast

from typing_extensions import TypeGuard

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT
from graphistry.Plottable import Plottable
from .engine_arrays import (
    array_namespace, col_to_array, ids_to_array, take_rows, select_by_ids,
    set_difference, union1d,
)
from .lookup import lookup_edge_rows, lookup_node_rows
from .registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, AdjacencyIndex, GfqlIndexRegistry, NodeIdIndex
from .types import ArrayLike, EdgeMatch, HopDirection, SimpleEqualityEdgeMatch


def _indices_for_direction(
    registry: GfqlIndexRegistry,
    direction: HopDirection,
    edges: DataFrameT,
    cols: Tuple[str, str],
    engine: Engine,
) -> Optional[List[AdjacencyIndex]]:
    out_idx = cast(Optional[AdjacencyIndex], registry.get_valid(EDGE_OUT_ADJ, edges, cols, engine))
    in_idx = cast(Optional[AdjacencyIndex], registry.get_valid(EDGE_IN_ADJ, edges, cols, engine))
    if direction == "forward":
        return None if out_idx is None else [out_idx]
    if direction == "reverse":
        return None if in_idx is None else [in_idx]
    if out_idx is None or in_idx is None:
        return None
    return [out_idx, in_idx]


def is_simple_equality_edge_match(
    edge_match: Optional[EdgeMatch],
) -> TypeGuard[SimpleEqualityEdgeMatch]:
    """True iff ``edge_match`` is a dict of plain scalar equalities.

    This is the only ``edge_match`` shape the index path accelerates parity-exact:
    it mirrors filter_by_dict's concrete scalar ``==`` branch. ASTPredicate values
    (predicate path), membership lists/sets/tuples (isin path), and nested dicts are
    NOT covered here — the caller keeps them on the scan path.
    """
    if not edge_match:
        return False
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    for v in edge_match.values():
        if isinstance(v, ASTPredicate):
            return False
        if isinstance(v, (list, tuple, set, dict)):
            return False
    return True


def _build_edge_keep_mask(
    edges: DataFrameT, edge_match: EdgeMatch, engine: Engine, xp: "object"
) -> Optional[ArrayLike]:
    """Boolean array over ORIGINAL edge rows (length E, same indexing as
    ``AdjacencyIndex.other_values`` / ``row_positions``) selecting rows that satisfy
    a simple-equality ``edge_match``.

    Built via each frame's native ``col == val`` (so cudf string columns stay on the
    cudf layer instead of a cupy string compare). Returns ``None`` on ANY unexpected
    shape or error, so the caller falls back to scan rather than risk a divergence.
    """
    try:
        if not is_simple_equality_edge_match(edge_match):
            return None
        mask: Optional[ArrayLike] = None
        for col, val in edge_match.items():
            if col not in edges.columns:
                return None
            if engine in (Engine.POLARS, Engine.POLARS_GPU):
                col_mask = cast(ArrayLike, (edges.get_column(col) == val).to_numpy())
            elif engine == Engine.CUDF:
                col_mask = cast(ArrayLike, (edges[col] == val).values)
            else:
                col_mask = cast(ArrayLike, (edges[col] == val).to_numpy())
            mask = col_mask if mask is None else cast(ArrayLike, cast(Any, mask) & cast(Any, col_mask))
        return mask
    except Exception:  # pragma: no cover - defensive parity guard
        return None


def index_seeded_hop(
    g: Plottable,
    registry: GfqlIndexRegistry,
    *,
    nodes: DataFrameT,
    node_col: str,
    src: str,
    dst: str,
    engine: Engine,
    hops: Optional[int],
    to_fixed_point: bool,
    direction: HopDirection,
    return_as_wave_front: bool,
    edge_match: Optional[EdgeMatch] = None,
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

    # Typed-edge (edge_match) support: a boolean mask over ORIGINAL edge rows that
    # pass the match predicate, applied to the CSR-matched rows each hop. Gated to
    # simple scalar equality + the wavefront path by the coverability check upstream
    # (maybe_index_hop); an unsupported shape returns None here => scan (parity-safe).
    edge_keep: Optional[ArrayLike] = None
    if edge_match:
        edge_keep = _build_edge_keep_mask(edges, edge_match, engine, xp)
        if edge_keep is None:
            return None

    # Do NOT narrow the seed to the index key dtype (a node-id int64 seed cast to
    # an int32 edge-endpoint key wraps large ids → false match). lookup promotes both
    # sides to a common dtype; numpy/cupy set ops promote on concat. So we keep ids at
    # their natural width throughout and only ever widen.
    seed = xp.unique(ids_to_array(nodes, node_col, engine))

    frontier = seed
    visited = seed[:0]
    edge_rows_parts: List[ArrayLike] = []
    first = True
    hop_count = 0

    while True:
        if not to_fixed_point and hop_count >= hops:  # type: ignore[operator]
            break
        if int(frontier.shape[0]) == 0:
            break
        hop_count += 1

        matched_parts: List[ArrayLike] = []
        neigh_parts: List[ArrayLike] = []
        for ix in indices:
            rows, matched = lookup_edge_rows(ix, frontier, xp)
            if edge_keep is not None:
                # Keep only CSR-matched rows whose edge passes edge_match. Wavefront-
                # only (coverability gate), so the `matched`/first-hop `visited`
                # bookkeeping below — which edge_match does NOT filter — is never read.
                rows = rows[edge_keep[rows]]
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
        endpoints = xp.unique(xp.concatenate([src_vals, dst_vals]))  # natural dtype, never narrowed
        needed = union1d(needed, endpoints, xp)

    # Materialize node rows. Prefer the node_id index (O(result·log N) searchsorted
    # gather) over an O(N) isin scan — this keeps warm seeded latency flat in N.
    node_idx = cast(Optional[NodeIdIndex], registry.get_valid(NODE_ID, g._nodes, (node_col,), engine))
    if node_idx is not None:
        node_rows = lookup_node_rows(node_idx, needed, xp)
        # lookup returns rows in id-hit order; sort ascending so out_nodes keep
        # the original .nodes table order (the index must never reorder .nodes).
        node_rows = xp.sort(node_rows)
        out_nodes = take_rows(g._nodes, node_rows, engine)
    else:
        out_nodes = select_by_ids(g._nodes, node_col, needed, engine)

    # The scan synthesizes a node row for EVERY edge endpoint, including ids
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
