"""Index-driven seeded traversal.

Answers hop()'s ``edges[edges[src].isin(frontier)]`` with a CSR searchsorted
gather. Returns a subgraph Plottable parity-matched to the eager hop() for the
covered cases, or ``None`` when a feature isn't covered (caller falls back to the
scan/join path).

Covered (v1): seeded (nodes given), integer ``hops`` >= 1 or ``to_fixed_point``,
direction forward/reverse/undirected, ``return_as_wave_front``, and a simple
scalar-equality ``edge_match`` or source/destination node match on the wavefront
path. Not covered (returns None): predicate/membership edge_match, node queries,
edge_query, target_wave_front, min_hops>1, output_min/max_hops, labeling,
missing node table.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple, cast

from typing_extensions import TypeGuard

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.Plottable import Plottable
from .engine_arrays import (
    array_namespace, col_to_array, ids_to_array, take_rows, select_by_ids,
    set_difference, sorted_unique, union1d,
)
from .lookup import lookup_edge_rows, lookup_node_rows
from .registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, AdjacencyIndex, GfqlIndexRegistry, NodeIdIndex
from .types import (
    ArrayLike, ArrayNamespace, EdgeMatch, HopDirection, ScalarMatchValue, SimpleEqualityEdgeMatch,
)

_EAGER_MASK_SWITCH_DIVISOR = 8
_EAGER_MASK_SWITCH_FLOOR = 1024


def _candidate_edge_mask_enabled() -> bool:
    """Candidate-row ``edge_match`` evaluation is on by default; set
    ``GFQL_INDEX_CANDIDATE_EDGE_MASK=0`` to force the whole-column mask on every hop.

    Follows the ``GFQL_LEAN_COMBINE`` precedent: the BOUNDARY is externally switchable so
    the differential harness can exercise both sides of it and assert they agree; the numeric
    thresholds above stay private module constants like ``_LEAN_SHRINK_RATIO``.
    """
    import os as _os

    return _os.environ.get("GFQL_INDEX_CANDIDATE_EDGE_MASK", "1") != "0"


def _indices_for_direction(
    registry: GfqlIndexRegistry,
    direction: HopDirection,
    edges: DataFrameT,
    cols: Tuple[str, str],
    engine: Engine,
    edge_match: Optional[EdgeMatch] = None,
) -> Optional[List[AdjacencyIndex]]:
    kinds = (
        (EDGE_OUT_ADJ,) if direction == "forward" else
        (EDGE_IN_ADJ,) if direction == "reverse" else
        (EDGE_OUT_ADJ, EDGE_IN_ADJ)
    )
    if edge_match:
        for column, value in edge_match.items():
            if (
                not isinstance(column, str)
                or isinstance(value, bool)
                or not isinstance(value, (str, int))
            ):
                continue
            typed = [
                registry.get_typed_adjacency_valid(
                    column, value, kind, edges, cols[0], cols[1], engine,
                ) for kind in kinds
            ]
            if all(index is not None for index in typed):
                return [index for index in typed if index is not None]
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
    from graphistry.compute.filter_by_dict import _is_membership_filter_value
    for v in edge_match.values():
        if isinstance(v, ASTPredicate):
            return False
        # Membership must be decided by the SAME helper the scan path uses
        # (filter_by_dict). A local (list, tuple, set, dict) check misses frozenset /
        # pd.Index / pd.Series, which the scan lowers to isin — an equality mask over
        # such a value is silently all-False (wrong answer), never an error.
        if _is_membership_filter_value(v) or isinstance(v, dict):
            return False
    return True


def is_simple_equality_node_match(
    node_match: Optional[object],
) -> bool:
    """True iff ``node_match`` is None or a SINGLE-key plain scalar-equality dict.

    The only node-match shape the node type-segment path accelerates parity-exact
    in v1: the segment covers one (column, value) pair, and residual keys would
    need id->row candidate filtering the segment path does not yet plumb. Boolean
    ``True``/``False`` values ARE scalar here (the canonical Cypher ``label__X``
    shape), unlike the typed-edge value check which rejects them.
    """
    if node_match is None:
        return True
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    from graphistry.compute.filter_by_dict import _is_membership_filter_value
    if not isinstance(node_match, dict) or len(node_match) != 1:
        return False
    for column, value in node_match.items():
        if not isinstance(column, str):
            return False
        if isinstance(value, ASTPredicate) or _is_membership_filter_value(value) or isinstance(value, dict):
            return False
    return True


def _sorted_id_mask(values: ArrayLike, sorted_ids: ArrayLike, xp: ArrayNamespace) -> ArrayLike:
    """Boolean mask over ``values`` marking membership in sorted-unique ``ids``.

    Promotes BOTH sides to a common dtype (never narrows the query side — the
    int64-cast lesson from the typed edge positions applies to id comparisons too).
    """
    n = int(values.shape[0])
    if n == 0 or int(sorted_ids.shape[0]) == 0:
        return xp.zeros(n, dtype=bool)
    v = values
    keys = sorted_ids
    if v.dtype != keys.dtype:
        common = xp.promote_types(v.dtype, keys.dtype)
        v = v.astype(common)
        keys = keys.astype(common)
    pos = xp.searchsorted(keys, v)
    pos_clipped = xp.where(pos < int(keys.shape[0]), pos, int(keys.shape[0]) - 1)
    return keys[pos_clipped] == v


def _sorted_intersection(a: ArrayLike, b: ArrayLike, xp: ArrayNamespace) -> ArrayLike:
    """Sorted-unique intersection of two sorted-unique id arrays."""
    if int(a.shape[0]) == 0 or int(b.shape[0]) == 0:
        return a[:0]
    return a[_sorted_id_mask(a, b, xp)]


class _EdgeMatchRowFilter:
    """Evaluates a simple-equality ``edge_match`` on the CSR-matched edge rows only.

    The mask is read as ``rows[keep[rows]]``, at the positions the adjacency lookup returned.

    A row is *mostly* gathered once per index — frontiers are set-differenced against
    ``visited`` — but not strictly: ``edge_match`` is only reachable with
    ``return_as_wave_front=True``, and that mode skips the first-hop ``visited`` seeding
    below, so seed ids can re-enter a later frontier and their rows be gathered twice.

    Column values are compared with each frame's native ``==`` (so cudf string columns
    stay on the cudf layer rather than becoming a cupy string compare), matching the
    eager form exactly.
    """

    __slots__ = ("_series", "_items", "_engine")

    # Typed slots: the per-column edge Series keyed by column name, the validated
    # (column, scalar) equalities in ``edge_match`` order, and the frame engine.
    _series: Dict[str, SeriesT]
    _items: List[Tuple[str, ScalarMatchValue]]
    _engine: Engine

    def __init__(
        self,
        series: Dict[str, SeriesT],
        items: List[Tuple[str, ScalarMatchValue]],
        engine: Engine,
    ) -> None:
        self._series = series
        self._items = items
        self._engine = engine

    def mask_for(self, rows: ArrayLike) -> Optional[ArrayLike]:
        """Boolean array over ``rows`` (positional, same order), or ``None`` on any
        unexpected shape/error so the caller falls back to the scan."""
        try:
            mask: Optional[ArrayLike] = None
            for col, val in self._items:
                sub = _gather_series(self._series[col], rows, self._engine)
                col_mask: ArrayLike
                # Null-safe materialization: on null-carrying columns (pandas nullable
                # Int64/boolean/string, polars nulls — which the NaN->null coercion
                # makes common) a bare == yields NA cells, and to_numpy() then produces
                # an OBJECT-dtype array that later explodes at rows[keep] (IndexError:
                # not int/bool). Null == val filters out on the scan path, so fill
                # False is parity-exact.
                if self._engine in (Engine.POLARS, Engine.POLARS_GPU):
                    col_mask = (sub == val).fill_null(False).to_numpy()
                elif self._engine == Engine.CUDF:
                    col_mask = (sub == val).fillna(False).values
                else:
                    col_mask = (sub == val).fillna(False).to_numpy(dtype=bool)
                mask = col_mask if mask is None else mask & col_mask
            return mask
        except Exception:  # pragma: no cover - defensive parity guard
            return None

    def full_mask(self) -> Optional[ArrayLike]:
        """The eager whole-column mask, length E — the pre-candidate-row form.

        Candidate-row evaluation is a win exactly while the candidates are a small
        fraction of the frame. A fixed-point walk that reaches most of the graph inverts
        that: it gathers up to 2E elements (out- and in-indices filtered separately), by
        random access, versus one sequential compare over E. The caller switches to this
        once it has gathered enough to know it is in that regime; see
        ``_EAGER_MASK_SWITCH_DIVISOR``.
        """
        try:
            mask: Optional[ArrayLike] = None
            for col, val in self._items:
                col_mask: ArrayLike
                series = self._series[col]
                # Same null-safe materialization as mask_for, so the two forms agree
                # cell for cell — this is the parity-critical property of the switch.
                if self._engine in (Engine.POLARS, Engine.POLARS_GPU):
                    col_mask = (series == val).fill_null(False).to_numpy()
                elif self._engine == Engine.CUDF:
                    col_mask = (series == val).fillna(False).values
                else:
                    col_mask = (series == val).fillna(False).to_numpy(dtype=bool)
                mask = col_mask if mask is None else mask & col_mask
            return mask
        except Exception:  # pragma: no cover - defensive parity guard
            return None


def _gather_series(series: SeriesT, rows: ArrayLike, engine: Engine) -> SeriesT:
    """Positionally gather ``rows`` out of a single column."""
    if engine in (Engine.POLARS, Engine.POLARS_GPU):
        import numpy as np

        return series.gather(np.asarray(rows))
    # pandas / cudf: positional take accepts numpy (pandas) or cupy (cudf) int arrays
    return series.take(rows)


def _build_edge_row_filter(
    edges: DataFrameT, edge_match: EdgeMatch, engine: Engine
) -> Optional[_EdgeMatchRowFilter]:
    """Validate a simple-equality ``edge_match`` against the edge schema and return a
    per-row evaluator, or ``None`` when the shape isn't covered (caller falls back to
    the scan rather than risk a divergence).

    All checks here are schema-level; no predicate is evaluated yet.
    """
    try:
        if not is_simple_equality_edge_match(edge_match):
            return None
        from graphistry.compute.filter_by_dict import (
            _is_numeric_dtype_safe, _is_string_dtype_safe,
        )
        n_edges = int(edges.shape[0])
        series: Dict[str, SeriesT] = {}
        items: List[Tuple[str, ScalarMatchValue]] = []
        for col, val in edge_match.items():
            if col not in edges.columns:
                return None
            col_series: SeriesT
            if engine in (Engine.POLARS, Engine.POLARS_GPU):
                col_series = edges.get_column(col)
            else:
                col_series = edges[col]
            # Obvious dtype mismatch (numeric col vs str val, string col vs numeric
            # val): the scan raises GFQLSchemaError E302 where a naive == is silently
            # all-False. Decline -> caller falls back to the scan, which raises the
            # SAME error (parity-exact; mirrors filter_by_dict's exact two checks,
            # skipped like the scan on empty frames).
            if n_edges > 0:
                dt = col_series.dtype
                if _is_numeric_dtype_safe(dt) and isinstance(val, str):
                    return None
                if (_is_string_dtype_safe(dt)
                        and isinstance(val, (int, float)) and not isinstance(val, bool)):
                    return None
            series[col] = col_series
            items.append((col, val))
        return _EdgeMatchRowFilter(series, items, engine)
    except Exception:  # pragma: no cover - defensive parity guard
        return None


def _node_match_allowed_ids(
    g: Plottable, registry: GfqlIndexRegistry, node_match: Optional[object],
    node_col: Optional[str], engine: Engine,
) -> Optional[ArrayLike]:
    """Sorted-unique ids admitted by one single-key scalar node match, or None to
    decline the indexed path (caller falls back to the scan).

    Membership comes from the LIVE node type segment, whose rows were decided by
    the canonical filter_by_dict predicate at build time — parity with the scan's
    allowed-id construction by construction. An EMPTY segment is a real zero-row
    answer (empty sorted array), NOT a decline."""
    from .engine_arrays import array_namespace, col_to_array, sorted_unique, take_rows

    if not is_simple_equality_node_match(node_match) or node_col is None or g._nodes is None:
        return None
    assert isinstance(node_match, dict)
    xp, _backend = array_namespace(engine)
    (column, value), = node_match.items()
    if not isinstance(value, (str, int)):  # bool is an int subclass; guards exotic literals
        return None
    segment = registry.get_node_segment_valid(column, value, g._nodes, engine)
    if segment is None:
        return None
    segment_rows = segment.row_positions
    if int(segment_rows.shape[0]) == 0:
        return segment_rows[:0]
    ids = col_to_array(take_rows(g._nodes, segment_rows, engine), node_col, engine)
    return sorted_unique(xp, ids)


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
    source_node_match: Optional[object] = None,
    destination_node_match: Optional[object] = None,
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
    indices = _indices_for_direction(registry, direction, edges, (src, dst), engine, edge_match)
    if indices is None:
        return None

    xp, _backend = array_namespace(engine)

    # Typed-edge (edge_match): gated to simple scalar equality + the wavefront path by the
    # coverability check upstream (maybe_index_hop); an unsupported shape returns None here =>
    # scan (parity-safe). Schema validation happens up front, before any traversal work.
    edge_filter: Optional[_EdgeMatchRowFilter] = None
    if edge_match:
        edge_filter = _build_edge_row_filter(edges, edge_match, engine)
        if edge_filter is None:
            return None

    # Node matches resolve to allowed ids; empty is an answer, missing is a decline.
    source_allowed: Optional[ArrayLike] = None
    dest_allowed: Optional[ArrayLike] = None
    if source_node_match is not None:
        source_allowed = _node_match_allowed_ids(
            g, registry, source_node_match, node_col, engine,
        )
        if source_allowed is None:
            return None
    if destination_node_match is not None:
        dest_allowed = _node_match_allowed_ids(
            g, registry, destination_node_match, node_col, engine,
        )
        if dest_allowed is None:
            return None
    # Guard for the candidate-row form (see _EdgeMatchRowFilter.full_mask): once cumulative
    # gathered rows cross the threshold, build the whole-column mask once and reuse it.
    gathered_rows = 0
    eager_keep: Optional[ArrayLike] = None
    switch_at = (
        max(_EAGER_MASK_SWITCH_FLOOR, len(edges) // _EAGER_MASK_SWITCH_DIVISOR)
        if _candidate_edge_mask_enabled() else 0  # 0 => build the whole-column mask up front
    )

    # Do NOT narrow the seed to the index key dtype (a node-id int64 seed cast to
    # an int32 edge-endpoint key wraps large ids → false match). lookup promotes both
    # sides to a common dtype; numpy/cupy set ops promote on concat. So we keep ids at
    # their natural width throughout and only ever widen.
    seed = sorted_unique(xp, ids_to_array(nodes, node_col, engine))

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
        # Source restriction applies to seeds and every carried-forward frontier.
        if source_allowed is not None:
            frontier = _sorted_intersection(frontier, source_allowed, xp)
            if int(frontier.shape[0]) == 0:
                break
        hop_count += 1

        matched_parts: List[ArrayLike] = []
        neigh_parts: List[ArrayLike] = []
        for ix in indices:
            rows, matched = lookup_edge_rows(ix, frontier, xp)
            if edge_filter is not None:
                # Keep only CSR-matched rows whose edge passes edge_match. Wavefront-
                # only (coverability gate), so the `matched`/first-hop `visited`
                # bookkeeping below — which edge_match does NOT filter — is never read.
                # Decide BEFORE gathering this batch, not after: a single hop's batch can
                # be arbitrarily large, so a post-hoc check would overshoot by up to one
                # whole batch. Checking the projected total keeps gathered <= switch_at.
                if (eager_keep is None
                        and gathered_rows + int(rows.shape[0]) > switch_at):
                    eager_keep = edge_filter.full_mask()
                    if eager_keep is None:
                        return None
                if eager_keep is not None:
                    rows = rows[eager_keep[rows]]
                else:
                    gathered_rows += int(rows.shape[0])
                    keep = edge_filter.mask_for(rows)
                    if keep is None:
                        # Evaluation failed on this candidate batch: abandon the indexed
                        # path entirely so the caller re-runs the hop on the scan. Nothing
                        # observable has been mutated, so this stays parity-safe.
                        return None
                    rows = rows[keep]
            if dest_allowed is not None:
                # Destination matching drops both frontier ids and edge rows.
                rows = rows[_sorted_id_mask(ix.other_values[rows], dest_allowed, xp)]
            edge_rows_parts.append(rows)
            neigh_parts.append(ix.other_values[rows])
            matched_parts.append(matched)

        neighbors = neigh_parts[0] if len(neigh_parts) == 1 else xp.concatenate(neigh_parts)
        if first and not return_as_wave_front:
            if edge_filter is None and dest_allowed is None:
                matched_all = (
                    matched_parts[0] if len(matched_parts) == 1 else xp.concatenate(matched_parts)
                )
            else:
                # A filter can reject a source's only edge: first-hop visited = surviving rows' sources.
                surviving = (
                    edge_rows_parts[0] if len(edge_rows_parts) == 1
                    else xp.concatenate(edge_rows_parts)
                )
                matched_all = col_to_array(edges, src, engine)[surviving]
            visited = sorted_unique(xp, matched_all)
        first = False

        cand = sorted_unique(xp, neighbors)
        new_frontier = set_difference(cand, visited, xp)
        visited = union1d(visited, new_frontier, xp)
        frontier = new_frontier

    if edge_rows_parts:
        concat_rows = (
            edge_rows_parts[0]
            if len(edge_rows_parts) == 1
            else xp.concatenate(edge_rows_parts)
        )
        all_rows = sorted_unique(xp, concat_rows)
    else:
        all_rows = indices[0].row_positions[:0]

    out_edges = take_rows(edges, all_rows, engine)

    needed = visited
    materialize_endpoints = not return_as_wave_front  # nodes is non-None here (guarded above)
    if materialize_endpoints and int(all_rows.shape[0]) > 0:
        src_vals = col_to_array(out_edges, src, engine)
        dst_vals = col_to_array(out_edges, dst, engine)
        endpoints = sorted_unique(xp, xp.concatenate([src_vals, dst_vals]))  # natural dtype, never narrowed
        needed = union1d(needed, endpoints, xp)

    # Materialize node rows: node_id index searchsorted gather when available, else isin scan.
    node_idx = cast(Optional[NodeIdIndex], registry.get_valid(NODE_ID, g._nodes, (node_col,), engine))
    if node_idx is not None:
        node_rows = lookup_node_rows(node_idx, needed, xp)
        # lookup returns rows in id-hit order; sort ascending so out_nodes keep
        # the original .nodes table order (the index must never reorder .nodes).
        node_rows = xp.sort(node_rows)
        out_nodes = take_rows(g._nodes, node_rows, engine)
    else:
        out_nodes = select_by_ids(g._nodes, node_col, needed, engine)

    present = col_to_array(out_nodes, node_col, engine)
    present_unique = sorted_unique(xp, present)
    reached_an_endpoint_with_no_node_row = (
        int(set_difference(needed, present_unique, xp).shape[0]) > 0)
    if reached_an_endpoint_with_no_node_row:
        return None
    # B2: the scan dedups output nodes by id (hop.py drop_duplicates(subset=[node])).
    # The select_by_ids path returns ALL rows per id, so a node table with DUPLICATE
    # ids would emit extra rows here. Fall back to scan rather than diverge. (node_id index
    # is unused for dup ids by construction, so this only guards the isin path.)
    if int(present.shape[0]) != int(present_unique.shape[0]):
        return None
    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
