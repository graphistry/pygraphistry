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

from typing import Dict, List, Optional, Tuple, cast

from typing_extensions import TypeGuard

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.Plottable import Plottable
from .engine_arrays import (
    array_namespace, col_to_array, ids_to_array, take_rows, select_by_ids,
    set_difference, union1d,
)
from .lookup import lookup_edge_rows, lookup_node_rows
from .registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID, AdjacencyIndex, GfqlIndexRegistry, NodeIdIndex
from .types import (
    ArrayLike, EdgeMatch, HopDirection, ScalarMatchValue, SimpleEqualityEdgeMatch,
)

# Cost guard for candidate-row edge_match evaluation. Gathering candidate rows beats one
# whole-column compare only while the candidates stay a small fraction of the frame; a
# fixed-point walk that reaches most of the graph inverts that and gathers up to 2E by
# random access. Once cumulative gathered rows reach E/DIVISOR we build the whole-column
# mask once and reuse it, bounding total predicate work at ~(1 + 1/DIVISOR)*E. The FLOOR
# keeps small frames on the candidate-row path, where the whole-column compare is cheap
# anyway and the switch would only add a branch.
_EAGER_MASK_SWITCH_DIVISOR = 8
_EAGER_MASK_SWITCH_FLOOR = 1024


def _candidate_edge_mask_enabled() -> bool:
    """Candidate-row ``edge_match`` evaluation is on by default; set
    ``GFQL_INDEX_CANDIDATE_EDGE_MASK=0`` to force the whole-column mask on every hop.

    Follows the ``GFQL_LEAN_COMBINE`` precedent: the BOUNDARY is externally switchable so
    the differential harness can exercise both sides of it and assert they agree, while the
    numeric thresholds above stay private module constants like ``_LEAN_SHRINK_RATIO`` —
    they are a cost heuristic, not an interface, and the guard already bounds the bad case.
    """
    import os as _os

    return _os.environ.get("GFQL_INDEX_CANDIDATE_EDGE_MASK", "1") != "0"


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


class _EdgeMatchRowFilter:
    """Evaluates a simple-equality ``edge_match`` on the CSR-matched edge rows only.

    The mask is read exactly once per hop, as ``rows[keep[rows]]`` — at the handful of
    positions the adjacency lookup returned. Materializing it over all E edges first
    therefore put an O(E) predicate scan inside an O(degree) traversal, which is what
    made the indexed path scale with the graph instead of with the answer. Evaluating
    ``col == val`` on the gathered candidate rows makes the predicate proportional to
    the edges the traversal actually visits, so a seeded hop examines O(edges traversed)
    elements.

    A row is *mostly* returned once per index — frontiers are set-differenced against
    ``visited`` — but not strictly: ``edge_match`` is only reachable with
    ``return_as_wave_front=True``, and that mode skips the first-hop ``visited`` seeding
    below, so seed ids can re-enter a later frontier and their rows be gathered twice.
    The worst case is a fixed-point undirected walk reaching the whole graph, where the
    out- and in-indices are filtered separately and the gathered total approaches 2E
    against the eager form's single sequential pass over E. That regime is a genuine
    REGRESSION for this form (measured: 1.94×E gathered, 1.2–1.6× slower than the eager
    mask), which is why the caller keeps a cumulative-gathered counter and falls back to
    ``full_mask()`` once it crosses a fraction of the frame.

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
    """Positionally gather ``rows`` out of a single column. O(len(rows))."""
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

    All checks here are schema-level (O(1) in E); no predicate is evaluated yet.
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

    # Typed-edge (edge_match) support: the match predicate is evaluated on the
    # CSR-matched rows of each hop, so it costs O(edges visited) rather than O(E).
    # Gated to simple scalar equality + the wavefront path by the coverability check
    # upstream (maybe_index_hop); an unsupported shape returns None here => scan
    # (parity-safe). Schema validation happens now, up front, so an uncovered
    # edge_match still declines before any traversal work.
    edge_filter: Optional[_EdgeMatchRowFilter] = None
    if edge_match:
        edge_filter = _build_edge_row_filter(edges, edge_match, engine)
        if edge_filter is None:
            return None
    # Cost guard for the candidate-row form (see _EdgeMatchRowFilter.full_mask). Cumulative
    # gathered rows; once they reach a fraction of the frame we are demonstrably NOT in the
    # seeded regime, so we pay for the whole-column mask once and reuse it. Bounds total
    # predicate work at ~(1 + 1/D)*E instead of the unbounded-in-hops gather, while a seeded
    # hop — which gathers ~degree — never comes close to the threshold and never builds it.
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

    # #1888 endpoint closure: the scan DROPS edges whose endpoints are absent from a
    # bound node table (compute/hop.py symmetric gate) — it no longer synthesizes
    # phantom node rows. The CSR gather here is built from the raw edge frame, so a
    # traversal that touches a dangling endpoint would emit an unclosed edge; decline
    # (return None) so the scan serves the closed answer. No-op when nodes are complete.
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
