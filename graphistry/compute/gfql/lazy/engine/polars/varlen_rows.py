"""Variable-length (``-[*i..k]->`` / unbounded) binding-row specializations.

Extracted from ``row_pipeline.py`` so the core row flow stays readable and so the
other in-flight PRs that touch that file do not have to rebase across the
specialization. The extraction itself was a pure code move: no behaviour change.

The unbounded arm computes an exhaustion depth with a dedup-by-node frontier walk
(O(N) per hop, not O(paths)) and then runs the SAME bounded pair-join loop the
`-[*1..k]->` arm uses. Deduping by endpoint changes no walk's EXISTENCE, only its
multiplicity, which the bounded loop reproduces in full.

Only ``min_hops <= 1`` reaches the unbounded arm — the gate in ``row_pipeline.py``
declines ``-[*2..]->`` because pandas' ``step_pairs`` prune by min_hops against a
dedup-by-node eccentricity that the raw-edge reconstruction here cannot reproduce.

openCypher TRAIL semantics: when ``pairs`` carries the stable
``__gfql_edge_ident__`` column, each expansion hop filters the new edge against
every edge already bound on the path (this segment's and prior elements', via
``trail_cols_in``) and records it as a ``__gfql_trail_*`` column — mirroring the
pandas ``_gfql_multihop_binding_rows`` twin exactly.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, TYPE_CHECKING
from graphistry.compute.gfql.identifiers import (
    TRAIL_EDGE_IDENT_COL,
    WALK_CURRENT_COL,
    WALK_FROM_COL,
    WALK_TO_COL,
    trail_column_name,
)

if TYPE_CHECKING:
    import polars as pl



def _directed_varlen_reachable_polars(
    state: "pl.LazyFrame",
    pairs: "pl.LazyFrame",
    state_cols: List[str],
    *,
    min_hops: int,
    max_hops: int,
    trail_cols_in: Optional[List[str]] = None,
) -> Tuple["pl.LazyFrame", List[str]]:
    """Bounded DIRECTED variable-length expansion of a bindings path bag.

    One row per distinct edge SEQUENCE under trail semantics: ``pairs`` is not
    deduped (parallel edges multiply per hop), and when it carries
    ``__gfql_edge_ident__`` a relationship binds at most once per path. Zero-hop
    rows (``min_hops == 0``) keep the seed row (endpoint == start) and come first,
    then hop 1, 2, ... — the same ``reachable`` concat order pandas builds.

    Stays fully lazy: all ``max_hops`` iterations are built without an eager
    ``.height`` early-break, because an empty intermediate lazily joins to empty and
    yields the identical result (pandas' break is an optimization, not semantics).

    Returns ``(frame, new_trail_cols)``; hop-k rows carry k trail columns, rows
    from shallower hops null-fill the deeper ones (diagonal concat).
    """
    import polars as pl

    trail = TRAIL_EDGE_IDENT_COL in pairs.collect_schema().names()
    outer_trail = list(trail_cols_in or [])
    segment_trail_cols: List[str] = []

    reachable: List["pl.LazyFrame"] = [state.select(state_cols)] if min_hops == 0 else []
    current = state
    for hop in range(1, max_hops + 1):
        current = current.join(pairs, left_on=WALK_CURRENT_COL, right_on=WALK_FROM_COL, how="inner")
        if trail:
            for used_col in outer_trail + segment_trail_cols:
                current = current.filter(
                    (pl.col(TRAIL_EDGE_IDENT_COL) != pl.col(used_col)) | pl.col(used_col).is_null()
                )
            hop_trail_col = trail_column_name(len(outer_trail) + len(segment_trail_cols))
            current = current.rename({TRAIL_EDGE_IDENT_COL: hop_trail_col})
            segment_trail_cols.append(hop_trail_col)
        current = current.drop(WALK_CURRENT_COL).rename({WALK_TO_COL: WALK_CURRENT_COL})
        current = current.select(state_cols + segment_trail_cols)
        if hop >= min_hops:
            reachable.append(current)
    if not reachable:
        return state.limit(0), segment_trail_cols
    return pl.concat(reachable, how="diagonal"), segment_trail_cols


def _directed_fixed_point_binding_rows_polars(
    state: "pl.LazyFrame",
    pairs: "pl.LazyFrame",
    state_cols: List[str],
    *,
    min_hops: int,
) -> Tuple["pl.LazyFrame", List[str]]:
    """Unbounded DIRECTED variable-length binding rows (``-[*0..]->`` / ``-[*]->``), #1709.

    The native twin of the ``max_hops is None and to_fixed_point`` arm of pandas'
    ``RowPipelineMixin._gfql_multihop_binding_rows``. One row per distinct edge
    SEQUENCE (Cypher trail multiplicity): ``pairs`` is never deduped, so parallel
    edges multiply per hop exactly as the pandas merge does. ``min_hops == 0``
    contributes the zero-hop rows (endpoint == start) first, then hop 1, 2, ... —
    the same ``reachable`` concat order pandas builds.

    Pandas discovers the traversal depth by expanding the PATH frontier until it is
    empty. This lowering splits that into (a) a cheap dedup-by-node frontier walk
    that computes the exhaustion depth ``D``, then (b) the SAME lazy bounded
    pair-join loop the ``-[*1..k]->`` arm uses, with ``max_hops = D``.

    Cyclic reachability: the node-frontier probe cannot see trail exhaustion (a
    cycle keeps the node frontier alive even though trails are finite), so a
    reachable cycle still raises the terminating-segments error here — the pandas
    twin now serves those via trail tracking; this decline is the #1903 residual.
    """
    import polars as pl
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

    pairs_df = _lazy_collect(pairs)
    pairs_lf = pairs_df.lazy()
    pairs_step = pairs_lf.select([WALK_FROM_COL, WALK_TO_COL])

    # (a) depth probe: dedup-by-node frontier, so each hop costs O(N) not O(paths).
    #
    # The loop is bounded by the REACHABLE node count, accumulated as the walk goes — not
    # by the graph's node count. A walk of length ``hop`` visits ``hop + 1`` nodes, every
    # one of them a seed or a frontier member, i.e. all inside ``seen``; so the moment
    # ``hop + 1`` exceeds ``seen.height`` some node has repeated, and a repeat on a walk IS
    # a reachable cycle. An acyclic reachable subgraph empties the frontier before the
    # bound; a reachable cycle keeps it non-empty past it.
    frontier = _lazy_collect(state.select(pl.col(WALK_CURRENT_COL)).unique())
    seen = frontier
    frontier_lf = frontier.lazy()
    depth = 0
    hop = 0
    # No matched edges at all -> no walk of length >= 1, so the fixed point is depth 0.
    exhausted = pairs_df.height == 0
    while not exhausted:
        hop += 1
        frontier = _lazy_collect(
            frontier_lf.join(pairs_step, left_on=WALK_CURRENT_COL, right_on=WALK_FROM_COL, how="inner")
            .select(pl.col(WALK_TO_COL).alias(WALK_CURRENT_COL))
            .unique()
        )
        if frontier.height == 0:
            exhausted = True
            break
        seen = pl.concat([seen, frontier], how="vertical_relaxed").unique()
        if hop >= seen.height:  # hop + 1 nodes visited, all within `seen` -> a repeat
            break
        frontier_lf = frontier.lazy()
        depth = hop
    if not exhausted:
        RowPipelineMixin._gfql_bindings_error(
            "Cypher multi-alias row bindings currently require terminating variable-length segments"
        )

    # (b) the SAME bounded expansion the `-[*1..k]->` arm runs, with max_hops = depth.
    return _directed_varlen_reachable_polars(
        state, pairs_lf, state_cols, min_hops=min_hops, max_hops=depth
    )
