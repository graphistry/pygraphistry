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
"""
from __future__ import annotations

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    import polars as pl


def _directed_varlen_reachable_polars(
    state: "pl.LazyFrame",
    pairs: "pl.LazyFrame",
    state_cols: List[str],
    *,
    min_hops: int,
    max_hops: int,
) -> "pl.LazyFrame":
    """Bounded DIRECTED variable-length expansion of a bindings path bag.

    One row per distinct edge SEQUENCE: ``pairs`` is NOT deduped, so parallel edges
    multiply per hop, matching pandas' ``_gfql_multihop_binding_rows`` merge. Zero-hop
    rows (``min_hops == 0``) keep the seed row (endpoint == start) and come first, then
    hop 1, 2, ... — the same ``reachable`` concat order pandas builds.

    Stays fully lazy: all ``max_hops`` iterations are built without an eager
    ``.height`` early-break, because an empty intermediate lazily joins to empty and
    yields the identical result (pandas' break is an optimization, not semantics).
    """
    import polars as pl

    reachable: List["pl.LazyFrame"] = [state] if min_hops == 0 else []
    current = state
    for hop in range(1, max_hops + 1):
        current = (
            current.join(pairs, left_on="__current__", right_on="__from__", how="inner")
            .drop("__current__")
            .rename({"__to__": "__current__"})
            .select(state_cols)
        )
        if hop >= min_hops:
            reachable.append(current)
    return pl.concat(reachable, how="vertical") if reachable else state.limit(0)


def _directed_fixed_point_binding_rows_polars(
    state: "pl.LazyFrame",
    pairs: "pl.LazyFrame",
    state_cols: List[str],
    *,
    min_hops: int,
) -> "pl.LazyFrame":
    """Unbounded DIRECTED variable-length binding rows (``-[*0..]->`` / ``-[*]->``), #1709.

    The native twin of the ``max_hops is None and to_fixed_point`` arm of pandas'
    ``RowPipelineMixin._gfql_multihop_binding_rows``. One row per distinct edge
    SEQUENCE (Cypher path multiplicity): ``pairs`` is never deduped, so parallel
    edges multiply per hop exactly as the pandas merge does. ``min_hops == 0``
    contributes the zero-hop rows (endpoint == start) first, then hop 1, 2, ... —
    the same ``reachable`` concat order pandas builds.

    Pandas discovers the traversal depth by expanding the PATH frontier until it is
    empty, which materializes every partial path at every hop. This lowering splits
    that into (a) a cheap dedup-by-node frontier walk that computes the exhaustion
    depth ``D``, then (b) the SAME lazy bounded pair-join loop the ``-[*1..k]->`` arm
    uses, with ``max_hops = D``. Step (a) is exact, not an approximation: the path
    frontier at hop ``h`` is non-empty iff a walk of length ``h`` leaves some seed,
    and deduping by endpoint node changes no walk's EXISTENCE — only its
    multiplicity, which step (b) then reproduces in full. So the emitted rows are
    identical to pandas' while the exponential path blow-up happens once, lazily.

    Non-termination: a cycle reachable from a seed makes the walk infinite, and
    pandas raises ``E108`` ("require terminating variable-length segments"). We raise
    the same error via the same helper, and we detect it strictly, by pigeonhole over
    the REACHABLE set: a walk of ``h`` edges visits ``h + 1`` nodes, all of them seeds
    or frontier members, so once ``h + 1`` exceeds the number of nodes seen the walk has
    repeated a node — a reachable cycle, exactly the condition under which pandas' own
    cap (``max(len(step_pairs), 1) + 1``) also fails to exhaust. Conversely an acyclic
    reachable subgraph empties the frontier first, so both engines exhaust. Same outcome
    on both sides of the branch, without pandas' cost of expanding paths into the cycle
    before giving up — and the bound is the reachable node count, so an unreachable
    remainder of the graph costs nothing (see the probe's own note).
    """
    import polars as pl
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

    pairs_df = _lazy_collect(pairs)
    pairs_lf = pairs_df.lazy()

    # (a) depth probe: dedup-by-node frontier, so each hop costs O(N) not O(paths).
    #
    # The loop is bounded by the REACHABLE node count, accumulated as the walk goes — not
    # by the graph's node count. A walk of length ``hop`` visits ``hop + 1`` nodes, every
    # one of them a seed or a frontier member, i.e. all inside ``seen``; so the moment
    # ``hop + 1`` exceeds ``seen.height`` some node has repeated, and a repeat on a walk IS
    # a reachable cycle. That is the same pigeonhole argument as a global-N bound but over
    # the only set that matters, and it is still exact in both directions: an acyclic
    # reachable subgraph empties the frontier before the bound, and a reachable cycle keeps
    # it non-empty past it. Bounding by the global count instead makes a two-node cycle
    # reachable from one seed cost O(graph) eager collects before raising — measured linear
    # in the GLOBAL node count, which at LDBC SF1 scale is minutes of spinning to reach a
    # validation error.
    frontier = _lazy_collect(state.select(pl.col("__current__")).unique())
    seen = frontier
    frontier_lf = frontier.lazy()
    depth = 0
    hop = 0
    # No matched edges at all -> no walk of length >= 1, so the fixed point is depth 0.
    exhausted = pairs_df.height == 0
    while not exhausted:
        hop += 1
        frontier = _lazy_collect(
            frontier_lf.join(pairs_lf, left_on="__current__", right_on="__from__", how="inner")
            .select(pl.col("__to__").alias("__current__"))
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
