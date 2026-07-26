"""Variable-length (``-[*i..k]->`` / unbounded) binding-row specializations.

Extracted from ``row_pipeline.py`` so the core row flow stays readable and so the
other in-flight PRs that touch that file do not have to rebase across ~270 lines
of specialization. Pure code move: no behaviour change.

The unbounded arm computes an exhaustion depth with a dedup-by-node frontier walk
(O(N) per hop, not O(paths)) and then runs the SAME bounded pair-join loop the
`-[*1..k]->` arm uses. Deduping by endpoint changes no walk's EXISTENCE, only its
multiplicity, which the bounded loop reproduces in full.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

# NOTE: RowPipelineMixin is imported FUNCTION-LOCALLY below, exactly as in the
# original site. Hoisting it to module scope reintroduces an import cycle
# (row.pipeline -> lazy engine -> back), so the move keeps it where it was.

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
    the same error via the same helper, and we detect it strictly: with ``N`` distinct
    nodes touched by ``pairs``, any walk of ``N`` edges visits ``N + 1`` nodes and so
    must repeat one (pigeonhole) — a non-empty frontier at hop ``N`` is a reachable
    cycle, exactly the condition under which pandas' own cap
    (``max(len(step_pairs), 1) + 1``) also fails to exhaust. Conversely an acyclic
    reachable subgraph has every walk shorter than ``N``, so both engines exhaust.
    Same outcome on both sides of the branch, without pandas' cost of expanding paths
    into the cycle before giving up.
    """
    import polars as pl
    from graphistry.compute.gfql.lazy import collect as _lazy_collect
    from graphistry.compute.gfql.row.pipeline import RowPipelineMixin

    pairs_df = _lazy_collect(pairs)
    pairs_lf = pairs_df.lazy()
    node_cap = int(
        pl.concat(
            [
                pairs_df.select(pl.col("__from__").alias("__n__")),
                pairs_df.select(pl.col("__to__").alias("__n__")),
            ],
            how="vertical",
        )
        .select(pl.col("__n__").n_unique())
        .item()
    )

    # (a) depth probe: dedup-by-node frontier, so each hop costs O(N) not O(paths).
    frontier_lf = state.select(pl.col("__current__")).unique()
    depth = 0
    exhausted = node_cap == 0  # no matched edges at all -> no walk of length >= 1
    for hop in range(1, node_cap + 1):
        frontier = _lazy_collect(
            frontier_lf.join(pairs_lf, left_on="__current__", right_on="__from__", how="inner")
            .select(pl.col("__to__").alias("__current__"))
            .unique()
        )
        if frontier.height == 0:
            exhausted = True
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
