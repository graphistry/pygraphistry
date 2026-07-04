"""Lazy Polars hop — build ONE ``pl.LazyFrame`` plan, collect ONCE on the target.

Mirrors the eager ``hop_eager`` join logic but (a) unrolls the fixed-hop BFS into one lazy plan —
the eager loop's only data-dependent control flow is the ``frontier.height==0`` early-break,
which merely short-circuits, so for a fixed hop count the straight-line plan is equivalent
(empty frontier → empty joins) — and (b) materializes out_edges + out_nodes in ONE
``collect_all`` so their shared edge-table subplan is read/transferred once. That single collect
is what makes GPU pay off vs the eager engine's many small per-op collects.

DRY: reuses ensure_nodes_polars / filter_by_dict_polars / generate_safe_column_name from the
eager engine verbatim. Returns None for anything uncovered (to_fixed_point, labels, min_hops>1,
*_query, output slicing) so the dispatcher falls back to the eager hop. Parity-gated vs eager.
"""
from typing import Any, Optional

from graphistry.Plottable import Plottable
from graphistry.compute.util import generate_safe_column_name
from graphistry.compute.gfql.lazy.engine.polars.dtypes import endpoint_ids
from graphistry.compute.gfql.lazy.engine.polars.hop_eager import ensure_nodes_polars
from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
from graphistry.compute.gfql.lazy import collect_all


def hop_lazy_or_eager(self: Plottable, nodes: Optional[Any] = None, hops: Optional[int] = 1, **kwargs: Any) -> Plottable:
    """Polars hop entry: lazy single-hop (collect-once) if covered, else eager hop. The single
    point both hop() dispatch and chain_polars go through, so chains get the lazy collect-once
    path (and the GPU target) per edge."""
    result = hop_polars_lazy(self, nodes, hops, **kwargs)
    if result is not None:
        return result
    from graphistry.compute.gfql.lazy.engine.polars.hop_eager import hop_polars
    return hop_polars(self, nodes, hops, **kwargs)


def hop_polars_lazy(
    self: Plottable,
    nodes: Optional[Any] = None,
    hops: Optional[int] = 1,
    *,
    min_hops: Optional[int] = None,
    max_hops: Optional[int] = None,
    output_min_hops: Optional[int] = None,
    output_max_hops: Optional[int] = None,
    label_node_hops: Optional[str] = None,
    label_edge_hops: Optional[str] = None,
    label_seeds: bool = False,
    to_fixed_point: bool = False,
    direction: str = "forward",
    edge_match: Optional[dict] = None,
    source_node_match: Optional[dict] = None,
    destination_node_match: Optional[dict] = None,
    source_node_query: Optional[str] = None,
    destination_node_query: Optional[str] = None,
    edge_query: Optional[str] = None,
    return_as_wave_front: bool = False,
    include_zero_hop_seed: bool = False,
    target_wave_front: Optional[Any] = None,
    intermediate_universe: Optional[Any] = None,  # eager-only (multi-hop); ignored on the lazy single-hop path
    min_hops_label_policy: bool = False,           # eager-only (min_hops>1); the lazy single-hop path declines min_hops
) -> Optional[Plottable]:
    import polars as pl
    from graphistry.Engine import Engine, df_to_engine

    # --- Defer cases the lazy fast-path doesn't cover -> caller uses eager hop ---
    if (to_fixed_point or label_node_hops or label_edge_hops or label_seeds
            or (min_hops is not None and min_hops > 1)
            or output_min_hops is not None or output_max_hops is not None
            or source_node_query is not None or destination_node_query is not None
            or edge_query is not None or include_zero_hop_seed):
        return None
    if direction not in ("forward", "reverse", "undirected"):
        return None
    resolved_max_hops = max_hops if max_hops is not None else hops
    if not isinstance(resolved_max_hops, int):
        return None
    # Single-hop only (the dominant case — every chain edge is one hop): collect-once is a clean
    # win (GPU 2.84x @1M, CPU parity). For hops>=2 the unrolled plan recomputes the big edge-join
    # in later hops (polars CSE doesn't dedup the deep BFS) and loses to eager, which materializes
    # the small frontier between hops -> defer. Multi-hop lazy (collect small frontier per hop,
    # heavy join on target) is a follow-up.
    if resolved_max_hops != 1:
        return None
    if target_wave_front is not None and nodes is None:
        return None

    if nodes is not None:
        nodes = df_to_engine(nodes, Engine.POLARS)
    if target_wave_front is not None:
        target_wave_front = df_to_engine(target_wave_front, Engine.POLARS)

    g = ensure_nodes_polars(self)
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._edges is not None and g._nodes is not None
    edges = g._edges
    all_nodes = g._nodes
    if edge_match is not None:
        edges = filter_by_dict_polars(edges, edge_match)

    from graphistry.compute.gfql.lazy.engine.polars.hop_eager import _hop_setup_columns, _build_hop_pairs
    FROM, TO, NID, EID, edges_idx, synth_eid, node_dtype = _hop_setup_columns(
        edges, all_nodes, node_col, g._edge)
    edges_lf = edges_idx.lazy()
    all_nodes_lf = all_nodes.lazy()
    pairs = _build_hop_pairs(edges_lf, direction, src, dst, node_dtype, FROM, TO, EID)

    def _idframe_lf(lf: Any, col: str) -> Any:
        return lf.select(pl.col(col).cast(node_dtype).alias(NID)).unique()

    allowed_source = (
        _idframe_lf(
            # Mirror the eager hop guard verbatim (hop_eager.py): source filter scoped to the
            # seed only for a single bounded hop. to_fixed_point is already declined upstream
            # (always False here) — kept so the two copies stay textually identical.
            filter_by_dict_polars(
                nodes if (nodes is not None and not to_fixed_point and resolved_max_hops == 1) else all_nodes,
                source_node_match,
            ).lazy(),
            node_col,
        )
        if source_node_match is not None else None
    )
    allowed_dest = (
        _idframe_lf(filter_by_dict_polars(all_nodes, destination_node_match).lazy(), node_col)
        if destination_node_match is not None else None
    )
    target_final = _idframe_lf(target_wave_front.lazy(), node_col) if target_wave_front is not None else None

    seed = _idframe_lf((nodes if nodes is not None else all_nodes).lazy(), node_col)
    empty_ids = all_nodes_lf.select(pl.col(node_col).cast(node_dtype).alias(NID)).filter(pl.lit(False))

    frontier = seed
    visited_nodes = empty_ids
    visited_edge_frames = []

    for i in range(resolved_max_hops):
        is_last = i == resolved_max_hops - 1
        frontier_iter = frontier if allowed_source is None else frontier.join(allowed_source, on=NID, how="semi")
        hop_edges = pairs.join(frontier_iter.rename({NID: FROM}), on=FROM, how="semi")
        if target_final is not None and is_last:
            hop_edges = hop_edges.join(target_final.rename({NID: TO}), on=TO, how="semi")
        if allowed_dest is not None:
            hop_edges = hop_edges.join(allowed_dest.rename({NID: TO}), on=TO, how="semi")

        if i == 0 and not return_as_wave_front:
            visited_nodes = hop_edges.select(pl.col(FROM).alias(NID)).unique()

        visited_edge_frames.append(hop_edges.select(pl.col(EID)))

        cand = hop_edges.select(pl.col(TO).alias(NID)).unique()
        new_frontier = cand.join(visited_nodes, on=NID, how="anti")
        visited_nodes = pl.concat([visited_nodes, new_frontier], how="vertical_relaxed").unique(subset=[NID])
        frontier = new_frontier

    if visited_edge_frames:
        visited_edges = pl.concat(visited_edge_frames, how="vertical_relaxed").unique(subset=[EID])
    else:
        visited_edges = edges_lf.select(pl.col(EID)).filter(pl.lit(False))

    out_edges_lf = edges_lf.join(visited_edges, on=EID, how="semi")
    if synth_eid:
        out_edges_lf = out_edges_lf.drop(EID)

    # Endpoint materialization: always compute (no eager .height guard — empty out_edges
    # yields empty endpoints, a no-op concat; same result).
    needed = visited_nodes
    if not (return_as_wave_front and nodes is not None):
        endpoints = endpoint_ids(out_edges_lf, src, dst, NID, node_dtype).unique(subset=[NID])
        needed = pl.concat([needed, endpoints], how="vertical_relaxed").unique(subset=[NID])

    out_nodes_lf = all_nodes_lf.join(needed.rename({NID: node_col}), on=node_col, how="semi")

    # ONE collect: out_edges + out_nodes share the edge-table subplan -> read once
    # (on GPU: transferred once). This is the whole point of going lazy.
    out_edges, out_nodes = collect_all([out_edges_lf, out_nodes_lf])
    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
