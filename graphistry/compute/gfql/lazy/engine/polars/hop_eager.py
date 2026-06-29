"""Native Polars hop() — Phase 1, vectorized.

Forward / reverse / undirected, integer hops (and to_fixed_point), default +
return_as_wave_front seed semantics, edge_match / source_node_match /
destination_node_match predicates, and target_wave_front (chain reverse pass).

Vectorization-first: the BFS keeps frontier / visited / allowed sets as polars
frames and advances via semi/anti joins — no Python-level per-element work, no
``.to_list()`` / ``is_in(python_list)`` ping-pong. Each hop is one big join.

Not yet ported (explicit NotImplementedError): hop labeling, min_hops>1,
output_min/max slicing, *_query strings, prune_to_endpoints,
include_zero_hop_seed. Parity vs pandas is the oracle.
"""
from typing import Any, Optional

from graphistry.Plottable import Plottable
from graphistry.compute.util import generate_safe_column_name
from .predicates import filter_by_dict_polars


def _unsupported(**kwargs: Any) -> None:
    unsupported = [k for k, v in kwargs.items() if v is not None and v is not False]
    if unsupported:
        raise NotImplementedError(
            "polars hop engine (Phase 1) does not yet support: "
            + ", ".join(sorted(unsupported))
            + ". Use engine='pandas' or extend graphistry/compute/gfql/lazy/engine/polars/hop_eager.py."
        )


def ensure_nodes_polars(g: Plottable) -> Plottable:
    """Materialize a polars node table from edges when absent (native — avoids the
    pandas-idiom ``materialize_nodes`` path, which uses drop_duplicates/reset_index)."""
    import polars as pl

    if g._nodes is not None:
        return g
    src, dst = g._source, g._destination
    assert src is not None and dst is not None and g._edges is not None
    node_id = g._node if g._node is not None else "id"
    ids = pl.concat(
        [g._edges.select(pl.col(src).alias(node_id)), g._edges.select(pl.col(dst).alias(node_id))],
        how="vertical_relaxed",
    ).unique()
    return g.nodes(ids, node_id)


def hop_polars(
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
) -> Plottable:
    import polars as pl
    from graphistry.Engine import Engine, df_to_engine

    if direction not in ("forward", "reverse", "undirected"):
        raise ValueError(
            f'Invalid direction: "{direction}", must be one of: '
            '"forward" (default), "reverse", "undirected"'
        )

    _unsupported(
        min_hops=min_hops if (min_hops is not None and min_hops > 1) else None,
        output_min_hops=output_min_hops,
        output_max_hops=output_max_hops,
        label_node_hops=label_node_hops,
        label_edge_hops=label_edge_hops,
        label_seeds=label_seeds or None,
        source_node_query=source_node_query,
        destination_node_query=destination_node_query,
        edge_query=edge_query,
        include_zero_hop_seed=include_zero_hop_seed or None,
    )

    if target_wave_front is not None and nodes is None:
        raise ValueError("target_wave_front requires nodes to target against (for intermediate hops)")

    if nodes is not None:
        nodes = df_to_engine(nodes, Engine.POLARS)
    if target_wave_front is not None:
        target_wave_front = df_to_engine(target_wave_front, Engine.POLARS)

    g = ensure_nodes_polars(self)

    node_col = g._node
    src = g._source
    dst = g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._edges is not None and g._nodes is not None
    edges = g._edges
    all_nodes = g._nodes

    if edge_match is not None:
        edges = filter_by_dict_polars(edges, edge_match)

    resolved_max_hops = max_hops if max_hops is not None else hops
    if to_fixed_point:
        resolved_max_hops = None
    elif not isinstance(resolved_max_hops, int):
        raise ValueError(
            f"Must provide integer hops when to_fixed_point is False, received: {resolved_max_hops}"
        )

    FROM = generate_safe_column_name("__gfql_from__", edges, prefix="__gfql_", suffix="__")
    TO = generate_safe_column_name("__gfql_to__", edges, prefix="__gfql_", suffix="__")
    NID = generate_safe_column_name("__gfql_nid__", all_nodes, prefix="__gfql_", suffix="__")

    # Reuse an existing edge-id binding (e.g. chain's __gfql_edge_index__) rather
    # than synthesizing a second monotonic index over the full edge table.
    if g._edge is not None and g._edge in edges.columns:
        EID = g._edge
        edges_idx = edges
        synth_eid = False
    else:
        EID = generate_safe_column_name("__gfql_eid__", edges, prefix="__gfql_", suffix="__")
        edges_idx = edges.with_row_index(EID)
        synth_eid = True

    # Align join-key dtype: node ids and edge endpoints must share a dtype for
    # polars joins (pandas coerces int/float; polars does not).
    node_dtype = all_nodes.schema[node_col]

    def _pairs(s: str, d: str) -> "pl.DataFrame":
        return edges_idx.select(
            pl.col(s).cast(node_dtype).alias(FROM),
            pl.col(d).cast(node_dtype).alias(TO),
            pl.col(EID),
        )

    if direction == "forward":
        pairs = _pairs(src, dst)
    elif direction == "reverse":
        pairs = _pairs(dst, src)
    else:
        pairs = pl.concat([_pairs(src, dst), _pairs(dst, src)], how="vertical_relaxed")

    def _idframe(df, col) -> "pl.DataFrame":
        return df.select(pl.col(col).cast(node_dtype).alias(NID)).unique()

    allowed_source = (
        _idframe(filter_by_dict_polars(nodes if (nodes is not None and not to_fixed_point and resolved_max_hops == 1) else all_nodes, source_node_match), node_col)
        if source_node_match is not None else None
    )
    allowed_dest = (
        _idframe(filter_by_dict_polars(all_nodes, destination_node_match), node_col)
        if destination_node_match is not None else None
    )
    target_final = _idframe(target_wave_front, node_col) if target_wave_front is not None else None

    seed = _idframe(nodes if nodes is not None else all_nodes, node_col)

    empty_ids = all_nodes.select(pl.col(node_col).cast(node_dtype).alias(NID)).clear()
    frontier = seed                      # DataFrame[NID]
    visited_nodes = empty_ids            # DataFrame[NID]
    visited_edge_frames = []             # collect per-hop EID frames; concat once at end
    current_hop = 0
    first = True

    while True:
        if not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops:
            break
        if frontier.height == 0:
            break
        current_hop += 1

        frontier_iter = frontier if allowed_source is None else frontier.join(allowed_source, on=NID, how="semi")
        hop_edges = pairs.join(frontier_iter.rename({NID: FROM}), on=FROM, how="semi")

        is_last = not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops
        if target_final is not None and is_last:
            hop_edges = hop_edges.join(target_final.rename({NID: TO}), on=TO, how="semi")
        if allowed_dest is not None:
            hop_edges = hop_edges.join(allowed_dest.rename({NID: TO}), on=TO, how="semi")

        if first and not return_as_wave_front:
            visited_nodes = hop_edges.select(pl.col(FROM).alias(NID)).unique()
        first = False

        visited_edge_frames.append(hop_edges.select(pl.col(EID)))

        cand = hop_edges.select(pl.col(TO).alias(NID)).unique()
        new_frontier = cand.join(visited_nodes, on=NID, how="anti")
        visited_nodes = pl.concat([visited_nodes, new_frontier], how="vertical_relaxed").unique(subset=[NID])
        frontier = new_frontier

    if visited_edge_frames:
        visited_edges = pl.concat(visited_edge_frames, how="vertical_relaxed").unique(subset=[EID])
    else:
        visited_edges = edges_idx.select(pl.col(EID)).clear()

    out_edges = edges_idx.join(visited_edges, on=EID, how="semi")
    if synth_eid:
        out_edges = out_edges.drop(EID)

    # Final node set: reached ∪ (edge endpoints, unless wavefront-with-seeds).
    needed = visited_nodes
    materialize_endpoints = not (return_as_wave_front and nodes is not None)
    if out_edges.height > 0 and materialize_endpoints:
        endpoints = pl.concat(
            [out_edges.select(pl.col(src).cast(node_dtype).alias(NID)),
             out_edges.select(pl.col(dst).cast(node_dtype).alias(NID))],
            how="vertical_relaxed",
        ).unique(subset=[NID])
        needed = pl.concat([needed, endpoints], how="vertical_relaxed").unique(subset=[NID])

    out_nodes = all_nodes.join(needed.rename({NID: node_col}), on=node_col, how="semi")

    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
