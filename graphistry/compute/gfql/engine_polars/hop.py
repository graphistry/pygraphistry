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
            + ". Use engine='pandas' or extend graphistry/compute/gfql/engine_polars/hop.py."
        )


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

    if nodes is not None:
        nodes = df_to_engine(nodes, Engine.POLARS)
    if target_wave_front is not None:
        target_wave_front = df_to_engine(target_wave_front, Engine.POLARS)

    g = self
    if g._nodes is None:
        g = g.materialize_nodes(engine="polars")

    node_col = g._node
    src = g._source
    dst = g._destination
    edges: "pl.DataFrame" = g._edges
    all_nodes: "pl.DataFrame" = g._nodes

    if edge_match is not None:
        edges = filter_by_dict_polars(edges, edge_match)

    resolved_max_hops = max_hops if max_hops is not None else hops
    if to_fixed_point:
        resolved_max_hops = None
    elif not isinstance(resolved_max_hops, int):
        raise ValueError(
            f"Must provide integer hops when to_fixed_point is False, received: {resolved_max_hops}"
        )

    EID = generate_safe_column_name("__gfql_eid__", edges, prefix="__gfql_", suffix="__")
    FROM = generate_safe_column_name("__gfql_from__", edges, prefix="__gfql_", suffix="__")
    TO = generate_safe_column_name("__gfql_to__", edges, prefix="__gfql_", suffix="__")
    NID = generate_safe_column_name("__gfql_nid__", all_nodes, prefix="__gfql_", suffix="__")
    edges_idx = edges.with_row_index(EID)

    def _pairs(s: str, d: str) -> "pl.DataFrame":
        return edges_idx.select(pl.col(s).alias(FROM), pl.col(d).alias(TO), pl.col(EID))

    if direction == "forward":
        pairs = _pairs(src, dst)
    elif direction == "reverse":
        pairs = _pairs(dst, src)
    else:
        pairs = pl.concat([_pairs(src, dst), _pairs(dst, src)], how="vertical_relaxed")

    # All "id-set" working frames use a single canonical column NID so joins are
    # uniform regardless of FROM/TO/node_col naming.
    def _idframe(df, col) -> "pl.DataFrame":
        return df.select(pl.col(col).alias(NID)).unique()

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

    empty_ids = all_nodes.select(pl.col(node_col).alias(NID)).clear()
    frontier = seed                      # DataFrame[NID]
    visited_nodes = empty_ids            # DataFrame[NID]
    visited_edges = edges_idx.select(pl.col(EID)).clear()  # DataFrame[EID]
    current_hop = 0
    first = True

    while True:
        if not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops:
            break
        if frontier.height == 0:
            break
        current_hop += 1

        frontier_iter = frontier if allowed_source is None else frontier.join(allowed_source, on=NID, how="semi")
        # edges leaving the frontier: semi-join pairs.FROM against frontier
        hop_edges = pairs.join(frontier_iter.rename({NID: FROM}), on=FROM, how="semi")

        is_last = not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops
        if target_final is not None and is_last:
            hop_edges = hop_edges.join(target_final.rename({NID: TO}), on=TO, how="semi")
        if allowed_dest is not None:
            hop_edges = hop_edges.join(allowed_dest.rename({NID: TO}), on=TO, how="semi")

        if first and not return_as_wave_front:
            visited_nodes = hop_edges.select(pl.col(FROM).alias(NID)).unique()
        first = False

        visited_edges = pl.concat(
            [visited_edges, hop_edges.select(pl.col(EID))], how="vertical_relaxed"
        ).unique(subset=[EID])

        cand = hop_edges.select(pl.col(TO).alias(NID)).unique()
        new_frontier = cand.join(visited_nodes, on=NID, how="anti")
        visited_nodes = pl.concat([visited_nodes, new_frontier], how="vertical_relaxed").unique(subset=[NID])
        frontier = new_frontier

    out_edges = edges_idx.join(visited_edges, on=EID, how="semi").drop(EID)

    # Final node set: reached ∪ (edge endpoints, unless wavefront-with-seeds).
    needed = visited_nodes
    materialize_endpoints = not (return_as_wave_front and nodes is not None)
    if out_edges.height > 0 and materialize_endpoints:
        endpoints = pl.concat(
            [out_edges.select(pl.col(src).alias(NID)), out_edges.select(pl.col(dst).alias(NID))],
            how="vertical_relaxed",
        ).unique(subset=[NID])
        needed = pl.concat([needed, endpoints], how="vertical_relaxed").unique(subset=[NID])

    out_nodes = all_nodes.join(needed.rename({NID: node_col}), on=node_col, how="semi")

    return g.nodes(out_nodes, node_col).edges(out_edges, src, dst)
