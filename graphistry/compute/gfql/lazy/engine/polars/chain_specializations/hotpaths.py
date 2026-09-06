"""The polars chain specializations: the plain single-hop branches (resident-index consult and
the skip-combine pass), the seeded lane and the seeded typed RETURN-destination reduction. Each
lane sits next to its admission predicate (``admission.py``); ``chain.py`` only dispatches."""
# ruff: noqa: E501

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union, cast

from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge, Direction
from graphistry.compute.chain_specializations.admission import _indexed_kernel_admits
from graphistry.compute.chain_fast_paths import (
    _ids_to_key_array, _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows, _seeded_scalar_filters,
    SeededReturn,
)
from graphistry.compute.endpoint_utils import drop_null_endpoint_edges
from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT
from ..dtypes import endpoint_ids
from ..hop_eager import ensure_nodes_polars
from ..predicates import filter_by_dict_polars
from .admission import polars_seeded_lane_admits

if TYPE_CHECKING:
    import polars as pl
    from graphistry.compute.gfql.index.registry import AdjacencyIndex, NodeIdIndex
    from ..dtypes import PolarsFrame


def _plain_seeded_index_hop_polars(g: Plottable, ops: Sequence[ASTObject]) -> Optional[Plottable]:
    """Consult the resident index for the plain seeded single hop; None when it declines."""
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index import get_index_policy, get_registry, maybe_index_hop
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
    n0, e1 = ops[0], ops[1]
    assert isinstance(n0, ASTNode) and isinstance(e1, ASTEdge)  # the predicate admitted this shape
    policy = get_index_policy(g)
    if get_registry(g).is_empty() and policy not in ("auto", "force"):
        return None
    gf0 = ensure_nodes_polars(g)
    seed0 = filter_by_dict_polars(gf0._nodes, n0.filter_dict)
    engine = Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else Engine.POLARS
    return maybe_index_hop(
        gf0, engine, nodes=seed0, hops=1, direction=e1.direction,
        return_as_wave_front=False, to_fixed_point=False, policy=policy,
    )


def _plain_single_hop_polars(g: Plottable, ops: Sequence[ASTObject]) -> Plottable:
    """Serve the plain single hop as one endpoint-filter pass, skipping forward/backward/combine."""
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.chain import _align_edge_endpoints, _restore_edge_dtypes
    n0, e1, n2 = ops
    assert isinstance(n0, ASTNode) and isinstance(e1, ASTEdge) and isinstance(n2, ASTNode)  # the predicate admitted this shape
    node_table_bound = g._nodes is not None
    gf = ensure_nodes_polars(g)
    ncol, scol, dcol = gf._node, gf._source, gf._destination
    assert ncol is not None and scol is not None and dcol is not None
    gf, restore = _align_edge_endpoints(gf, ncol, scol, dcol)
    edges = drop_null_endpoint_edges(gf._edges, scol, dcol)
    n_from, n_to = (n0, n2) if e1.direction != "reverse" else (n2, n0)
    all_ids = gf._nodes.select(pl.col(ncol))

    def _filter_ids(node_op: ASTNode) -> "Optional[PolarsFrame]":
        if not node_op.filter_dict:
            return None
        return filter_by_dict_polars(gf._nodes, node_op.filter_dict).select(pl.col(ncol))

    filter_sides = ((scol, _filter_ids(n_from)), (dcol, _filter_ids(n_to)))
    for endpoint_col, filter_ids in filter_sides:
        if filter_ids is not None:
            edges = edges.join(filter_ids, left_on=endpoint_col, right_on=ncol, how="semi")
    # A filtered side drew its ids FROM the node table; a synthesized one is vacuously closed.
    sides_not_closed_by_a_filter = (
        [col for col, filter_ids in filter_sides if filter_ids is None]
        if node_table_bound else [])
    endpoints = endpoint_ids(edges, scol, dcol, ncol)
    if sides_not_closed_by_a_filter:
        from graphistry.compute.gfql.lazy import collect_all
        unresolvable, nodes = collect_all([
            endpoints.lazy().join(all_ids.lazy(), on=ncol, how="anti").select(pl.len()),
            gf._nodes.lazy().join(endpoints.lazy(), on=ncol, how="semi"),
        ])
        if unresolvable.item() > 0:
            for endpoint_col in sides_not_closed_by_a_filter:
                edges = edges.join(all_ids, left_on=endpoint_col, right_on=ncol, how="semi")
            nodes = gf._nodes.join(
                endpoint_ids(edges, scol, dcol, ncol), on=ncol, how="semi")
    else:
        nodes = gf._nodes.join(endpoints, on=ncol, how="semi")
    return gf.nodes(nodes, ncol).edges(_restore_edge_dtypes(edges, scol, dcol, restore), scol, dcol)


def _seeded_typed_return_dst_polars(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
    preserve_input_order: bool = False,
    index_ctx: Optional[Tuple["NodeIdIndex", "AdjacencyIndex", ArrayNamespace, "Engine"]] = None,
) -> Optional[SeededReturn]:
    """Polars analog of _seeded_typed_return_dst_pandas_cudf: same seed-first
    reduction (seed out-edges -> typed-edge filter -> destination nodes) expressed
    with polars filters, so a seeded cypher RETURN on polars/polars-gpu also lands
    sub-ms. Returns ``(dst_node_rows, edges)`` (polars frames) or None to fall back
    to the full lazy pipeline. Value-identical node set to the full path for the
    covered shape (scalar filters, directed, single hop); row order may differ."""
    import polars as pl
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars
    if direction == "undirected":
        return None
    nodes_df, edges_df = g._nodes, g._edges
    # eager polars frames only; mixed-engine node/edge frames take the full path
    if not isinstance(nodes_df, pl.DataFrame) or not isinstance(edges_df, pl.DataFrame):
        return None

    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None or not n0f:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)

    # membership sets are drop_nulls()'d (null ids never link) and passed via implode() (Series-arg is_in is deprecated)
    ctx = index_ctx if index_ctx is not None else _resident_seed_indexes(
        g, nodes_df, edges_df, node, src, dst, direction)
    nid_ctx = (ctx[0], ctx[2], ctx[3]) if ctx is not None else _resident_node_id_index(g, nodes_df, node)
    seed_nodes, how = _seed_node_rows(g, nodes_df, n0f, node, nid_ctx, n0.filter_dict)
    edges = dstn = None
    kernel_admits = False
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        edges = _index_edge_rows(
            adj, seed_nodes.get_column(node), xp, idx_engine, edges_df,
            preserve_input_order=preserve_input_order)
        kernel_admits = _indexed_kernel_admits(
            seed_nodes, edges, n0f, node, how, ctx, len(nodes_df), len(edges_df))
        if edges is not None:
            edges = filter_by_dict_polars(edges, e1.edge_match)
            dstn = _index_node_rows(nid, edges.get_column(to_col), xp, idx_engine, nodes_df)
    if dstn is None:
        from_ids = seed_nodes.get_column(node).drop_nulls()
        if from_ids.len() == 0:
            return nodes_df.clear(), edges_df.clear(), seed_nodes, kernel_admits
        edges = edges_df.filter(pl.col(from_col).is_in(from_ids.implode()))
        edges = filter_by_dict_polars(edges, e1.edge_match)
        dst_ids = edges.get_column(to_col).drop_nulls().unique()
        dstn = nodes_df.filter(pl.col(node).is_in(dst_ids.implode()))
    assert edges is not None and dstn is not None  # both branches above assign
    dstn = filter_by_dict_polars(dstn, n2.filter_dict)
    # drop dangling edges + dedup destination nodes (mirror the pandas tail)
    keep_ids = dstn.get_column(node).drop_nulls()
    edges = edges.filter(pl.col(to_col).is_in(keep_ids.implode()))
    dstn = dstn.filter(pl.col(node).is_in(edges.get_column(to_col).implode())).unique(subset=[node], maintain_order=True)
    return dstn, edges, seed_nodes, kernel_admits



def _try_seeded_chain_polars(g: Plottable, ops: Sequence[ASTObject]) -> Optional[Plottable]:
    """Serve a native directed scalar hop through the resident seed indexes, preserving
    Polars table order and aliases; declines (None) without valid resident indexes."""
    import polars as pl
    from graphistry.compute.gfql.index.api import _record_indexed_traversal
    if not polars_seeded_lane_admits(ops):
        return None
    n0, e1, n2 = ops
    assert isinstance(n0, ASTNode) and isinstance(e1, ASTEdge) and isinstance(n2, ASTNode) and n0.filter_dict  # the predicate admitted this shape
    nodes, edges = g._nodes, g._edges
    node, src, dst = g._node, g._source, g._destination
    if (not isinstance(nodes, pl.DataFrame) or not isinstance(edges, pl.DataFrame)
            or node is None or src is None or dst is None):
        return None
    if nodes.schema[node] != edges.schema[src] or nodes.schema[node] != edges.schema[dst]:
        return None
    aliases = [op._name for op in ops if op._name is not None]
    if len(aliases) != len(set(aliases)):
        return None
    if any(name in nodes.columns for name in (n0._name, n2._name) if name is not None):
        return None
    if e1._name is not None and e1._name in edges.columns:
        return None
    ctx = _resident_seed_indexes(g, nodes, edges, node, src, dst, e1.direction)
    if ctx is None:
        return None
    reduced = _seeded_typed_return_dst_polars(
        g, n0, n2, e1, src, dst, node, e1.direction, preserve_input_order=True, index_ctx=ctx)
    if reduced is None:
        return None
    _, kept_edges, _, _ = reduced
    if not isinstance(kept_edges, pl.DataFrame):
        return None
    endpoint_ids = pl.concat([kept_edges.get_column(src), kept_edges.get_column(dst)]).drop_nulls().unique()
    nid_ctx = _resident_node_id_index(g, nodes, node)
    result_nodes = None
    if nid_ctx is not None:
        nid, xp, engine = nid_ctx
        result_nodes = _index_node_rows(nid, endpoint_ids, xp, engine, nodes, preserve_input_order=True)
    if result_nodes is None:
        result_nodes = nodes.filter(pl.col(node).is_in(endpoint_ids.implode()))
        if result_nodes.get_column(node).n_unique() != result_nodes.height:
            return None
    if not isinstance(result_nodes, pl.DataFrame):
        return None
    from_col, to_col = (src, dst) if e1.direction == "forward" else (dst, src)
    flags = [
        pl.col(node).is_in(kept_edges.get_column(endpoint).implode()).fill_null(False).alias(name)
        for name, endpoint in ((n0._name, from_col), (n2._name, to_col)) if name is not None
    ]
    if flags:
        result_nodes = result_nodes.with_columns(flags)
    if e1._name is not None:
        kept_edges = kept_edges.with_columns(pl.lit(True).alias(e1._name))
    _record_indexed_traversal(
        seam="native_seeded_hop", engine=ctx[3], served=True, reason="served", hop_count=1,
        public_seed_scan=node not in n0.filter_dict, hop_details=[{"hop": 1}])
    return g.nodes(result_nodes).edges(kept_edges)
