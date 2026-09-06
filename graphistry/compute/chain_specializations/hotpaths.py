"""The pandas/cuDF chain specializations: the single-node lane, the seeded typed single hop
and the seeded typed RETURN-destination reduction. Each lane sits next to its admission
predicate (``admission.py``); ``chain.py`` only dispatches."""
# ruff: noqa: E501

from typing import List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

from graphistry.Engine import Engine, EngineAbstract, df_concat
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge, Direction
from graphistry.compute.chain_fast_paths import (
    _ids_to_key_array, _index_edge_rows, _index_node_rows, _record_native_seed_lane,
    _resident_node_id_index, _resident_seed_indexes, _seed_node_rows, _seeded_scalar_filters,
    _tag_fast_path_aliases, SeededReturn,
)
from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT, SeriesT
from .admission import _indexed_kernel_admits, native_fast_path_admits

if TYPE_CHECKING:
    from graphistry.compute.gfql.index.registry import AdjacencyIndex, NodeIdIndex


def _single_node_rows_via_index_or_filter(
    g: Plottable, n0: ASTNode, engine_abs: "EngineAbstract",
) -> DataFrameT:
    """Resolve a single node op through a resident index or the canonical filter."""
    from graphistry.compute.filter_by_dict import filter_by_dict
    nodes_df = g._nodes
    assert nodes_df is not None
    if not n0.filter_dict:
        return nodes_df
    node = g._node
    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df) if node is not None else None
    if node is not None and n0f:
        nid_ctx = _resident_node_id_index(g, nodes_df, node)
        rows, how = _seed_node_rows(g, nodes_df, n0f, node, nid_ctx, n0.filter_dict)
        if how != "scan":
            _record_native_seed_lane(nodes_df, seam="native_seed_lookup", reason=how, hop_count=0,
                                     public_seed_scan=node not in n0.filter_dict)
        return rows
    return filter_by_dict(nodes_df, n0.filter_dict, engine_abs)



def _seeded_typed_hop_pandas_cudf(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Plottable]:
    """Engine-generic (pandas + cuDF) fast path for a scalar-filtered
    seeded typed 1-hop. Value-identical to the general seeded branch for the covered
    shape (all node/edge filters are plain scalars, directed) — same rows, columns,
    and dtypes; row order and RangeIndex may differ — collapsing it into a
    few DataFrame filters so a seeded lookup lands sub-ms. Uses only the shared
    pandas/cuDF DataFrame API (no numpy array drops) so the same body runs on both
    engines. Returns None to fall back for anything it does not cover (predicates,
    undirected, missing columns) — the caller then runs the general branch."""
    if direction == "undirected":
        return None

    nodes_df, edges_df = g._nodes, g._edges
    if nodes_df is None or edges_df is None:
        return None
    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)

    # seed first; valid resident indexes serve the seed, frontier and gather positionally, any decline falls back to the scan body with identical results
    ctx = _resident_seed_indexes(g, nodes_df, edges_df, node, src, dst, direction) if n0f else None
    seed_nodes = edges = cand = None
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        seed_nodes, _ = _seed_node_rows(g, nodes_df, n0f, node, (nid, xp, idx_engine), n0.filter_dict)
        edges = _index_edge_rows(adj, seed_nodes[node], xp, idx_engine, edges_df)
        if edges is not None:
            if ef:
                for k, v in ef.items():
                    edges = edges[edges[k] == v]
            if 'cudf' in str(type(edges).__module__):
                import cudf as _cd  # type: ignore
                endpoint_ids = _cd.concat([edges[src], edges[dst]])
            else:
                import pandas as _pd
                endpoint_ids = _pd.concat([edges[src], edges[dst]])
            cand = _index_node_rows(nid, endpoint_ids, xp, idx_engine, nodes_df)
    served_via_index = cand is not None
    if cand is None:
        if n0f:
            seed_nodes = nodes_df
            for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
                seed_nodes = seed_nodes[seed_nodes[k] == v]
            edges = edges_df[edges_df[from_col].isin(seed_nodes[node].dropna())]
        else:
            edges = edges_df
        if ef:  # typed edge (edge_match) — now on the reduced frontier
            for k, v in ef.items():
                edges = edges[edges[k] == v]

        # membership sets are dropna()'d: null ids never link, matching the full path's joins
        cand = nodes_df[
            nodes_df[node].isin(edges[src].dropna()) | nodes_df[node].isin(edges[dst].dropna())
        ].drop_duplicates(subset=[node])
    assert edges is not None and cand is not None  # both branches above assign
    if served_via_index:
        _record_native_seed_lane(nodes_df, seam="native_seeded_hop", reason="served", hop_count=1,
                                 public_seed_scan=node not in n0f)
    if n2f:  # destination-node filter (to-side)
        n2_cand = cand
        for k, v in n2f.items():
            n2_cand = n2_cand[n2_cand[k] == v]
        n2_ok = n2_cand[node]
    else:
        n2_ok = cand[node]
    to_vals = edges[to_col]
    keep = edges[src].isin(cand[node].dropna()) & edges[dst].isin(cand[node].dropna()) & to_vals.isin(n2_ok.dropna())
    edges = edges[keep]
    cand = cand[cand[node].isin(edges[src]) | cand[node].isin(edges[dst])]
    return g.nodes(cand).edges(edges)


def _seeded_typed_return_dst_pandas_cudf(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[SeededReturn]:
    """Cypher RETURN-alias fast path: like _seeded_typed_hop_pandas_cudf but
    returns ONLY the destination (RETURN-alias) node rows + surviving edges — no
    seed-node gather, no Plottable round-trip — so the seeded cypher projection
    lands sub-ms. Engine-generic (pandas + cuDF): only the shared DataFrame API,
    no numpy array drops. Returns ``(dst_node_rows, edges)`` or None to fall back."""
    if direction == "undirected":
        return None
    nodes_df, edges_df = g._nodes, g._edges
    if nodes_df is None or edges_df is None:
        return None
    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None or not n0f:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)
    # id filter first, then the object filters on the survivors; membership sets are dropna()'d so null ids never link
    ctx = _resident_seed_indexes(g, nodes_df, edges_df, node, src, dst, direction)
    nid_ctx = (ctx[0], ctx[2], ctx[3]) if ctx is not None else _resident_node_id_index(g, nodes_df, node)
    seed_nodes, how = _seed_node_rows(g, nodes_df, n0f, node, nid_ctx, n0.filter_dict)
    edges = dstn = None
    kernel_admits = False
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        edges = _index_edge_rows(adj, seed_nodes[node], xp, idx_engine, edges_df)
        kernel_admits = _indexed_kernel_admits(
            seed_nodes, edges, n0f, node, how, ctx, len(nodes_df), len(edges_df))
        if edges is not None:
            if ef:
                for k, v in ef.items():
                    edges = edges[edges[k] == v]
            dstn = _index_node_rows(nid, edges[to_col], xp, idx_engine, nodes_df)
    if dstn is None:
        edges = edges_df[edges_df[from_col].isin(seed_nodes[node].dropna())]
        if ef:
            for k, v in ef.items():
                edges = edges[edges[k] == v]
        # destination nodes: real nodes that are to-endpoints of the surviving edges
        dstn = nodes_df[nodes_df[node].isin(edges[to_col].dropna())]
    assert edges is not None and dstn is not None  # both branches above assign
    if n2f:
        for k, v in n2f.items():
            dstn = dstn[dstn[k] == v]
    edges = edges[edges[to_col].isin(dstn[node].dropna())]
    dstn = dstn[dstn[node].isin(edges[to_col].dropna())].drop_duplicates(subset=[node])
    return dstn, edges, seed_nodes, kernel_admits



def _try_chain_fast_path(
    g_in: Plottable,
    ops: List[ASTObject],
    engine_concrete: Engine,
    start_nodes: Optional[DataFrameT] = None,
) -> Optional[Plottable]:
    """Degenerate-shape fast path (pandas/cuDF): node-only ``MATCH (n)`` or a plain
    single-hop ``MATCH (a)-[e]->(b)`` skip the forward/backward/combine BFS machinery.
    Returns the result Plottable, or ``None`` to fall through to the full path.

    Same node/edge sets + VALUES as the full machinery (trackA_golden + hop/chain
    suites); the 1-hop additionally preserves int node dtypes (the full path upcasts
    int→float via merge — the merge is the artifact, int is the Cypher-conformant type).
    Gated to unqueried nodes + a plain single-hop edge; NAMED ops are served (the alias
    flags are reconstructed by `_tag_fast_path_aliases`) except when undirected or when
    the same alias is reused. filtered-undirected and seeded chains fall through.
    polars/dask/spark also fall through (own fast path / lazy semantics)."""
    from graphistry.compute.filter_by_dict import filter_by_dict

    shape = native_fast_path_admits(ops, engine_concrete, start_nodes)
    if shape is None:
        return None
    engine_abs = EngineAbstract(engine_concrete.value)

    def _materialize_fast_path_graph() -> Plottable:
        from graphistry.compute.ComputeMixin import _coerce_input_formats  # lazy — avoids circular import
        g = g_in.materialize_nodes(engine=EngineAbstract(engine_concrete.value))
        return _coerce_input_formats(g, engine_concrete)

    if shape == "single-node":
        n0 = ops[0]
        assert isinstance(n0, ASTNode)  # the predicate admitted this shape
        g = _materialize_fast_path_graph()
        if g._nodes is None:
            return None
        nodes = _single_node_rows_via_index_or_filter(g, n0, engine_abs)
        if n0._name is not None:
            alias_was_column = n0._name in nodes.columns
            if alias_was_column:
                nodes = nodes.drop(columns=[n0._name])
            nodes = nodes.assign(**{n0._name: True})
            other_columns = [c for c in nodes.columns if c != n0._name]
            if g._node in other_columns:
                other_columns = [g._node, *[c for c in other_columns if c != g._node]]
            if alias_was_column:
                nodes = nodes[[*other_columns, n0._name]]
            else:
                nodes = nodes[[*other_columns[:1], n0._name, *other_columns[1:]]]
            nodes = nodes.reset_index(drop=True)
        edges = g._edges.iloc[0:0] if g._edges is not None else None
        return g.nodes(nodes).edges(edges) if edges is not None else g.nodes(nodes)

    n0, e1, n2 = ops
    assert isinstance(n0, ASTNode) and isinstance(e1, ASTEdge) and isinstance(n2, ASTNode)  # the predicate admitted this shape
    alias_n0, alias_e1, alias_n2 = n0._name, e1._name, n2._name
    direction = e1.direction
    unconstrained = not n0.filter_dict and not n2.filter_dict
    g = _materialize_fast_path_graph()
    if g._nodes is None or g._edges is None:
        return None
    src, dst, node = g._source, g._destination, g._node
    if src is None or dst is None or node is None:
        return None  # no edge/node bindings -> can't fast-path; full path handles it
    if alias_n0 == node or alias_n2 == node:
        return None  # a node alias equal to the node-id binding: the full path raises, never serve
    if alias_e1 is not None and direction in ("forward", "reverse") \
            and alias_e1 == (src if direction == "forward" else dst):
        return None  # an edge alias equal to the from-side binding: lanes disagree on the node set
    concat = df_concat(engine_concrete)
    if unconstrained:
        node_ids = g._nodes[node].dropna()  # validate both endpoints; NaN ids never match
        edges = g._edges[g._edges[src].isin(node_ids) & g._edges[dst].isin(node_ids)]
        if e1.edge_match:
            edges = filter_by_dict(edges, e1.edge_match, engine_abs)
    else:
        if engine_concrete in (Engine.PANDAS, Engine.CUDF):  # seed-first: reduce edges by the node filters before the edge scan
            _fast_res = _seeded_typed_hop_pandas_cudf(g, n0, n2, e1, src, dst, node, direction)
            if _fast_res is not None:
                return _tag_fast_path_aliases(
                    _fast_res, alias_n0, alias_e1, alias_n2, src, dst, node, direction)
        from_col, to_col = (src, dst) if direction == "forward" else (dst, src)
        edges = g._edges
        if n0.filter_dict:
            from_ids = filter_by_dict(g._nodes, n0.filter_dict, engine_abs)[node]
            edges = edges[edges[from_col].isin(from_ids)]
        if e1.edge_match:
            edges = filter_by_dict(edges, e1.edge_match, engine_abs)
        if n2.filter_dict:
            to_present = edges[to_col].dropna().unique()
            to_nodes = filter_by_dict(
                g._nodes[g._nodes[node].isin(to_present)], n2.filter_dict, engine_abs)
            edges = edges[edges[to_col].isin(to_nodes[node])]
        ep = concat([
            edges[[src]].rename(columns={src: node}),
            edges[[dst]].rename(columns={dst: node}),
        ]).drop_duplicates()
        cand = g._nodes[g._nodes[node].isin(ep[node])].drop_duplicates(subset=[node])
        valid = cand[node].dropna()
        edges = edges[edges[src].isin(valid) & edges[dst].isin(valid)]
        final = concat([
            edges[[src]].rename(columns={src: node}),
            edges[[dst]].rename(columns={dst: node}),
        ]).drop_duplicates()
        nodes = cand[cand[node].isin(final[node])]
        return _tag_fast_path_aliases(
            g.nodes(nodes).edges(edges), alias_n0, alias_e1, alias_n2, src, dst, node, direction)
    endpoints = concat([
        edges[[src]].rename(columns={src: node}),
        edges[[dst]].rename(columns={dst: node}),
    ]).drop_duplicates()
    nodes = g._nodes[g._nodes[node].isin(endpoints[node])]
    nodes = nodes.drop_duplicates(subset=[node])  # the full path's merge collapses duplicate node-id rows
    return _tag_fast_path_aliases(
        g.nodes(nodes).edges(edges), alias_n0, alias_e1, alias_n2, src, dst, node, direction)
