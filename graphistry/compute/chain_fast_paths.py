"""Seeded typed-hop fast-path specializations for the chain executor.

Extracted verbatim from chain.py (#1755) to keep that orchestrator readable: the seeded
typed 1-hop pandas/cuDF reduction and the seeded typed RETURN-destination pandas/cuDF and
polars reductions. Pure code move, no behavior change. chain.py imports from here (one
direction); this module imports only leaf modules (no back-edge into chain.py).
"""
# ruff: noqa: E501

from typing import Any, Dict, Optional, Sequence, Tuple, TYPE_CHECKING, Union, cast

from graphistry.Plottable import Plottable
from .ast import ASTNode, ASTEdge, Direction
from .typing import ArrayLike, ArrayNamespace, DataFrameT, SeriesT

if TYPE_CHECKING:
    from graphistry.Engine import Engine
    from graphistry.compute.gfql.index.registry import AdjacencyIndex, NodeIdIndex


def _seeded_scalar_filters(fd: Optional[Dict[str, Any]], df: DataFrameT) -> Optional[Dict[str, Any]]:
    """Resolve a filter dict to plain scalar column==value pairs, or None to bail
    to the general path. Mirrors filter_by_dict.resolve_filter_column exactly for
    the shapes it accepts: the cypher ``label__X: True`` form maps to ``type``
    equality ONLY when no list-valued ``labels`` column exists (labels-containment
    is not scalar equality) and the frame is not edge-shaped — same precedence as
    the live resolver. Anything else (predicates, non-scalar values, absent
    columns) bails, so the full path keeps its exact semantics incl. E301."""
    from graphistry.compute.filter_by_dict import _looks_like_edge_dataframe
    if not fd:
        return {}
    cols = set(df.columns)
    out: Dict[str, Any] = {}
    for k, v in fd.items():
        if not isinstance(v, (int, float, str, bool)):
            return None  # predicate / non-scalar -> bail to the general path
        if k in cols:
            out[k] = v
        elif (isinstance(k, str) and k.startswith("label__") and v is True
              and "labels" not in cols and "type" in cols
              and not _looks_like_edge_dataframe(df)):
            out["type"] = k[len("label__"):]
        else:
            return None  # labels-list / unknown column -> bail
    return out


def _resident_seed_indexes(
    g: Plottable, nodes_df: DataFrameT, edges_df: DataFrameT,
    node: str, src: str, dst: str, direction: Direction,
) -> Optional[Tuple["NodeIdIndex", "AdjacencyIndex", ArrayNamespace, "Engine"]]:
    """(node_id_index, adjacency_index, xp, engine) when BOTH resident indexes
    validly cover this directed seeded hop on these EXACT frames (fingerprint +
    identity via get_valid), else None — callers keep the scan path, so a stale
    or absent index can never change results, only speed."""
    from graphistry.Engine import Engine, is_polars_df
    from graphistry.compute.gfql.index import get_registry
    from graphistry.compute.gfql.index.registry import EDGE_OUT_ADJ, EDGE_IN_ADJ, NODE_ID
    from graphistry.compute.gfql.index.engine_arrays import array_namespace
    registry = get_registry(g)
    if registry.is_empty():
        return None
    mod = str(type(nodes_df).__module__)
    if is_polars_df(nodes_df):
        engine = Engine.POLARS
    elif 'cudf' in mod:
        engine = Engine.CUDF
    elif mod.startswith('pandas'):
        engine = Engine.PANDAS
    else:
        return None
    kind = EDGE_OUT_ADJ if direction == "forward" else EDGE_IN_ADJ
    engines = [engine]
    if engine == Engine.POLARS:
        # an index built with explicit engine='polars-gpu' serves the same eager
        # polars frames (same numpy sidecars + polars row-gather)
        engines.append(Engine.POLARS_GPU)
    adj = nid = None
    for eng_try in engines:
        adj = registry.get_valid(kind, edges_df, (src, dst), eng_try)
        nid = registry.get_valid(NODE_ID, nodes_df, (node,), eng_try)
        if adj is not None and nid is not None:
            break
    if adj is None or nid is None:
        return None
    xp, _ = array_namespace(engine)
    # get_valid returns the union type; kind selection above guarantees the concrete classes
    return cast("NodeIdIndex", nid), cast("AdjacencyIndex", adj), xp, engine


def _ids_to_key_array(
    vals: Union["SeriesT", Sequence[Any]], keys: ArrayLike, xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    """Values (python list / Series / array) -> deduped backend array in the index
    key dtype, nulls dropped (null ids never link — matching the scan path's
    dropna semantics). None when the cast is not value-safe (mismatched families
    like str-vs-int decline to the scan path rather than risk false matches)."""
    try:
        if 'cudf' in str(type(vals).__module__):
            vals = vals.dropna()  # type: ignore[union-attr]  # cudf Series by module check
            raw = vals.values  # type: ignore[union-attr]  # device array; to_numpy() raises on nulls + round-trips host
        elif hasattr(vals, "to_numpy"):
            raw = vals.to_numpy()
        else:
            raw = vals
        arr = xp.asarray(raw)
        if arr.dtype.kind == "f":
            arr = arr[~xp.isnan(arr)]
        if arr.dtype.kind not in "iuf" or keys.dtype.kind not in "iuf":
            return None  # numeric id families only: object/str ids keep the scan path (null-object semantics)
        if arr.dtype != keys.dtype:
            common = xp.promote_types(arr.dtype, keys.dtype)
            if arr.dtype.kind in "iu" and keys.dtype.kind in "iu" and common.kind == "f":
                # int64<->uint64 promotes to float64, which collapses distinct ids
                # >= 2^53 into false matches; the scan path compares exactly -> decline.
                return None
            arr = arr.astype(common)
        return xp.unique(arr)
    except (TypeError, ValueError):
        return None


def _index_node_rows(
    nid: "NodeIdIndex", ids: Union["SeriesT", Sequence[Any]],
    xp: ArrayNamespace, engine: "Engine", nodes_df: DataFrameT,
) -> Optional[DataFrameT]:
    """Node rows whose id is in ``ids`` via the resident node-id index (positional
    gather; row order is id-sorted, covered by the value-identical contract)."""
    from graphistry.compute.gfql.index.lookup import lookup_node_rows
    from graphistry.compute.gfql.index.engine_arrays import take_rows
    arr = _ids_to_key_array(ids, nid.keys_sorted, xp)
    if arr is None:
        return None
    return take_rows(nodes_df, lookup_node_rows(nid, arr, xp), engine)


def _index_edge_rows(
    adj: "AdjacencyIndex", ids: Union["SeriesT", Sequence[Any]],
    xp: ArrayNamespace, engine: "Engine", edges_df: DataFrameT,
) -> Optional[DataFrameT]:
    """Edge rows incident to ``ids`` on the indexed side via the CSR adjacency
    (searchsorted gather; replaces the O(E) isin scan)."""
    from graphistry.compute.gfql.index.lookup import lookup_edge_rows
    from graphistry.compute.gfql.index.engine_arrays import take_rows
    arr = _ids_to_key_array(ids, adj.keys_sorted, xp)
    if arr is None:
        return None
    rows, _ = lookup_edge_rows(adj, arr, xp)
    return take_rows(edges_df, rows, engine)


def _seeded_typed_hop_pandas_cudf(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Plottable]:
    """#1755 lever-3: engine-generic (pandas + cuDF) fast path for a scalar-filtered
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

    # from-side seed FIRST: reduce edges to the seed's out-edges before the
    # edge_match compare, so the type filter runs on the tiny frontier rather than
    # all edges — this is what makes a seeded lookup sub-ms. The id filter goes
    # first (int, unique -> ~1 row in one pass) so any remaining object filters
    # (label__X->type) run on that tiny survivor frame, not the whole node table.
    # Resident-index acceleration (#1658 x #1755): when node-id + directional
    # adjacency indexes are valid for these exact frames, the O(N) seed scan, the
    # O(E) frontier isin, and the O(N) candidate gather become positional lookups.
    # Any decline (no index, stale fingerprint, unsafe id cast) falls back to the
    # scan body below — identical results either way, only speed differs.
    # Decoupled index use: the seed row lookup uses the node-id index ONLY when
    # the seed filter includes the binding column, but the frontier edge gather
    # (CSR adjacency) and candidate gathers (node-id index) engage regardless of
    # HOW the seed rows were found — their inputs are binding-column values, which
    # are the index key domain. (LDBC/user pattern: seed on the `id` PROPERTY
    # while the graph binds a different key column — previously disqualified the
    # whole index path.)
    ctx = _resident_seed_indexes(g, nodes_df, edges_df, node, src, dst, direction) if n0f else None
    seed_nodes = edges = cand = None
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        if node in n0f:
            seed_nodes = _index_node_rows(nid, [n0f[node]], xp, idx_engine, nodes_df)
        if seed_nodes is not None:
            for k, v in n0f.items():
                if k != node:
                    seed_nodes = seed_nodes[seed_nodes[k] == v]
        else:
            seed_nodes = nodes_df
            for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
                seed_nodes = seed_nodes[seed_nodes[k] == v]
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

        # Gather candidate endpoint nodes (both endpoints of surviving edges), then run
        # the dest filter, dangling-edge drop and final-node selection on the small
        # candidate/edge frames. Selecting from nodes_df keeps only real nodes, so the
        # endpoint-in-nodes check subsumes the old NaN-endpoint guard. Membership sets
        # are dropna()'d: pandas .isin matches NaN<->NaN, but the general branch's BFS
        # joins never join on null keys, so a null id/endpoint must not link.
        cand = nodes_df[
            nodes_df[node].isin(edges[src].dropna()) | nodes_df[node].isin(edges[dst].dropna())
        ].drop_duplicates(subset=[node])
    assert edges is not None and cand is not None  # both branches above assign
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
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    """#1755 cypher RETURN-alias fast path: like _seeded_typed_hop_pandas_cudf but
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
    # id-first seed reduction: filter by the id column first (int/unique -> ~1 row)
    # so any remaining object filters (label__X->type) run on the tiny survivor
    # frame, never materializing an object column over the whole node table.
    # Membership sets are dropna()'d: pandas .isin matches NaN<->NaN, but the full
    # pipeline's joins never join on null keys, so a null id/endpoint must not link.
    ctx = _resident_seed_indexes(g, nodes_df, edges_df, node, src, dst, direction)
    seed_nodes = edges = dstn = None
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        if node in n0f:
            seed_nodes = _index_node_rows(nid, [n0f[node]], xp, idx_engine, nodes_df)
        if seed_nodes is not None:
            for k, v in n0f.items():
                if k != node:
                    seed_nodes = seed_nodes[seed_nodes[k] == v]
        else:
            # property-seeded (binding col not in the filter): scan the seed row,
            # then the CSR/node-index gathers below still engage — binding-column
            # values are the index key domain no matter how the seed was found.
            seed_nodes = nodes_df
            for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
                seed_nodes = seed_nodes[seed_nodes[k] == v]
        edges = _index_edge_rows(adj, seed_nodes[node], xp, idx_engine, edges_df)
        if edges is not None:
            if ef:
                for k, v in ef.items():
                    edges = edges[edges[k] == v]
            dstn = _index_node_rows(nid, edges[to_col], xp, idx_engine, nodes_df)
    if dstn is None:
        seed_nodes = nodes_df
        for k, v in sorted(n0f.items(), key=lambda kv: 0 if kv[0] == node else 1):
            seed_nodes = seed_nodes[seed_nodes[k] == v]
        edges = edges_df[edges_df[from_col].isin(seed_nodes[node].dropna())]
        if ef:
            for k, v in ef.items():
                edges = edges[edges[k] == v]
        # destination nodes = real nodes that are edge to-endpoints, then the dest
        # filter, dangling-edge drop and dedup on the small dst/edge frames.
        dstn = nodes_df[nodes_df[node].isin(edges[to_col].dropna())]
    assert edges is not None and dstn is not None  # both branches above assign
    if n2f:
        for k, v in n2f.items():
            dstn = dstn[dstn[k] == v]
    edges = edges[edges[to_col].isin(dstn[node].dropna())]
    dstn = dstn[dstn[node].isin(edges[to_col].dropna())].drop_duplicates(subset=[node])
    return dstn, edges


def _seeded_typed_return_dst_polars(
    g: Plottable, n0: ASTNode, n2: ASTNode, e1: ASTEdge,
    src: str, dst: str, node: str, direction: Direction,
) -> Optional[Tuple[DataFrameT, DataFrameT]]:
    """#1755 polars analog of _seeded_typed_return_dst_pandas_cudf: same seed-first
    reduction (seed out-edges -> typed-edge filter -> destination nodes) expressed
    with polars filters, so a seeded cypher RETURN on polars/polars-gpu also lands
    sub-ms. Returns ``(dst_node_rows, edges)`` (polars frames) or None to fall back
    to the full lazy pipeline. Value-identical node set to the full path for the
    covered shape (scalar filters, directed, single hop); row order may differ."""
    import polars as pl
    if direction == "undirected":
        return None
    nodes_df, edges_df = g._nodes, g._edges
    # Eager polars frames only: LazyFrame has no get_column, and mixed-engine
    # node/edge frames must take the full path — decline rather than crash.
    if not isinstance(nodes_df, pl.DataFrame) or not isinstance(edges_df, pl.DataFrame):
        return None

    n0f = _seeded_scalar_filters(n0.filter_dict, nodes_df)
    n2f = _seeded_scalar_filters(n2.filter_dict, nodes_df)
    ef = _seeded_scalar_filters(e1.edge_match, edges_df)
    if n0f is None or n2f is None or ef is None or not n0f:
        return None
    from_col, to_col = (src, dst) if direction == "forward" else (dst, src)

    # from-side seed: reduce the node frame to the seed rows, take their ids.
    # Membership sets are drop_nulls()'d (null ids/endpoints never link, matching
    # the full pipeline's joins) and passed via .implode() (Series-arg is_in is
    # deprecated in polars 1.42, see polars#22149).
    ctx = _resident_seed_indexes(g, nodes_df, edges_df, node, src, dst, direction)
    seed_nodes = edges = dstn = None
    if ctx is not None:
        nid, adj, xp, idx_engine = ctx
        if node in n0f:
            seed_nodes = _index_node_rows(nid, [n0f[node]], xp, idx_engine, nodes_df)
        if seed_nodes is not None:
            for k, v in n0f.items():
                if k != node:
                    seed_nodes = seed_nodes.filter(pl.col(k) == v)
        else:
            # property-seeded: scan the seed row; CSR/node-index gathers below
            # still engage (binding-column values = index key domain).
            seed_nodes = nodes_df
            for k, v in n0f.items():
                seed_nodes = seed_nodes.filter(pl.col(k) == v)
        edges = _index_edge_rows(adj, seed_nodes.get_column(node), xp, idx_engine, edges_df)
        if edges is not None:
            for k, v in ef.items():
                edges = edges.filter(pl.col(k) == v)
            dstn = _index_node_rows(nid, edges.get_column(to_col), xp, idx_engine, nodes_df)
    if dstn is None:
        seed_nodes = nodes_df
        for k, v in n0f.items():
            seed_nodes = seed_nodes.filter(pl.col(k) == v)
        from_ids = seed_nodes.get_column(node).drop_nulls()
        if from_ids.len() == 0:
            return nodes_df.clear(), edges_df.clear()
        edges = edges_df.filter(pl.col(from_col).is_in(from_ids.implode()))
        for k, v in ef.items():  # typed edge on the reduced frontier
            edges = edges.filter(pl.col(k) == v)
        dst_ids = edges.get_column(to_col).drop_nulls().unique()
        dstn = nodes_df.filter(pl.col(node).is_in(dst_ids.implode()))
    assert edges is not None and dstn is not None  # both branches above assign
    for k, v in n2f.items():  # destination-node filter
        dstn = dstn.filter(pl.col(k) == v)
    # drop dangling edges + dedup destination nodes (mirror the pandas tail)
    keep_ids = dstn.get_column(node).drop_nulls()
    edges = edges.filter(pl.col(to_col).is_in(keep_ids.implode()))
    dstn = dstn.filter(pl.col(node).is_in(edges.get_column(to_col).implode())).unique(subset=[node], maintain_order=True)
    return dstn, edges
