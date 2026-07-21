"""Seeded typed-hop fast-path specializations for the chain executor.

Extracted verbatim from chain.py (#1755) to keep that orchestrator readable: the seeded
typed 1-hop pandas/cuDF reduction and the seeded typed RETURN-destination pandas/cuDF and
polars reductions. Pure code move, no behavior change. chain.py imports from here (one
direction); this module imports only leaf modules (no back-edge into chain.py).
"""
# ruff: noqa: E501

from typing import Any, Dict, Optional, Tuple

from graphistry.Plottable import Plottable
from .ast import ASTNode, ASTEdge, Direction
from .typing import DataFrameT


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
    for k, v in n2f.items():  # destination-node filter
        dstn = dstn.filter(pl.col(k) == v)
    # drop dangling edges + dedup destination nodes (mirror the pandas tail)
    keep_ids = dstn.get_column(node).drop_nulls()
    edges = edges.filter(pl.col(to_col).is_in(keep_ids.implode()))
    dstn = dstn.filter(pl.col(node).is_in(edges.get_column(to_col).implode())).unique(subset=[node], maintain_order=True)
    return dstn, edges
