"""Native polars graph-degree helpers (get_degrees / get_indegrees / get_outdegrees).

Extracted from chain.py (node-degree computation, not traversal orchestration). Parity with
ComputeMixin.get_degrees & friends; pure polars, no pandas bridge (plan.md NO-CHEATING).
Reached via .get_degrees() etc. dispatch and the GFQL CALL executor.
"""
from typing import Optional

from graphistry.Engine import Engine, EngineAbstract, EngineAbstractType, resolve_engine
from graphistry.Plottable import Plottable
from .dtypes import is_lazy, colnames, col_dtype
from .hop_eager import ensure_nodes_polars


def _index_engine(engine: Optional[EngineAbstractType] = None) -> Engine:
    if engine is not None and engine not in (EngineAbstract.AUTO, "auto"):
        return resolve_engine(engine)
    from graphistry.compute.gfql.lazy import active_target, ExecutionTarget
    return Engine.POLARS_GPU if active_target() == ExecutionTarget.GPU else Engine.POLARS


def _endpoint_counts(edges, key_col: str, node_dt, node_col: str, alias: str):
    """group_by-count on ONE edge endpoint, key cast to node-id dtype so left-join keys match
    when node/endpoint dtypes diverge (polars won't auto-cast int<->float keys; pandas merges fine)."""
    import polars as pl
    return edges.group_by(pl.col(key_col).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(alias)
    )


def get_degrees_polars(
    g: Plottable,
    col: str = "degree",
    degree_in: str = "degree_in",
    degree_out: str = "degree_out",
    engine: Optional[EngineAbstractType] = None,
) -> Plottable:
    """Native ``get_degrees`` — parity with ComputeMixin.get_degrees.

    group_by-count per endpoint, left-joined onto the node table (materialized from edges when
    absent). Oracle contract: isolated and src-only/dst-only nodes get 0; self-loops double-count
    (one in + one out); all three columns Int32. Empty edges need no special case — left-join +
    fill_null(0) yields all-zero, matching the pandas empty-edges branch. The ``engine``
    sub-param selects the resident index engine when an index fast path is available;
    the scan path is already committed to polars execution.
    """
    import polars as pl

    g = ensure_nodes_polars(g)
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._nodes is not None and g._edges is not None
    nodes, edges = g._nodes, g._edges

    # GFQL #1658 index fast path (#5 degree-cache / #3 membership): degrees from a
    # resident CSR index (O(N) gather) instead of the group_by below. Eager-only
    # (LazyFrame get_column would force a collect); policy 'off' skips. Missing
    # or stale indexes return None and fall through; real errors should surface.
    from graphistry.compute.gfql.index import get_index_policy
    if not is_lazy(nodes) and not is_lazy(edges) and get_index_policy(g) != "off":
        from graphistry.compute.gfql.index import get_registry
        from graphistry.compute.gfql.index.degrees import degrees_from_index
        _reg = get_registry(g)
        if not _reg.is_empty():
            _d = degrees_from_index(_reg, nodes, node_col, edges, (src, dst), _index_engine(engine))
            if _d is not None:
                _in, _out = _d
                drop0 = [c for c in colnames(nodes) if c in (degree_in, degree_out, col)]
                base0 = nodes.drop(drop0) if drop0 else nodes
                out0 = base0.with_columns(
                    pl.Series(degree_in, _in).cast(pl.Int32),
                    pl.Series(degree_out, _out).cast(pl.Int32),
                ).with_columns((pl.col(degree_in) + pl.col(degree_out)).cast(pl.Int32).alias(col))
                return g.nodes(out0, node_col)

    node_dt = col_dtype(nodes, node_col)
    in_counts = _endpoint_counts(edges, dst, node_dt, node_col, degree_in)
    out_counts = _endpoint_counts(edges, src, node_dt, node_col, degree_out)

    # Drop pre-existing degree columns (mirrors the pandas keep-subset) so a re-run
    # overwrites instead of producing *_right join collisions.
    drop_cols = [c for c in colnames(nodes) if c in (degree_in, degree_out, col)]
    base = nodes.drop(drop_cols) if drop_cols else nodes

    out = (
        base.join(in_counts, on=node_col, how="left")
        .join(out_counts, on=node_col, how="left")
        .with_columns(
            pl.col(degree_in).fill_null(0).cast(pl.Int32),
            pl.col(degree_out).fill_null(0).cast(pl.Int32),
        )
        .with_columns(
            (pl.col(degree_in) + pl.col(degree_out)).cast(pl.Int32).alias(col)
        )
    )
    return g.nodes(out, node_col)


def _single_direction_degree_polars(g: Plottable, key_col: str, col: str) -> Plottable:
    """Shared body for get_indegrees / get_outdegrees — parity with
    ComputeMixin._single_direction_degree. group_by-count on ONE endpoint (destination for in,
    source for out) left-joined onto the node table; isolated / opposite-endpoint-only nodes get
    0; a self-loop counts ONCE per direction (not double-counted like the get_degrees total);
    column is Int32; count key cast to node-id dtype (polars won't auto-cast int<->float join
    keys). Empty-edges mirrors the oracle EXACTLY and differs from get_degrees: no edges AND the
    target column already exists -> node table returned UNCHANGED (pandas keeps pre-existing
    values); otherwise materialize all-zero Int32.
    """
    import polars as pl

    g = ensure_nodes_polars(g)
    node_col = g._node
    assert node_col is not None
    assert g._nodes is not None and g._edges is not None
    nodes, edges = g._nodes, g._edges

    if is_lazy(edges):
        edges_empty = edges.select(pl.len()).collect().item() == 0
    else:
        edges_empty = edges.height == 0
    if edges_empty:
        if col in colnames(nodes):
            return g.nodes(nodes, node_col)
        return g.nodes(nodes.with_columns(pl.lit(0).cast(pl.Int32).alias(col)), node_col)

    counts = _endpoint_counts(edges, key_col, col_dtype(nodes, node_col), node_col, col)
    # Drop a pre-existing degree column (mirrors the pandas keep-subset) so a re-run
    # overwrites instead of a *_right join collision.
    base = nodes.drop(col) if col in colnames(nodes) else nodes
    out = base.join(counts, on=node_col, how="left").with_columns(
        pl.col(col).fill_null(0).cast(pl.Int32)
    )
    return g.nodes(out, node_col)


def get_indegrees_polars(g: Plottable, col: str = "degree_in", engine: Optional[EngineAbstractType] = None) -> Plottable:
    """Native ``get_indegrees`` (parity with ComputeMixin.get_indegrees): in-degree = count of
    edges whose DESTINATION endpoint is the node."""
    assert g._destination is not None, "Missing destination binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._destination, col)


def get_outdegrees_polars(g: Plottable, col: str = "degree_out", engine: Optional[EngineAbstractType] = None) -> Plottable:
    """Native ``get_outdegrees`` (parity with ComputeMixin.get_outdegrees): out-degree = count of
    edges whose SOURCE endpoint is the node."""
    assert g._source is not None, "Missing source binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._source, col)
