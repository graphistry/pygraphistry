"""Native polars graph-degree helpers (get_degrees / get_indegrees / get_outdegrees).

Extracted from ``chain.py`` — these are node-degree computations, not chain/traversal
orchestration. Parity with ``ComputeMixin.get_degrees`` & friends; pure polars, no
pandas bridge (see plan.md NO-CHEATING). Reached via ``.get_degrees()`` etc. dispatch
and the GFQL ``CALL`` executor.
"""
from typing import Optional

from graphistry.Plottable import Plottable
from .dtypes import is_lazy, colnames
from .hop_eager import ensure_nodes_polars


def get_degrees_polars(
    g: Plottable,
    col: str = "degree",
    degree_in: str = "degree_in",
    degree_out: str = "degree_out",
    engine: Optional[str] = None,
) -> Plottable:
    """Native polars ``get_degrees`` — parity with ComputeMixin.get_degrees.

    Per-node in/out degree via a group_by-count on the edge endpoint columns,
    left-joined onto the node table (materialized from edges when absent). Pure
    polars — NO pandas bridge (see plan.md NO-CHEATING). Parity contract vs the
    pandas oracle: isolated nodes and src-only / dst-only nodes get 0; self-loops
    are double-counted (one in + one out); all three columns are Int32. Empty
    edges need no special case — the left-join + fill_null(0) yields all-zero
    degrees, matching the pandas empty-edges branch.

    The ``engine`` safelist sub-param is accepted and ignored: execution is already
    committed to polars and the result stays parity-equal to the pandas oracle.
    """
    import polars as pl

    g = ensure_nodes_polars(g)
    node_col, src, dst = g._node, g._source, g._destination
    assert node_col is not None and src is not None and dst is not None
    assert g._nodes is not None and g._edges is not None
    nodes, edges = g._nodes, g._edges

    # Align the count keys to the node-id dtype so the left-join keys match even when
    # a user node table's id dtype diverges from the edge endpoint dtype (polars will
    # not auto-cast int<->float join keys, where pandas merges fine).
    node_dt = (nodes.collect_schema() if is_lazy(nodes) else nodes.schema)[node_col]

    in_counts = edges.group_by(pl.col(dst).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(degree_in)
    )
    out_counts = edges.group_by(pl.col(src).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(degree_out)
    )

    # Drop any pre-existing degree columns, mirroring the pandas keep-subset, so a
    # re-run overwrites rather than producing ``*_right`` join collisions.
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
    """Native shared body for get_indegrees / get_outdegrees — parity with
    ComputeMixin._single_direction_degree. A group_by-count on ONE edge endpoint
    (``destination`` for in-degree, ``source`` for out-degree), left-joined onto the
    node table. Isolated / opposite-endpoint-only nodes get 0; a self-loop counts once
    for the relevant direction (single-direction is NOT double-counted, unlike the
    get_degrees total); the degree column is Int32. The count key is cast to the node-id
    dtype so the join keys match even when the edge endpoint dtype diverges (polars won't
    auto-cast int<->float join keys where pandas merges fine).

    The empty-edges case mirrors the pandas oracle EXACTLY, which differs from
    get_degrees: when there are no edges AND the target column already exists, the node
    table is returned UNCHANGED (pandas keeps the pre-existing values); otherwise the
    column is materialized as all-zero Int32. Pure polars — NO pandas bridge (see
    NO-CHEATING).
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

    node_dt = (nodes.collect_schema() if is_lazy(nodes) else nodes.schema)[node_col]
    counts = edges.group_by(pl.col(key_col).cast(node_dt).alias(node_col)).agg(
        pl.len().alias(col)
    )
    # Drop a pre-existing degree column so a re-run overwrites rather than producing a
    # ``*_right`` join collision (mirrors the pandas keep-subset).
    base = nodes.drop(col) if col in colnames(nodes) else nodes
    out = base.join(counts, on=node_col, how="left").with_columns(
        pl.col(col).fill_null(0).cast(pl.Int32)
    )
    return g.nodes(out, node_col)


def get_indegrees_polars(g: Plottable, col: str = "degree_in") -> Plottable:
    """Native polars ``get_indegrees`` — parity with ComputeMixin.get_indegrees.

    In-degree = count of edges whose DESTINATION endpoint is the node. Pure polars —
    NO pandas bridge (see NO-CHEATING).
    """
    assert g._destination is not None, "Missing destination binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._destination, col)


def get_outdegrees_polars(g: Plottable, col: str = "degree_out") -> Plottable:
    """Native polars ``get_outdegrees`` — parity with ComputeMixin.get_outdegrees.

    Out-degree = count of edges whose SOURCE endpoint is the node. Pure polars —
    NO pandas bridge (see NO-CHEATING).
    """
    assert g._source is not None, "Missing source binding; set via .bind() or .edges()"
    return _single_direction_degree_polars(g, g._source, col)
