"""Bounded forward hop over caller-supplied immutable local Parquet scans."""

from __future__ import annotations

from concurrent.futures import CancelledError
from time import monotonic
from typing import TYPE_CHECKING, Optional

from packaging.version import Version

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.parquet import ParquetHopLimitError, ParquetHopLimits

from .membership import id_set, is_in_ids
from .nan_clean import _pl_nan_to_null

if TYPE_CHECKING:
    import polars as pl


def _check_running(limits: ParquetHopLimits, deadline: float) -> None:
    if limits.cancel_event is not None and limits.cancel_event.is_set():
        raise CancelledError("Parquet hop cancelled")
    if monotonic() >= deadline:
        raise ParquetHopLimitError("Parquet hop exceeded timeout_seconds")


def _bounded_scan(
    plan: "pl.LazyFrame", rows: int, limit_name: str, limits: ParquetHopLimits, deadline: float,
) -> "pl.DataFrame":
    _check_running(limits, deadline)
    result = plan.limit(rows + 1).collect(engine="streaming")
    _check_running(limits, deadline)
    if result.height > rows:
        raise ParquetHopLimitError(f"Parquet hop exceeded {limit_name}")
    return result


def _scan_membership(key: str, ids: "pl.Series") -> "pl.Expr":
    import polars as pl

    # Give Parquet statistics simple bounds to prune even when the reader
    # cannot derive them from the exact membership expression.
    column = pl.col(key)
    low, high = ids.min(), ids.max()
    return is_in_ids(column, ids) & column.is_between(
        pl.lit(low, dtype=ids.dtype), pl.lit(high, dtype=ids.dtype),
    )


def _lookup_nodes(
    nodes: "pl.LazyFrame", key: str, ids: "pl.Series", limits: ParquetHopLimits, deadline: float,
) -> "pl.Series":
    lookup = nodes.filter(_scan_membership(key, ids)).select(key)
    found = _bounded_scan(lookup, min(len(ids), limits.max_nodes), "max_nodes in node lookup", limits, deadline)[key]
    if found.n_unique() != len(found):
        raise ValueError("Parquet hop requires unique node identities")
    return found


def _validate_keys(
    nodes: "pl.Schema", edges: "pl.Schema", seeds: "pl.Schema", node: str, src: str, dst: str, edge: str,
) -> None:
    import polars as pl

    dtype = nodes.get(node)
    if dtype not in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64) or (
        edges.get(src) != dtype or edges.get(dst) != dtype or seeds.get(node) != dtype
    ):
        raise ValueError("Parquet hop requires matching Int32, Int64, UInt32 or UInt64 node, endpoint and seed columns")
    if edges.get(edge) not in (pl.Int32, pl.Int64, pl.UInt32, pl.UInt64):
        raise ValueError("Parquet hop requires an Int32, Int64, UInt32 or UInt64 edge identity column")


def _visit(
    nodes: "pl.LazyFrame", edges: "pl.LazyFrame", keys: tuple[str, str, str, str], edge_dtype: "pl.DataType",
    seeds: "pl.Series", hops: int, limits: ParquetHopLimits, deadline: float,
) -> tuple["pl.Series", "pl.Series"]:
    import polars as pl

    node, src, dst, edge = keys
    visited = seeds.clear()
    frontier = _lookup_nodes(nodes, node, seeds, limits, deadline) if len(seeds) else seeds
    edge_ids = pl.Series(edge, [], dtype=edge_dtype)
    for step in range(hops):
        if frontier.is_empty():
            break
        candidates = _bounded_scan(
            edges.filter(_scan_membership(src, frontier) & pl.col(dst).is_not_null()).select(src, dst, edge),
            limits.max_edges, "max_edges in frontier scan", limits, deadline,
        )
        if candidates.is_empty():
            break
        destinations = _lookup_nodes(nodes, node, candidates[dst].unique(), limits, deadline)
        matches = candidates.filter(is_in_ids(pl.col(dst), destinations))
        if len(edge_ids) + matches.height > limits.max_edges:
            raise ParquetHopLimitError("Parquet hop exceeded max_edges")
        edge_ids = pl.concat([edge_ids, matches[edge]])
        if step == 0:
            # Isolated and missing seeds are absent from the default hop output.
            visited = matches[src].unique()
        frontier = destinations.filter(~destinations.is_in(id_set(visited)))
        if len(visited) + len(frontier) > limits.max_nodes:
            raise ParquetHopLimitError("Parquet hop exceeded max_nodes")
        visited = pl.concat([visited, frontier])
    if edge_ids.null_count() or edge_ids.n_unique() != len(edge_ids):
        raise ValueError("Parquet hop requires unique non-null edge identities")
    return visited, edge_ids


def _rehydrate(
    source: "pl.LazyFrame", schema: "pl.Schema", key: str, ids: "pl.Series",
    limits: ParquetHopLimits, deadline: float,
) -> "pl.DataFrame":
    import polars as pl

    if ids.is_empty():
        return pl.DataFrame(schema=schema)
    result = _bounded_scan(source.filter(_scan_membership(key, ids)), len(ids), "rehydration rows", limits, deadline)
    if result.height != len(ids) or result[key].n_unique() != len(ids):
        raise ValueError(f"Parquet hop requires stable, unique identities in {key!r}")
    cleaned = _pl_nan_to_null(result)
    assert isinstance(cleaned, pl.DataFrame)
    return cleaned


def hop_parquet(
    g: Plottable, seeds: object, hops: Optional[int], limits: ParquetHopLimits,
) -> Plottable:
    import polars as pl

    deadline = monotonic() + limits.timeout_seconds
    _check_running(limits, deadline)
    if Version(pl.__version__) < Version("1.29.0"):
        raise NotImplementedError("Parquet hop requires Polars 1.29 or newer")
    if type(hops) is not int or hops < 1:
        raise NotImplementedError("Parquet hop requires a positive finite hops value")
    if hops > limits.max_hops:
        raise ParquetHopLimitError("Parquet hop exceeded max_hops")
    if not isinstance(seeds, pl.DataFrame):
        raise NotImplementedError("Parquet hop requires explicit seeds as a Polars DataFrame")
    if seeds.height > limits.max_nodes:
        raise ParquetHopLimitError("Parquet hop seeds exceeded max_nodes")

    nodes, edges = g._nodes, g._edges
    node, src, dst, edge = g._node, g._source, g._destination, g._edge
    if not isinstance(nodes, pl.LazyFrame) or not isinstance(edges, pl.LazyFrame):
        raise NotImplementedError("Parquet hop requires lazy node and edge scans")
    if node is None or src is None or dst is None or edge is None or len({src, dst, edge}) != 3:
        raise ValueError("Parquet hop requires node, source, destination and distinct edge bindings")

    node_schema, edge_schema = nodes.collect_schema(), edges.collect_schema()
    _validate_keys(node_schema, edge_schema, seeds.schema, node, src, dst, edge)
    node_ids, edge_ids = _visit(
        nodes, edges, (node, src, dst, edge), edge_schema[edge], seeds[node].drop_nulls().unique(), hops, limits, deadline,
    )
    out_edges = _rehydrate(edges, edge_schema, edge, edge_ids, limits, deadline)
    edge_bytes = out_edges.estimated_size()
    if edge_bytes > limits.max_result_bytes:
        raise ParquetHopLimitError("Parquet hop exceeded max_result_bytes in edge rehydration")
    out_nodes = _rehydrate(nodes, node_schema, node, node_ids, limits, deadline)
    if edge_bytes + out_nodes.estimated_size() > limits.max_result_bytes:
        raise ParquetHopLimitError("Parquet hop exceeded max_result_bytes in node rehydration")
    _check_running(limits, deadline)
    return g.nodes(out_nodes, node).edges(out_edges, src, dst, edge=edge)
