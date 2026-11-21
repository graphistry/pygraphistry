from __future__ import annotations

from concurrent.futures import CancelledError
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.gfql.index import with_index_policy
from graphistry.compute.gfql.parquet import ParquetHopLimitError, ParquetHopLimits
from graphistry.compute.gfql.lazy.engine.polars import hop_parquet

if TYPE_CHECKING:
    import polars as pl
else:
    pl = pytest.importorskip("polars", minversion="1.29")


@pytest.fixture
def frames() -> tuple["pl.DataFrame", "pl.DataFrame"]:
    nodes = pl.DataFrame({
        "id": [0, 1, 2, 3, 4, 5, 6],
        "property": ["zero", "one", None, "", "four", None, "isolated"],
        "score": [1., float("nan"), None, -1., 1., 2., None],
    })
    edges = pl.DataFrame({
        "eid": list(range(16)),
        "src": [0, 0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 99, None, 5, 2],
        "dst": [1, 1, 0, 2, 3, 0, 3, 4, 1, 5, 3, None, 3, 0, 99, 5],
        "property": ["parallel A", "parallel B", "self-loop", None] * 4,
    })
    return nodes, edges


def _graph(nodes: "pl.DataFrame | pl.LazyFrame", edges: "pl.DataFrame | pl.LazyFrame") -> Plottable:
    return graphistry.nodes(nodes, "id").edges(edges, "src", "dst", "eid")


def _scan_graph(tmp_path: Path, nodes: "pl.DataFrame", edges: "pl.DataFrame", hive: bool = False) -> Plottable:
    node_path = tmp_path / "nodes.parquet"
    nodes.write_parquet(node_path, row_group_size=2)
    edge_path = tmp_path / "edges.parquet"
    if hive:
        edge_path = tmp_path / "edges"
        for offset in range(0, edges.height, 4):
            part = edge_path / f"bucket={offset // 4}"
            part.mkdir(parents=True)
            edges.slice(offset, 4).write_parquet(part / "part.parquet", row_group_size=2)
        edge_scan = pl.scan_parquet(str(edge_path / "**" / "*.parquet"), hive_partitioning=True).drop("bucket")
    else:
        edges.write_parquet(edge_path, row_group_size=2)
        edge_scan = pl.scan_parquet(edge_path)
    return _graph(pl.scan_parquet(node_path), edge_scan)


@pytest.mark.parametrize("seeds", [[0], [0, 4, 4, 99], [6], [99], [None], [], [2, 6, 99, None]])
@pytest.mark.parametrize("hops", [1, 2, 4, 8])
def test_parquet_hop_matches_public_polars(
    tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], seeds: list[int | None], hops: int,
) -> None:
    from polars.testing import assert_frame_equal

    nodes, edges = frames
    lazy_graph = _scan_graph(tmp_path, nodes, edges, hive=True)
    starts = pl.DataFrame({"id": seeds}, schema={"id": pl.Int64})
    expected = _graph(nodes, edges).hop(starts, hops=hops, engine="polars")
    result = lazy_graph.hop(starts, hops=hops, parquet_limits=ParquetHopLimits())
    assert_frame_equal(result._nodes.sort("id"), expected._nodes.sort("id"))
    assert_frame_equal(result._edges.sort("eid"), expected._edges.sort("eid"))
    assert (result._node, result._edge, result._source, result._destination) == ("id", "eid", "src", "dst")
    assert isinstance(lazy_graph._nodes, pl.LazyFrame) and isinstance(lazy_graph._edges, pl.LazyFrame)


@pytest.mark.parametrize("dtype,offset", [("Int32", -2), ("UInt32", 2**31), ("UInt64", 2**63)])
def test_identity_types(tmp_path: Path, dtype: str, offset: int) -> None:
    from polars.testing import assert_frame_equal

    key_type = getattr(pl, dtype)
    nodes = pl.DataFrame({"id": pl.Series([offset, offset + 1], dtype=key_type)})
    edges = pl.DataFrame({
        "src": pl.Series([offset], dtype=key_type),
        "dst": pl.Series([offset + 1], dtype=key_type),
        "eid": pl.Series([2**63], dtype=pl.UInt64),
    })
    result = _scan_graph(tmp_path, nodes, edges).hop(nodes.head(1), parquet_limits=ParquetHopLimits())
    assert_frame_equal(result._nodes.sort("id"), nodes)
    assert_frame_equal(result._edges, edges)


def test_parquet_plans_filter_and_bound_every_collection(
    tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], monkeypatch: pytest.MonkeyPatch,
) -> None:
    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)
    original_collect = pl.LazyFrame.collect
    plans: list[str] = []

    def collect(frame: "pl.LazyFrame", *args: object, **kwargs: object) -> "pl.DataFrame":
        # Eager Polars methods also use LazyFrame.collect internally.
        plan = frame.explain()
        if "Parquet SCAN" in plan:
            plans.append(plan)
            assert "SELECTION:" in plan and "SLICE" in plan
            assert "is_between" in plan and "is_in" in plan
            assert kwargs["engine"] == "streaming"
        return original_collect(frame, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    result = g.hop(nodes.head(1), hops=2, parquet_limits=ParquetHopLimits())
    assert result._edges.height == 5
    assert sum("PROJECT 3/4 COLUMNS" in plan for plan in plans) == 2
    assert sum("PROJECT 1/3 COLUMNS" in plan for plan in plans) == 3
    assert sum("PROJECT */" in plan for plan in plans) == 2


@pytest.mark.parametrize("case", [
    {"direction": "reverse"}, {"direction": "undirected"}, {"min_hops": 1},
    {"max_hops": 2}, {"output_min_hops": 0}, {"output_max_hops": 1},
    {"label_node_hops": "hop"}, {"label_edge_hops": "hop"}, {"label_seeds": True},
    {"to_fixed_point": True}, {"return_as_wave_front": True}, {"include_zero_hop_seed": True},
    {"edge_match": {}}, {"source_node_match": {}}, {"destination_node_match": {}},
    {"source_node_query": "id == 0"}, {"destination_node_query": "id == 1"},
    {"edge_query": "eid == 0"}, {"target_wave_front": "placeholder"},
    {"engine": "pandas"}, {"engine": "polars-gpu"}, {"hops": None}, {"hops": 0},
])
def test_unsupported_opt_in_declines_before_collection(
    tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], monkeypatch: pytest.MonkeyPatch,
    case: dict[str, object],
) -> None:
    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)

    def unexpected(*args: object, **kwargs: object) -> None:
        pytest.fail("unsupported opt-in query should decline before collecting")

    monkeypatch.setattr(pl.LazyFrame, "collect", unexpected)
    with pytest.raises(NotImplementedError, match="Parquet hop"):
        g.hop(nodes.head(1), parquet_limits=ParquetHopLimits(), **case)


def test_missing_seeds_and_existing_reverse_route(tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"]) -> None:
    from polars.testing import assert_frame_equal

    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)
    with pytest.raises(NotImplementedError, match="explicit seeds"):
        g.hop(parquet_limits=ParquetHopLimits())
    expected = _graph(nodes, edges).hop(nodes.head(1), direction="reverse", engine="polars")
    actual = g.hop(nodes.head(1), direction="reverse", engine="polars")
    assert_frame_equal(actual._nodes.sort("id"), expected._nodes.sort("id"))
    assert_frame_equal(actual._edges.sort("eid"), expected._edges.sort("eid"))


@pytest.mark.parametrize("limit, value, hops", [
    ("max_nodes", 1, 1), ("max_edges", 2, 1), ("max_edges", 4, 2),
    ("max_hops", 1, 2), ("max_result_bytes", 1, 1),
])
def test_decline_result_instead_of_truncating(
    tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], limit: str, value: int, hops: int,
) -> None:
    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)
    with pytest.raises(ParquetHopLimitError, match=limit):
        g.hop(nodes.head(1), hops=hops, parquet_limits=ParquetHopLimits(**{limit: value}))
    assert isinstance(g._edges, pl.LazyFrame) and isinstance(g._nodes, pl.LazyFrame)


@pytest.mark.parametrize("kind", ["duplicate_node", "duplicate_edge", "duplicate_matched_edge", "null_edge", "seed_dtype", "edge_binding"])
def test_identity_contract(tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], kind: str) -> None:
    nodes, edges = frames
    starts = nodes.head(1)
    if kind == "duplicate_node":
        nodes = pl.concat([nodes, nodes.slice(1, 1)])
    if kind == "duplicate_edge":
        edges = edges.with_columns(pl.when(pl.col("eid") == 14).then(0).otherwise(pl.col("eid")).alias("eid"))
    if kind == "duplicate_matched_edge":
        edges = edges.with_columns(pl.when(pl.col("eid") == 1).then(0).otherwise(pl.col("eid")).alias("eid"))
    if kind == "null_edge":
        edges = edges.with_columns(pl.when(pl.col("eid") == 0).then(None).otherwise(pl.col("eid")).alias("eid"))
    if kind == "seed_dtype":
        starts = starts.with_columns(pl.col("id").cast(pl.Int32))
    g = _scan_graph(tmp_path, nodes, edges)
    if kind == "edge_binding":
        g = _graph(g._nodes, g._edges.drop("eid"))
    with pytest.raises((ValueError, ParquetHopLimitError), match="Parquet hop"):
        g.hop(starts, parquet_limits=ParquetHopLimits())


@pytest.mark.parametrize("limit_stage", ["before_scan", "during_rehydration", "timeout"])
def test_cancellation_and_timeout(
    tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"], monkeypatch: pytest.MonkeyPatch,
    limit_stage: str,
) -> None:
    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)
    event = Event()
    original_collect = pl.LazyFrame.collect

    def collect(frame: "pl.LazyFrame", *args: object, **kwargs: object) -> "pl.DataFrame":
        result = original_collect(frame, *args, **kwargs)
        if "Parquet SCAN" in frame.explain() and "PROJECT */" in frame.explain():
            event.set()
        return result

    if limit_stage == "before_scan":
        event.set()
    monkeypatch.setattr(pl.LazyFrame, "collect", collect)
    if limit_stage == "timeout":
        times = iter([0.0, 0.0, 1.0])
        monkeypatch.setattr(hop_parquet, "monotonic", lambda: next(times))
        with pytest.raises(ParquetHopLimitError, match="timeout_seconds"):
            g.hop(nodes.head(1), parquet_limits=ParquetHopLimits(cancel_event=event, timeout_seconds=0.5))
    else:
        with pytest.raises(CancelledError):
            g.hop(nodes.head(1), parquet_limits=ParquetHopLimits(cancel_event=event))
    assert isinstance(g._edges, pl.LazyFrame) and isinstance(g._nodes, pl.LazyFrame)


def test_scan_failure_surfaces_without_a_partial_result(tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"]) -> None:
    nodes, edges = frames
    g = _scan_graph(tmp_path, nodes, edges)
    (tmp_path / "edges.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        g.hop(nodes.head(1), parquet_limits=ParquetHopLimits())
    assert isinstance(g._edges, pl.LazyFrame)


@pytest.mark.parametrize("options", [
    {"max_nodes": 0}, {"max_edges": -1}, {"max_hops": True}, {"max_result_bytes": 0},
    {"timeout_seconds": 0.0}, {"timeout_seconds": float("inf")},
])
def test_invalid_limits(options: dict[str, int | float]) -> None:
    with pytest.raises(ValueError, match="positive"):
        ParquetHopLimits(**options)


def test_index_policies_decline(tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"]) -> None:
    nodes, edges = frames
    graph = _scan_graph(tmp_path, nodes, edges)
    for policy in ("auto", "force"):
        with pytest.raises(NotImplementedError, match="without resident indexes"):
            with_index_policy(graph, policy).hop(nodes.head(1), parquet_limits=ParquetHopLimits())

    indexed = _graph(nodes, edges).gfql_index_edges(direction="forward", engine="polars")
    with pytest.raises(NotImplementedError, match="without resident indexes"):
        indexed.hop(nodes.head(1), parquet_limits=ParquetHopLimits())
    assert indexed.hop(nodes.head(1), engine="polars")._edges.height == 3


def test_exact_limits_succeed(tmp_path: Path, frames: tuple["pl.DataFrame", "pl.DataFrame"]) -> None:
    nodes, edges = frames
    graph = _scan_graph(tmp_path, nodes, edges)
    result = graph.hop(nodes.head(1), parquet_limits=ParquetHopLimits(max_nodes=2, max_edges=3, max_hops=1))
    assert result._nodes.height == 2 and result._edges.height == 3
