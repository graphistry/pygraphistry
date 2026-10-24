"""#1949: temporal membership filters obey the same typed coercion as equality."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
from graphistry.compute.filter_by_dict import filter_by_dict

try:
    import polars as pl
except ImportError:  # pragma: no cover - CPU lanes also run without optional engines
    pl = None

try:
    import cudf
except ImportError:  # pragma: no cover - exercised on DGX
    cudf = None


def _nodes_pd() -> pd.DataFrame:
    return pd.DataFrame({
        "id": ["p", "q", "r"],
        "ts_aw": pd.to_datetime([
            "2020-06-15T08:30:00Z", "2021-06-15T08:30:00Z", "2022-01-01T00:00:00Z",
        ], utc=True),
        "ts_naive": pd.to_datetime(["2020-06-15T08:30:00", "2021-06-15T08:30:00", "2022-01-01T00:00:00"]),
    })


def _edges_pd() -> pd.DataFrame:
    return pd.DataFrame({"s": ["p"], "d": ["q"]})


def _ids(nodes: pd.DataFrame) -> list[str]:
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    column = "id" if "id" in nodes.columns else "n.id"
    return sorted(nodes[column].dropna().tolist())


def _graph(engine: str):
    nodes, edges = _nodes_pd(), _edges_pd()
    if engine == "polars":
        assert pl is not None
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        assert cudf is not None
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _graph_from_nodes(nodes: pd.DataFrame, engine: str):
    edges = _edges_pd()
    if engine == "polars":
        assert pl is not None
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(
            pl.from_pandas(edges), "s", "d"
        )
    if engine == "cudf":
        assert cudf is not None
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(edges), "s", "d"
        )
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _nodes_pd_unit(unit: str) -> pd.DataFrame:
    exact_ns = {"s": 1_000_000_000, "ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
    values_ns = [-1, exact_ns, exact_ns + 1, None]
    stamps = pd.Series(values_ns, dtype="Int64").astype(
        "datetime64[ns, UTC]"
    ).astype(pd.DatetimeTZDtype(unit, "UTC"))
    return pd.DataFrame({"id": ["neg", "exact", "pos", "null"], "ts": stamps})


@pytest.mark.parametrize(
    ("members", "expected"),
    [
        (["2021-06-15T08:30:00"], ["q"]),
        (["2021-06-15T09:30:00+01:00"], ["q"]),
        (["2020-06-15T08:30:00Z", "2022-01-01T00:00:00Z"], ["p", "r"]),
        (["2021-06-16T00:00:00Z"], []),
    ],
)
def test_pandas_aware_datetime_membership_matches_typed_instant(members, expected):
    nodes = _nodes_pd()
    assert _ids(filter_by_dict(nodes, {"ts_aw": members}, engine="pandas")) == expected
    query = "MATCH (n) WHERE n.ts_aw IN $members RETURN n.id"
    assert _ids(_graph("pandas").gfql(query, engine="pandas", params={"members": members})._nodes) == expected


def test_pandas_aware_datetime_isin_predicate_matches_typed_instant():
    from graphistry.compute.predicates.is_in import is_in

    assert _ids(filter_by_dict(
        _nodes_pd(), {"ts_aw": is_in(["2021-06-15T08:30:00"])}, engine="pandas"
    )) == ["q"]


def test_pandas_cypher_literal_membership_matches_typed_instant():
    query = "MATCH (n) WHERE n.ts_aw IN ['2021-06-15T08:30:00'] RETURN n.id"
    assert _ids(_graph("pandas").gfql(query, engine="pandas")._nodes) == ["q"]


def test_pandas_naive_datetime_membership_matches_local_wall_time():
    assert _ids(filter_by_dict(
        _nodes_pd(), {"ts_naive": ["2021-06-15T08:30:00"]}, engine="pandas"
    )) == ["q"]


@pytest.mark.skipif(pl is None, reason="polars not installed")
@pytest.mark.parametrize("unit", ["ms", "us", "ns"])
def test_polars_membership_preserves_column_precision(unit):
    exact_ns = {"ms": 1_000_000, "us": 1_000, "ns": 1}[unit]
    exact = pd.Timestamp(exact_ns, unit="ns").isoformat()
    adjacent = pd.Timestamp(exact_ns + 1, unit="ns").isoformat()
    assert _ids(filter_by_dict(
        pl.from_pandas(_nodes_pd_unit(unit)), {"ts": [exact, adjacent]}, engine="polars"
    )) == ["exact", "pos"]
    assert _ids(filter_by_dict(
        pl.from_pandas(_nodes_pd_unit(unit)), {"ts": [adjacent]}, engine="polars"
    )) == (["pos"] if unit == "ns" else [])


def test_polars_membership_signed_and_adjacent_bounds():
    engine = "polars"
    if pl is None:
        pytest.skip("polars not installed")
    nodes = _nodes_pd_unit("us")
    negative = pd.Timestamp(-1_000, unit="ns").isoformat()
    negative_adjacent = pd.Timestamp(-999, unit="ns").isoformat()
    exact = pd.Timestamp(1_000, unit="ns").isoformat()
    positive_adjacent = pd.Timestamp(1_001, unit="ns").isoformat()
    nodes = pl.from_pandas(nodes)
    assert _ids(filter_by_dict(
        nodes, {"ts": [negative, negative_adjacent]}, engine=engine
    )) == ["neg"]
    assert _ids(filter_by_dict(
        nodes, {"ts": [negative_adjacent]}, engine=engine
    )) == []
    assert _ids(filter_by_dict(
        nodes, {"ts": [exact, positive_adjacent]}, engine=engine
    )) == ["exact", "pos"]
    assert _ids(filter_by_dict(
        nodes, {"ts": [positive_adjacent]}, engine=engine
    )) == []


@pytest.mark.skipif(pl is None, reason="polars not installed")
def test_polars_isin_preserves_adjacent_precision():
    from graphistry.compute.predicates.is_in import is_in

    nodes = _nodes_pd_unit("ns")
    exact = pd.Timestamp(1, unit="ns").isoformat()
    adjacent = pd.Timestamp(2, unit="ns").isoformat()
    assert _ids(filter_by_dict(
        nodes, {"ts": is_in([exact])}, engine="polars"
    )) == ["exact"]
    assert _ids(filter_by_dict(
        nodes, {"ts": is_in([adjacent])}, engine="polars"
    )) == ["pos"]


def test_polars_typed_timestamp_membership_and_isin_preserve_precision():
    from graphistry.compute.predicates.is_in import is_in

    nodes = _nodes_pd_unit("ns")
    exact = pd.Timestamp(1, unit="ns", tz="UTC")
    adjacent = pd.Timestamp(2, unit="ns", tz="UTC")
    outside = pd.Timestamp(3, unit="ns", tz="UTC")
    assert _ids(filter_by_dict(nodes, {"ts": [exact, outside]}, engine="polars")) == ["exact"]
    assert _ids(filter_by_dict(nodes, {"ts": [outside]}, engine="polars")) == []
    assert _ids(filter_by_dict(nodes, {"ts": [adjacent]}, engine="polars")) == ["pos"]
    assert _ids(filter_by_dict(nodes, {"ts": is_in([exact])}, engine="polars")) == ["exact"]
    assert _ids(filter_by_dict(nodes, {"ts": is_in([outside])}, engine="polars")) == []


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_typed_timestamp_timezone_mismatch_is_not_member(engine):
    if engine == "polars":
        if pl is None:
            pytest.skip("polars not installed")
        nodes = pl.from_pandas(_nodes_pd())
    else:
        nodes = _nodes_pd()
    assert _ids(filter_by_dict(nodes, {"ts_aw": [pd.Timestamp("2021-06-15")]}, engine=engine)) == []


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_typed_nat_is_not_member(engine):
    if engine == "polars":
        if pl is None:
            pytest.skip("polars not installed")
        nodes = pl.from_pandas(_nodes_pd())
    else:
        nodes = _nodes_pd()
    assert _ids(filter_by_dict(nodes, {"ts_aw": [pd.NaT]}, engine=engine)) == []


def test_polars_non_temporal_mixed_isin_declines():
    assert pl is not None
    from graphistry.compute.predicates.is_in import is_in

    nodes = pl.DataFrame({"v": [1, 2, 3]})
    with pytest.raises(NotImplementedError):
        filter_by_dict(nodes, {"v": is_in([1, "a"])}, engine="polars")


def test_polars_membership_empty_null_and_three_valued():
    engine = "polars"
    if pl is None:
        pytest.skip("polars not installed")
    nodes = pl.from_pandas(_nodes_pd_unit("ns"))
    exact = pd.Timestamp(1, unit="ns").isoformat()
    assert _ids(filter_by_dict(nodes, {"ts": []}, engine=engine)) == []
    assert _ids(filter_by_dict(nodes, {"ts": [None]}, engine=engine)) == []
    assert _ids(filter_by_dict(
        nodes, {"ts": [exact, None]}, engine=engine
    )) == ["exact"]


@pytest.mark.parametrize("column", ["ts_aw", "ts_naive"])
def test_pandas_unparseable_membership_fails_typed(column):
    with pytest.raises(GFQLSchemaError) as exc_info:
        filter_by_dict(_nodes_pd(), {column: ["not-a-timestamp"]}, engine="pandas")
    assert exc_info.value.code == ErrorCode.E302
    assert exc_info.value.context["field"] == column


@pytest.mark.skipif(
    cudf is None or os.environ.get("TEST_CUDF") != "1",
    reason="cuDF lane: set TEST_CUDF=1 (e.g. dgx-spark)",
)
def test_cudf_aware_datetime_membership_matches_typed_instant():
    graph = _graph("cudf")
    assert isinstance(graph._nodes, cudf.DataFrame)
    query = "MATCH (n) WHERE n.ts_aw IN ['2021-06-15T08:30:00'] RETURN n.id"
    assert _ids(graph.gfql(query, engine="cudf")._nodes) == ["q"]


def test_polars_aware_datetime_membership_matches_typed_instant():
    if pl is None:
        pytest.skip("polars not installed")
    nodes = filter_by_dict(
        pl.from_pandas(_nodes_pd()), {"ts_aw": ["2021-06-15T08:30:00"]}, engine="polars"
    )
    assert _ids(nodes) == ["q"]


def test_polars_cypher_literal_membership_matches_typed_instant():
    if pl is None:
        pytest.skip("polars not installed")
    query = "MATCH (n) WHERE n.ts_aw IN ['2021-06-15T08:30:00'] RETURN n.id"
    assert _ids(_graph("polars").gfql(query, engine="polars")._nodes) == ["q"]


def test_polars_unparseable_membership_fails_typed():
    if pl is None:
        pytest.skip("polars not installed")
    with pytest.raises(GFQLSchemaError) as exc_info:
        filter_by_dict(
            pl.from_pandas(_nodes_pd()), {"ts_aw": ["not-a-timestamp"]}, engine="polars"
        )
    assert exc_info.value.code == ErrorCode.E302
    assert exc_info.value.context["field"] == "ts_aw"
