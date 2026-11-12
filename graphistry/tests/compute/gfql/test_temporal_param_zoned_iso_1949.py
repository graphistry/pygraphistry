"""#1949: Cypher parameter values follow the same temporal routing as literals."""

from __future__ import annotations

import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError

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
        "id": ["p", "q", "r", "s", "t"],
        "ts": pd.to_datetime([
            "2020-06-15T08:30:00", "2021-06-15T08:30:00",
            "2022-01-01T00:00:00", "2019-01-01T00:00:00", None,
        ]),
    })


def _edges_pd() -> pd.DataFrame:
    return pd.DataFrame({"s": ["p"], "d": ["q"]})


def _graph(engine: str):
    nodes, edges = _nodes_pd(), _edges_pd()
    if engine == "polars":
        assert pl is not None
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        assert cudf is not None
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(cudf.from_pandas(edges), "s", "d")
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _ids(
    graph,
    query: str,
    *,
    engine: str | None = None,
    params: dict | None = None,
) -> list:
    nodes = graph.gfql(
        query, engine=engine or _engine_name(graph), params=params
    )._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return sorted(nodes["n.id"].dropna().tolist())


def _pairs(
    graph,
    query: str,
    *,
    engine: str | None = None,
    params: dict | None = None,
) -> list:
    nodes = graph.gfql(
        query, engine=engine or _engine_name(graph), params=params
    )._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    rows = []
    for left, right in nodes[["a.id", "b.id"]].itertuples(index=False, name=None):
        rows.append((left, None if pd.isna(right) else right))
    return sorted(rows, key=lambda row: row[0])


def _engine_name(graph) -> str:
    if pl is not None and isinstance(graph._nodes, (pl.DataFrame, pl.LazyFrame)):
        return "polars"
    return "pandas"


@pytest.mark.parametrize("op", ["=", ">", "<="])
def test_parameter_matches_zoned_literal(op: str):
    graph = _graph("pandas")
    cut = "2021-01-01T00:00:00+01:00"
    literal_query = f"MATCH (n) WHERE n.ts {op} '{cut}' RETURN n.id"
    parameter_query = f"MATCH (n) WHERE n.ts {op} $cut RETURN n.id"
    assert _ids(graph, literal_query) == _ids(graph, parameter_query, params={"cut": cut})


@pytest.mark.parametrize(
    ("op", "cut", "expected"),
    [
        ("=", "2021-06-15T09:30:00+01:00", ["q"]),
        (">", "2021-01-01T00:00:00Z", ["q", "r"]),
        ("<=", "2021-01-01T00:00:00Z", ["p", "s"]),
    ],
)
def test_parameter_matches_zoned_independent_rows(
    op: str, cut: str, expected: list[str]
):
    graph = _graph("pandas")
    literal_query = f"MATCH (n) WHERE n.ts {op} '{cut}' RETURN n.id"
    parameter_query = f"MATCH (n) WHERE n.ts {op} $cut RETURN n.id"
    assert _ids(graph, literal_query) == expected
    assert _ids(graph, parameter_query, params={"cut": cut}) == expected


def test_connected_join_parameter_has_independent_rows():
    graph = _graph("pandas")
    cut = "2021-01-01T00:00:00Z"
    literal_query = f"MATCH (a)-[]->(b) WHERE b.ts > '{cut}' RETURN a.id, b.id"
    parameter_query = "MATCH (a)-[]->(b) WHERE b.ts > $cut RETURN a.id, b.id"
    assert _pairs(graph, literal_query) == [("p", "q")]
    assert _pairs(graph, parameter_query, params={"cut": cut}) == [("p", "q")]


def test_optional_match_parameter_has_independent_rows():
    graph = _graph("pandas")
    cut = "2021-01-01T00:00:00Z"
    literal_query = (
        f"MATCH (a) OPTIONAL MATCH (a)-[]->(b) WHERE b.ts > '{cut}' "
        "RETURN a.id, b.id"
    )
    parameter_query = (
        "MATCH (a) OPTIONAL MATCH (a)-[]->(b) WHERE b.ts > $cut "
        "RETURN a.id, b.id"
    )
    expected = [
        ("p", "q"), ("q", None), ("r", None), ("s", None), ("t", None),
    ]
    assert _pairs(graph, literal_query) == expected
    assert _pairs(graph, parameter_query, params={"cut": cut}) == expected


@pytest.mark.skipif(
    cudf is None or os.environ.get("TEST_CUDF") != "1",
    reason="cuDF lane: set TEST_CUDF=1 (e.g. dgx-spark)",
)
def test_cudf_public_route_literal_parameter_and_controls():
    graph = _graph("cudf")
    assert isinstance(graph._nodes, cudf.DataFrame)
    assert isinstance(graph._edges, cudf.DataFrame)
    cut = "2021-01-01T00:00:00Z"
    assert _ids(
        graph,
        f"MATCH (n) WHERE n.ts > '{cut}' RETURN n.id",
        engine="cudf",
    ) == ["q", "r"]
    assert _ids(
        graph,
        "MATCH (n) WHERE n.ts > $cut RETURN n.id",
        engine="cudf",
        params={"cut": cut},
    ) == ["q", "r"]
    assert _ids(
        graph,
        "MATCH (n) WHERE n.ts > $cut RETURN n.id",
        engine="cudf",
        params={"cut": "2023-01-01T00:00:00Z"},
    ) == []


def test_connected_join_and_optional_match_parameter_parity():
    graph = _graph("pandas")
    cut = "2021-01-01T00:00:00Z"
    for prefix in ("MATCH (a)-[]->(b) WHERE b.ts", "MATCH (a) OPTIONAL MATCH (a)-[]->(b) WHERE b.ts"):
        literal_query = f"{prefix} > '{cut}' RETURN a.id, b.id"
        parameter_query = f"{prefix} > $cut RETURN a.id, b.id"
        literal_nodes = graph.gfql(literal_query, engine="pandas")._nodes
        parameter_nodes = graph.gfql(
            parameter_query, engine="pandas", params={"cut": cut}
        )._nodes
        literal_nodes = literal_nodes.to_pandas() if hasattr(literal_nodes, "to_pandas") else literal_nodes
        parameter_nodes = parameter_nodes.to_pandas() if hasattr(parameter_nodes, "to_pandas") else parameter_nodes
        pd.testing.assert_frame_equal(literal_nodes, parameter_nodes)


def test_polars_zoned_row_residues_typed_decline():
    if pl is None:
        pytest.skip("polars not installed")
    graph = _graph("polars")
    for query in (
        "MATCH (n) WHERE n.ts > '2021-01-01T00:00:00Z' RETURN n.id",
        "MATCH (n) WHERE n.ts > $cut RETURN n.id",
    ):
        with pytest.raises(NotImplementedError, match="where_rows"):
            graph.gfql(query, engine="polars", params={"cut": "2021-01-01T00:00:00Z"})


def test_missing_parameter_declines_typed():
    with pytest.raises(GFQLValidationError):
        _graph("pandas").gfql(
            "MATCH (n) WHERE n.ts > $cut RETURN n.id", engine="pandas"
        )


def test_non_temporal_parameter_is_not_temporal_residual():
    graph = _graph("pandas")
    assert _ids(
        graph,
        "MATCH (n) WHERE n.id = $who RETURN n.id",
        params={"who": "q"},
    ) == ["q"]
