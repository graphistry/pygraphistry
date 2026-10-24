from __future__ import annotations

from typing import Any, cast, List, Optional

import pandas as pd
import pytest
from graphistry.tests.test_compute import CGFull


class _CypherTestGraph(CGFull):
    pass


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    return cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


def _mk_polars_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    pl = pytest.importorskip("polars")
    return cast(
        _CypherTestGraph,
        _CypherTestGraph().nodes(pl.from_pandas(nodes_df), "id").edges(pl.from_pandas(edges_df), "s", "d"),
    )


def _require_cudf_runtime() -> Any:
    cudf = pytest.importorskip("cudf")
    try:
        _ = cudf.Series([1, 2, 3])
    except Exception as exc:  # pragma: no cover - environment-dependent
        pytest.skip(f"cudf installed but runtime is unavailable: {exc}")
    return cudf


def _mk_cudf_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    cudf = _require_cudf_runtime()
    return cast(
        _CypherTestGraph,
        _CypherTestGraph().nodes(cudf.from_pandas(nodes_df), "id").edges(cudf.from_pandas(edges_df), "s", "d"),
    )


def _mk_graph_for_engine(engine: str, nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    if engine == "cudf":
        return _mk_cudf_graph(nodes_df, edges_df)
    if engine == "polars":
        return _mk_polars_graph(nodes_df, edges_df)
    return _mk_graph(nodes_df, edges_df)


def _result_ids(result: Any) -> list[Optional[str]]:
    nodes = result._nodes
    if hasattr(nodes, "to_arrow"):
        nodes = nodes.to_arrow().to_pandas()
    elif hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    return nodes["id"].tolist()


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "MATCH (n {v: null}) WHERE n.v IS NULL RETURN n.id AS id",
            [],
            id="pattern-null-and-is-null",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v = null AND n.v <> 'x' RETURN n.id AS id",
            [],
            id="equality-and-inequality",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v <> 'x' AND n.v = null RETURN n.id AS id",
            [],
            id="inequality-and-equality",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v = null AND n.v IS NOT NULL RETURN n.id AS id",
            [],
            id="equality-and-is-not-null",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v = null RETURN n.id AS id",
            [],
            id="equality-alone",
        ),
        pytest.param(
            "MATCH (n) WHERE null = n.v RETURN n.id AS id",
            [],
            id="reversed-equality",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v IS NOT NULL AND (n.v = 'x' OR n.v = 'y') RETURN n.id AS id",
            ["b", "c"],
            id="positive-or-control",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v IS NULL AND NOT n.v = null RETURN n.id AS id",
            [],
            id="contradictory-not-equality",
        ),
    ],
)
def test_cypher_null_equality_conjunction(engine: str, query: str, expected: List[Optional[str]]) -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "v": [None, "x", "y"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    result = _mk_graph_for_engine(engine, nodes_df, edges_df).gfql(query, strict="strict")

    assert _result_ids(result) == expected


def test_raw_json_filter_dict_null_is_no_match() -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "v": [None, "x", "y"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    result = _mk_graph(nodes_df, edges_df).gfql([{"type": "Node", "filter_dict": {"v": None}}], strict="strict")

    assert _result_ids(result) == []
