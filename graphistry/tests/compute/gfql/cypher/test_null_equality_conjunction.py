from __future__ import annotations

import os
from typing import Any, cast, List, Optional

import pandas as pd
import pytest
from graphistry.tests.test_compute import CGFull
from graphistry.Engine import series_to_pylist
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError


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
    if os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cuDF lane requires TEST_CUDF=1")
    try:
        import cudf
    except ModuleNotFoundError as exc:
        pytest.fail(f"TEST_CUDF=1 requires the cuDF package: {exc}")
    try:
        _ = cudf.Series([1, 2, 3])
    except Exception as exc:
        pytest.fail(f"TEST_CUDF=1 requires a usable cuDF runtime: {exc}")
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
    return cast(List[Optional[str]], series_to_pylist(result._nodes["id"]))


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize(
    ("query", "expected"),
    [
        pytest.param(
            "MATCH (n) WHERE NOT (n.v = null) RETURN n.id AS id",
            [],
            id="not-null-equality",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v = null OR true RETURN n.id AS id",
            ["a", "b", "c"],
            id="null-equality-or-true",
        ),
        pytest.param(
            "MATCH (n) WHERE n.v = null OR false RETURN n.id AS id",
            [],
            id="null-equality-or-false",
        ),
        pytest.param(
            "MATCH (n) WITH n.v = null AS unknown RETURN unknown IS NULL AS known",
            [True, True, True],
            id="null-equality-projection-unknown",
        ),
    ],
)
def test_cypher_null_unknown_controls(
    engine: str, query: str, expected: List[object]
) -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "v": [None, "x", "y"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    result = _mk_graph_for_engine(engine, nodes_df, edges_df).gfql(query, strict="strict")

    output_name = "known" if "known" in query else "id"
    assert series_to_pylist(result._nodes[output_name]) == expected


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_cypher_missing_property_null_equality_errors_in_strict_mode(engine: str) -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    graph = _mk_graph_for_engine(engine, nodes_df, edges_df)
    with pytest.raises(GFQLSchemaError) as exc_info:
        graph.gfql("MATCH (n) WHERE n.missing = null RETURN n.id AS id", strict="strict")
    assert exc_info.value.code == ErrorCode.E301
    assert exc_info.value.context["field"] == "missing"
    assert exc_info.value.context["suggestion"].startswith("Available columns:")


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
            "MATCH (n) WHERE n.v = null AND n.v = null RETURN n.id AS id",
            [],
            id="repeated-equality",
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


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
def test_cypher_null_where_parameter_matches_nothing(engine: str) -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "v": [None, "x", "y"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    result = _mk_graph_for_engine(engine, nodes_df, edges_df).gfql(
        "MATCH (n) WHERE n.v = $p RETURN n.id AS id",
        params={"p": None},
        strict="strict",
    )

    assert _result_ids(result) == []


def test_raw_json_filter_dict_null_is_no_match() -> None:
    nodes_df = pd.DataFrame({"id": ["a", "b", "c"], "v": [None, "x", "y"]})
    edges_df = pd.DataFrame({"s": ["a"], "d": ["b"]})
    result = _mk_graph(nodes_df, edges_df).gfql([{"type": "Node", "filter_dict": {"v": None}}], strict="strict")

    assert _result_ids(result) == []
