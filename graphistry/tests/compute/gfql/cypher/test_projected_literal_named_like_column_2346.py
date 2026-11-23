"""A projected literal is evaluated, even when its text equals a column name (#2346).

Row projections used to resolve the expression text as a column name first, so
`RETURN 5 AS v` over a table with a column named `5` returned that column.
"""
import os

import pandas as pd
import pytest

import graphistry

ENGINES = [
    "pandas",
    "polars",
    pytest.param(
        "polars-gpu",
        marks=pytest.mark.skipif(
            os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu"
        ),
    ),
    pytest.param(
        "cudf",
        marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"),
    ),
]

_NODES = pd.DataFrame(
    {
        "id": [1, 2],
        "5": ["p", "q"],
        "-5": ["m", "n"],
        "5.0": ["f", "g"],
        "true": ["a", "b"],
        "false": ["c", "d"],
        "null": ["x", "y"],
    }
)
_EDGES = pd.DataFrame({"src": [1], "dst": [2]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine.startswith("polars"):
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        pytest.importorskip("cupy")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _column(result, name):
    frame = result._nodes
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return [None if pd.isna(value) else value for value in frame[name].tolist()]


def _kind(result, name):
    frame = result._nodes
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame[name].dtype.kind


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "literal, expected",
    [("5", 5), ("-5", -5), ("5.0", 5.0), ("true", True), ("false", False), ("null", None)],
)
def test_literal_named_like_a_column_is_evaluated(engine, literal, expected):
    result = _graph(engine).gfql(f"MATCH (n) RETURN {literal} AS v", engine=engine)
    assert _column(result, "v") == [expected, expected]
    if isinstance(expected, bool):
        assert _kind(result, "v") == "b"
    elif isinstance(expected, int):
        assert _kind(result, "v") == "i"
    elif isinstance(expected, float):
        assert _kind(result, "v") == "f"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("literal, expected_ids", [("true", [1, 2]), ("false", [])])
def test_boolean_predicate_named_like_a_column_is_evaluated(engine, literal, expected_ids):
    result = _graph(engine).gfql(f"MATCH (n) WHERE {literal} RETURN n.id AS i", engine=engine)
    assert _column(result, "i") == expected_ids


@pytest.mark.parametrize("engine", ENGINES)
def test_literal_named_like_a_column_survives_with(engine):
    result = _graph(engine).gfql("MATCH (n) WITH 5 AS v RETURN v", engine=engine)
    assert _column(result, "v") == [5, 5]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("column, expected", [("5", ["p", "q"]), ("true", ["a", "b"]), ("null", ["x", "y"])])
def test_column_named_like_a_literal_is_still_read_by_property(engine, column, expected):
    result = _graph(engine).gfql(f"MATCH (n) RETURN n.`{column}` AS v", engine=engine)
    assert _column(result, "v") == expected
