"""A projected literal is evaluated, even when its text equals a column name (#2346).

Row projections used to resolve the expression text as a column name first, so
`RETURN 5 AS v` over a table with a column named `5` returned that column.
"""
import pandas as pd
import pytest

import graphistry

ENGINES = ["pandas", "polars", "cudf"]

_NODES = pd.DataFrame({"id": [1, 2], "5": ["p", "q"], "true": ["a", "b"], "null": ["x", "y"]})
_EDGES = pd.DataFrame({"src": [1], "dst": [2]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "polars":
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


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize(
    "literal, expected",
    [("5", 5), ("true", True), ("null", None)],
)
def test_literal_named_like_a_column_is_evaluated(engine, literal, expected):
    result = _graph(engine).gfql(f"MATCH (n) RETURN {literal} AS v", engine=engine)
    assert _column(result, "v") == [expected, expected]


@pytest.mark.parametrize("engine", ENGINES)
def test_literal_named_like_a_column_survives_with(engine):
    result = _graph(engine).gfql("MATCH (n) WITH 5 AS v RETURN v", engine=engine)
    assert _column(result, "v") == [5, 5]


@pytest.mark.parametrize("engine", ENGINES)
def test_column_named_like_a_literal_is_still_read_by_property(engine):
    result = _graph(engine).gfql("MATCH (n) RETURN n.`5` AS v", engine=engine)
    assert _column(result, "v") == ["p", "q"]
