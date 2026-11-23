"""A string parameter is a string literal in every row-expression position (#2324)."""
import os

import pandas as pd
import pytest

import graphistry


@pytest.fixture(params=["pandas", "cudf", "polars", "polars-gpu"])
def engine(request):
    if request.param == "cudf":
        pytest.importorskip("cudf")
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cudf tests need TEST_CUDF=1")
    elif request.param == "polars-gpu":
        pytest.importorskip("cudf_polars")
        if os.environ.get("TEST_POLARS_GPU") != "1":
            pytest.skip("requires actual polars-gpu")
    elif request.param == "polars":
        pytest.importorskip("polars")
    return request.param


_NODES = pd.DataFrame({"id": [1, 2], "name": ["a", "id"]})
_EDGES = pd.DataFrame({"src": [1], "dst": [2]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.edges(edges, "src", "dst").nodes(nodes, "id")


def _rows(engine, query, value):
    """Rows of the query on every engine: none of them declines a string parameter today."""
    out = _graph(engine).gfql(query, params={"v": value}, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


# A string equal to a column name or to expression text must stay a string, never a column or an expression.
_VALUES = [
    "x", "name", "n.id", "it's", "a b", "",
    'say "hi"', "a\\b", "a\\'b", "a\\\\'b", "line1\nline2", "h\u00e9llo \u2713",
    "1 + 1", "$v", "null", "true", "n.id + 1", "'; RETURN 1 //",
]


@pytest.mark.parametrize("value", _VALUES)
def test_return_of_a_string_parameter_is_the_string(engine, value):
    rows = _rows(engine, "MATCH (n) WHERE n.id = 1 RETURN $v AS v", value)
    assert rows == [{"v": value}]


@pytest.mark.parametrize("value", _VALUES)
def test_with_then_return_carries_the_string(engine, value):
    rows = _rows(engine, "MATCH (n) WITH $v AS p, n RETURN p, n.id AS i ORDER BY i", value)
    assert rows == [{"p": value, "i": 1}, {"p": value, "i": 2}]


@pytest.mark.parametrize("value", _VALUES)
def test_aggregates_and_grouping_over_a_string_parameter(engine, value):
    count = _rows(engine, "MATCH (n) RETURN count($v) AS c", value)
    assert count == [{"c": 2}]
    grouped = _rows(engine, "MATCH (n) RETURN $v AS v, count(*) AS c", value)
    assert grouped == [{"v": value, "c": 2}]


def test_collect_of_a_string_parameter(engine):
    rows = _rows(engine, "MATCH (n) RETURN collect($v) AS c", "x")
    assert len(rows) == 1 and list(rows[0]["c"]) == ["x", "x"]  # Polars returns the list as an array


@pytest.mark.parametrize("value", _VALUES)
def test_where_and_unwind_with_a_string_parameter_keep_their_answers(engine, value):
    where = _rows(engine, "MATCH (n) WHERE n.name = $v RETURN n.id AS i", "id")
    assert where == [{"i": 2}]
    unwound = _rows(engine, "UNWIND [$v] AS x RETURN x", value)
    assert unwound == [{"x": value}]


def test_non_string_parameters_are_unchanged(engine):
    rows = _rows(engine, "MATCH (n) WHERE n.id = 1 RETURN $v AS v", 5)
    assert rows == [{"v": 5}]
