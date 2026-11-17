import os

import pandas as pd
import pytest

import graphistry


def _nodes(engine):
    nodes = pd.DataFrame({
        "id": ["a", "b"],
        "type": ["P", "Q"],
        "label__Foo": [True, False],
        "name": ["x", "y"],
    })
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(nodes)
    if engine == "polars":
        import polars as pl

        return pl.from_pandas(nodes)
    return nodes


def _rows(frame):
    if hasattr(frame, "to_pandas"):
        frame = frame.to_pandas()
    return frame.to_dict("records")


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


@pytest.mark.parametrize("key,expected", [
    ("type", ["P", "Q"]),
    ("id", ["a", "b"]),
    ("label__Foo", [True, False]),
    ("name", ["x", "y"]),
], ids=["type", "id", "label-flag", "property"])
def test_constant_string_subscript_matches_dotted_property(engine, key, expected):
    graph = graphistry.nodes(_nodes(engine), "id")
    rows = _rows(graph.gfql(
        f"MATCH (n) RETURN n['{key}'] AS v, n.{key} AS dotted",
        engine=engine,
    )._nodes)
    assert [(row["v"], row["dotted"]) for row in rows] == list(zip(expected, expected))


@pytest.mark.parametrize("key_expr,expected", [
    ("'nam' + 'e'", ["x", "y"]),
    ("$key", ["x", "y"]),
], ids=["constant-concatenation", "parameter"])
def test_nonliteral_constant_string_subscript_matches_dotted_property(engine, key_expr, expected):
    graph = graphistry.nodes(_nodes(engine), "id")
    rows = _rows(graph.gfql(
        f"MATCH (n) RETURN n[{key_expr}] AS v",
        engine=engine,
        params={"key": "name"},
    )._nodes)
    assert [row["v"] for row in rows] == expected
