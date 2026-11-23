import os

import pandas as pd
import pytest

import graphistry


def _engine_frame(engine):
    frame = pd.DataFrame({"id": ["a", "b"], "i": [1, 2], "s": ["1a", "b"]})
    if engine == "pandas":
        return frame
    if engine == "cudf":
        import cudf

        return cudf.from_pandas(frame)
    import polars as pl

    return pl.from_pandas(frame)


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


@pytest.mark.parametrize("query,expected", [
    ("RETURN 1 STARTS WITH '1' AS v", [None]),
    ("RETURN 1 ENDS WITH '1' AS v", [None]),
    ("RETURN 1 CONTAINS '1' AS v", [None]),
    ("RETURN 'abc' STARTS WITH 1 AS v", [None]),
    ("RETURN 'abc' ENDS WITH 1 AS v", [None]),
    ("RETURN 'abc' CONTAINS 1 AS v", [None]),
    ("RETURN true STARTS WITH 't' AS v", [None]),
    ("RETURN null STARTS WITH 'a' AS v", [None]),
])
def test_nonstring_string_predicates_return_null(engine, query, expected):
    graph = graphistry.nodes(_engine_frame(engine), "id")
    rows = _rows(graph.gfql(query, engine=engine)._nodes)
    assert [row["v"] for row in rows] == expected


@pytest.mark.parametrize("query,expected,key", [
    ("MATCH (n) RETURN n.i STARTS WITH '1' AS v", [None, None], "v"),
    ("MATCH (n) RETURN n.i ENDS WITH '1' AS v", [None, None], "v"),
    ("MATCH (n) RETURN n.i CONTAINS '1' AS v", [None, None], "v"),
    ("MATCH (n) RETURN n.s STARTS WITH 1 AS v", [None, None], "v"),
    ("MATCH (n) RETURN n.s ENDS WITH 1 AS v", [None, None], "v"),
    ("MATCH (n) RETURN n.s CONTAINS 1 AS v", [None, None], "v"),
    ("MATCH (n) WHERE n.i STARTS WITH '1' RETURN n.id AS id", [], "id"),
])
def test_nonstring_column_string_predicates_return_null(engine, query, expected, key):
    graph = graphistry.nodes(_engine_frame(engine), "id")
    rows = _rows(graph.gfql(query, engine=engine)._nodes)
    if expected and isinstance(expected[0], dict):
        assert rows == expected
    else:
        assert [row[key] for row in rows] == expected


def test_string_column_string_predicates_keep_the_contract(engine):
    graph = graphistry.nodes(_engine_frame(engine), "id")
    rows = _rows(graph.gfql("MATCH (n) WHERE n.s STARTS WITH '1' RETURN n.id AS id", engine=engine)._nodes)
    assert [row["id"] for row in rows] == ["a"]
