"""A boolean value projected in a WITH is a value, never a node/relationship alias (#2368)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError


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


_NODES = pd.DataFrame({"id": [1, 2], "name": ["a", "b"], "flag": [True, False]})
_EDGES = pd.DataFrame({"s": [1], "d": [2], "ok": [True]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _rows(engine, query):
    """Rows, or None when a Polars engine declines the query (parity-or-error)."""
    try:
        out = _graph(engine).gfql(query, engine=engine)._nodes
    except NotImplementedError:
        assert engine in {"polars", "polars-gpu"}, "only the Polars engines may decline"
        return None
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


_VALUE_MISUSE = [
    "MATCH (n) WITH n, n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n, n.flag AS t RETURN keys(t) AS v",
    "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(t) AS v",
]
_DOTTED_READ = "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v"


def _assert_type_error(engine, query):
    with pytest.raises((GFQLValidationError, NotImplementedError)) as caught:
        _graph(engine).gfql(query, engine=engine)
    assert "unsupported-cypher-query" not in str(caught.value), "must be the value type error, not a compiler decline"


@pytest.mark.parametrize("query", _VALUE_MISUSE)
def test_a_projected_boolean_is_a_type_error_for_entity_functions(engine, query):
    _assert_type_error(engine, query)


def test_a_projected_boolean_is_a_type_error_for_a_dotted_property_read(engine):
    if engine in {"polars", "polars-gpu"}:
        pytest.skip("the Polars native lowering answers null for a property of a boolean (separate from #2368, which observed pandas and cuDF)")
    _assert_type_error(engine, _DOTTED_READ)


def test_real_aliases_beside_a_projected_boolean_keep_working(engine):
    rows = _rows(engine, "MATCH (n) WITH n, n.flag AS t RETURN properties(n) AS v, t AS b")
    assert rows is None or (sorted(bool(row["b"]) for row in rows) == [False, True] and all("flag" in str(row["v"]) and "n:" not in str(row["v"]) for row in rows))
    rel = _rows(engine, "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(r) AS v, t AS b")
    assert rel is None or (len(rel) == 1 and bool(rel[0]["b"]) is True and "ok" in str(rel[0]["v"]) and "r:" not in str(rel[0]["v"]))


def test_a_map_variable_and_a_plain_alias_are_unchanged(engine):
    mapped = _rows(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")
    assert mapped is None or all(row["v"] == "{a: 1}" or row["v"] == {"a": 1} for row in mapped)
    plain = _rows(engine, "MATCH (n) RETURN properties(n) AS v")
    assert plain is None or len(plain) == 2
