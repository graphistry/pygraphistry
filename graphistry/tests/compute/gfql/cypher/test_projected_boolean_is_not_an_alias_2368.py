"""A boolean value projected in a WITH is a value, never a node/relationship alias (#2368)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLCallExecutionError


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


_POLARS = {"polars", "polars-gpu"}


def _run(engine, query):
    out = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


def _expect_value_type_error(engine, query):
    """pandas and cuDF raise the row-expression type error; the Polars engines decline the `select` row op (NotImplementedError)."""
    with pytest.raises(NotImplementedError if engine in _POLARS else GFQLCallExecutionError):
        _run(engine, query)


_VALUE_MISUSE = [
    "MATCH (n) WITH n, n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n.flag AS t RETURN properties(t) AS v",
    "MATCH (n) WITH n, n.flag AS t RETURN keys(t) AS v",
    "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(t) AS v",
]
_SHADOWING = [  # a projected value that reuses an alias name is a value, not the alias
    "MATCH (n) WITH n.flag AS n RETURN properties(n) AS v",
    "MATCH (n) WITH n.flag AS n RETURN keys(n) AS v",
    "MATCH (n) WITH n.flag AS n RETURN n.name AS v",
    "MATCH (n) WITH n ORDER BY n.id LIMIT 2 WITH n.flag AS n RETURN properties(n) AS v",
    "MATCH (n) WITH DISTINCT n WITH n.flag AS n RETURN keys(n) AS v",
    "MATCH (n) WITH n.flag AS n, count(*) AS c RETURN properties(n) AS v",
    "MATCH ()-[r]->() WITH r.ok AS r RETURN properties(r) AS v",
]
_NO_MATCH = [  # a boolean from a source that has no MATCH alias at all
    "UNWIND [true, false] AS b RETURN properties(b) AS v",
    "UNWIND [true, false] AS b RETURN keys(b) AS v",
    "WITH true AS b RETURN properties(b) AS v",
    "WITH true AS b RETURN b.name AS v",
]
_DOTTED_READ = "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v"


@pytest.mark.parametrize("query", _VALUE_MISUSE + _SHADOWING + _NO_MATCH)
def test_a_projected_boolean_is_a_type_error_for_entity_functions(engine, query):
    _expect_value_type_error(engine, query)


def test_a_projected_boolean_is_a_type_error_for_a_dotted_property_read(engine):
    if engine in _POLARS:
        pytest.skip("the Polars native lowering answers null for a property of a boolean (issue #2384); #2368 observed pandas and cuDF")
    _expect_value_type_error(engine, _DOTTED_READ)


def test_real_aliases_beside_a_projected_boolean_keep_working(engine):
    node_query = "MATCH (n) WITH n, n.flag AS t RETURN properties(n) AS v, t AS b"
    rel_query = "MATCH ()-[r]->() WITH r, r.ok AS t RETURN properties(r) AS v, t AS b"
    if engine in _POLARS:
        for query in (node_query, rel_query):
            with pytest.raises(NotImplementedError):
                _run(engine, query)
        return
    rows = _run(engine, node_query)
    assert sorted(row["b"] for row in rows) == [False, True]
    assert sorted(row["v"] for row in rows) == ["{flag: false, name: 'b'}", "{flag: true, name: 'a'}"]
    rel = _run(engine, rel_query)
    assert [row["b"] for row in rel] == [True]
    assert [row["v"] for row in rel] == ["{ok: true}"]


def test_a_renamed_alias_is_still_an_alias(engine):
    query = "MATCH (n) WITH n AS m, n.flag AS t RETURN properties(m) AS v, t AS b"
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):
            _run(engine, query)
        return
    rows = _run(engine, query)
    assert sorted(row["v"] for row in rows) == ["{flag: false, name: 'b'}", "{flag: true, name: 'a'}"]


def test_a_map_variable_and_a_plain_alias_are_unchanged(engine):
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):
            _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")
        return
    assert [row["v"] for row in _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")] == ["{a: 1}", "{a: 1}"]
    assert len(_run(engine, "MATCH (n) RETURN properties(n) AS v")) == 2
