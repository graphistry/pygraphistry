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
    "UNWIND [true] AS a UNWIND [false] AS b RETURN properties(a) AS v",
    "UNWIND [true] AS a UNWIND [false] AS b RETURN properties(b) AS v",
]
_DOTTED_READS = [
    "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v",
    "MATCH (n) WITH n.flag AS t RETURN t.name AS v",
    "MATCH (n) WITH n, n.id AS t RETURN t.name AS v",
    "MATCH (n) WITH n.flag AS t WHERE t.name = 'a' RETURN t AS v",
    "MATCH (n) WITH n.flag AS t RETURN t AS v ORDER BY t.name",
    "MATCH (n) WITH n.flag AS t RETURN t.name AS k, count(*) AS c",
    "MATCH (n) WITH n.flag AS t RETURN collect(t.name) AS v",
]
_TYPE_ERROR = "property access requires a graph element alias, entity value, or map"


@pytest.mark.parametrize("query", _VALUE_MISUSE + _SHADOWING + _NO_MATCH)
def test_a_projected_boolean_is_a_type_error_for_entity_functions(engine, query):
    _expect_value_type_error(engine, query)


@pytest.mark.parametrize("query", _DOTTED_READS)
def test_a_dotted_property_read_of_a_projected_scalar_is_a_type_error(engine, query):
    """Every engine, Polars lowering included, raises the row-expression type error (#2384)."""
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run(engine, query)


def test_a_dotted_property_read_of_a_projected_string_is_a_type_error_or_a_polars_decline(engine):
    """pandas' answer for a string depends on its values (entity or duration text), so Polars declines."""
    query = "MATCH (n) WITH n, n.name AS t RETURN t.name AS v"
    with pytest.raises(NotImplementedError if engine in _POLARS else GFQLCallExecutionError):
        _run(engine, query)


def _run_on(engine, flags, query):
    nodes = pd.DataFrame({"id": list(range(len(flags))), "name": ["x"] * len(flags), "flag": pd.array(flags, dtype="boolean")})
    if engine == "cudf":
        import cudf

        nodes = cudf.from_pandas(nodes)
    elif engine in _POLARS:
        import polars as pl

        nodes = pl.from_pandas(nodes)
    out = graphistry.nodes(nodes, "id").gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out


def test_a_dotted_property_read_of_a_projected_boolean_with_some_nulls_is_a_type_error(engine):
    with pytest.raises(GFQLCallExecutionError, match=_TYPE_ERROR):
        _run_on(engine, [True, None], "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v")


def test_a_dotted_property_read_of_an_all_null_projected_boolean_is_null(engine):
    out = _run_on(engine, [None, None], "MATCH (n) WITH n, n.flag AS t RETURN t.name AS v")
    assert out["v"].isna().all() and len(out) == 2


def test_a_dotted_property_read_of_a_projected_boolean_over_no_rows_is_empty(engine):
    out = _run_on(engine, [True, False], "MATCH (n) WHERE n.id > 100 WITH n, n.flag AS t RETURN t.name AS v")
    assert len(out) == 0


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
    assert sorted(row["v"] for row in rows) == ["{id: 1, name: 'a', flag: true}", "{id: 2, name: 'b', flag: false}"]
    rel = _run(engine, rel_query)
    assert [row["b"] for row in rel] == [True]
    assert [row["v"] for row in rel] == ["{ok: true}"]


def test_a_map_variable_and_a_plain_alias_are_unchanged(engine):
    if engine in _POLARS:
        with pytest.raises(NotImplementedError):
            _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")
        return
    assert [row["v"] for row in _run(engine, "MATCH (n) WITH n, {a: 1} AS m RETURN properties(m) AS v")] == ["{a: 1}", "{a: 1}"]
    assert len(_run(engine, "MATCH (n) RETURN properties(n) AS v")) == 2
