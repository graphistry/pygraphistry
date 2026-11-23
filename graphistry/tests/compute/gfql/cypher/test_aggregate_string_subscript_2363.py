"""An aggregate over a string subscript (count(a['name'])) answers exactly like the dotted spelling (#2363)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLCallExecutionError, GFQLUnsupportedError


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


_NODES = pd.DataFrame({"id": [1, 2, 3], "name": ["x", "y", "z"], "w": [5, 7, 9]})
_EDGES = pd.DataFrame({"s": [1, 2], "d": [2, 3], "kind": ["k1", "k2"], "t": [10, 20]})


def _graph(engine):
    nodes, edges = _NODES, _EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in {"polars", "polars-gpu"}:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _run(engine, query):
    out = _graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return [{key: (list(value) if hasattr(value, "tolist") and not isinstance(value, (int, float, str)) else value) for key, value in row.items()} for row in out.to_dict("records")]


def _rows(engine, query):
    """Rows of the query. The one decline today: polars-gpu when the installed cudf-polars cannot run the plan on the GPU
    (every aggregate on 25.02, the cartesian count on 26.02) raises NotImplementedError or GFQLUnsupportedError; that
    decline, for that engine, is returned as None and nothing else is tolerated."""
    try:
        return _run(engine, query)
    except (NotImplementedError, GFQLUnsupportedError) as exc:
        assert engine == "polars-gpu" and ("not fully GPU-executable" in str(exc) or "cannot execute this aggregation" in str(exc)), str(exc)
        return None


_ONE_HOP = "MATCH (a)-[r]->(b) RETURN {agg} AS v"
_CASES = [
    ("count(a{k})", "name", 2),
    ("count(b{k})", "name", 2),
    ("collect(a{k})", "name", ["x", "y"]),
    ("sum(a{k})", "w", 12),
    ("sum(b{k})", "w", 16),
    ("avg(a{k})", "w", 6.0),
    ("min(b{k})", "w", 7),
    ("max(b{k})", "w", 9),
    ("count(DISTINCT b{k})", "name", 2),
]


@pytest.mark.parametrize("agg,prop,expected", _CASES, ids=[case[0] for case in _CASES])
def test_node_alias_aggregates_match_the_dotted_spelling(engine, agg, prop, expected):
    subscript = _rows(engine, _ONE_HOP.format(agg=agg.format(k=f"['{prop}']")))
    dotted = _rows(engine, _ONE_HOP.format(agg=agg.format(k=f".{prop}")))
    assert subscript == dotted
    if subscript is not None:
        assert subscript == [{"v": expected}]


@pytest.mark.parametrize("agg,expected", [("count(r{k})", 2), ("sum(r{k})", 30), ("max(r{k})", 20)])
def test_relationship_alias_aggregates_match_the_dotted_spelling(engine, agg, expected):
    prop = "kind" if agg.startswith("count") else "t"
    subscript = _rows(engine, _ONE_HOP.format(agg=agg.format(k=f"['{prop}']")))
    dotted = _rows(engine, _ONE_HOP.format(agg=agg.format(k=f".{prop}")))
    assert subscript == dotted
    if subscript is not None:
        assert subscript == [{"v": expected}]


def test_cartesian_and_grouped_aggregates_over_subscripts(engine):
    cartesian = _rows(engine, "MATCH (a), (b) RETURN count(a['name']) AS v")
    assert cartesian is None or cartesian == [{"v": 9}]
    grouped = _rows(engine, "MATCH (a)-[r]->(b) RETURN a['name'] AS k, sum(b['w']) AS v ORDER BY k")
    assert grouped is None or grouped == [{"k": "x", "v": 7}, {"k": "y", "v": 9}]
    single = _rows(engine, "MATCH (a) RETURN count(a['name']) AS v")
    assert single is None or single == [{"v": 3}]


# The subscript spelling of a non-entity value (a map, a list, a scalar, null, an UNWIND variable) answers exactly as on
# `develop`; the fix only changes which property columns are materialized for an alias. Outcomes below were measured on `develop`:
# pandas and cuDF answer or raise GFQLCallExecutionError; the Polars engines decline these WITH/UNWIND/projection row ops.
_ROWS, _ERR = "rows", "error"
_NON_ENTITY = [
    ("WITH {name: 1} AS m RETURN m['name'] AS v", _ROWS, [{"v": 1}]),
    ("WITH {name: 1} AS m RETURN count(m['name']) AS v", _ROWS, [{"v": 1}]),
    ("UNWIND [{name: 1}, {name: 2}] AS m RETURN m['name'] AS v ORDER BY v", _ROWS, [{"v": 1}, {"v": 2}]),
    ("UNWIND [{name: 1}, {name: 2}] AS m RETURN count(m['name']) AS v", _ROWS, [{"v": 2}]),
    ("UNWIND [{name: 1}, {name: 2}] AS m RETURN sum(m['name']) AS v", _ROWS, [{"v": 3}]),
    ("WITH null AS z RETURN z['name'] AS v", _ROWS, [{"v": None}]),
    ("WITH [1, 2] AS l RETURN l['name'] AS v", _ERR, None),
    ("WITH 5 AS s RETURN s['name'] AS v", _ERR, None),
    ("WITH 5 AS s RETURN count(s['name']) AS v", _ERR, None),
    ("WITH 'abc' AS s RETURN s['name'] AS v", _ERR, None),
]


@pytest.mark.parametrize("query,kind,expected", _NON_ENTITY, ids=[case[0] for case in _NON_ENTITY])
def test_a_subscript_of_a_non_entity_value_keeps_its_outcome(engine, query, kind, expected):
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises(NotImplementedError):
            _run(engine, query)
    elif kind == _ERR:
        with pytest.raises(GFQLCallExecutionError):
            _run(engine, query)
    else:
        assert _run(engine, query) == expected


# Where develop's pandas and cuDF disagree about an aggregate over a non-entity value, each keeps its own answer.
@pytest.mark.parametrize("query", [
    "WITH [1, 2] AS l RETURN count(l['name']) AS v",
    "WITH null AS z RETURN count(z['name']) AS v",
])
def test_an_aggregate_over_a_list_or_null_subscript_keeps_each_engines_develop_answer(engine, query):
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises(NotImplementedError):
            _run(engine, query)
    elif engine == "pandas":
        assert _run(engine, query) == [{"v": 0}]
    else:
        with pytest.raises(GFQLCallExecutionError):
            _run(engine, query)


# Property names that are not plain identifiers, or that collide with identity, relationship ends, aliases or internal columns.
_NAMES_NODES = pd.DataFrame({
    "id": [1, 2, 3], "name": ["x", "y", "z"], "type": ["T1", "T2", "T3"], "my name": ["m1", "m2", "m3"], "a.b": [1, 2, 3],
    "a-b": [4, 5, 6], "1x": [7, 8, 9], "h\u00e9llo": ["u", "v", "w"], "": ["e1", "e2", "e3"], "b": ["B1", "B2", "B3"], "_x": [10, 20, 30],
})
_NAMES_EDGES = pd.DataFrame({"s": [1, 2], "d": [2, 3], "type": ["k1", "k2"]})
_POLARS = {"polars", "polars-gpu"}
_NIE = "NotImplementedError"
# What polars-gpu says when the installed cudf-polars cannot run a plan on the GPU (25.02: group-by and order-by; 26.02: some joins).
_GPU_DECLINES = ("not fully GPU-executable", "cannot execute this aggregation", "does not yet natively support cypher row op 'order_by'")


def _names_graph(engine):
    nodes, edges = _NAMES_NODES, _NAMES_EDGES
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    elif engine in _POLARS:
        import polars as pl

        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _v(*values):
    return [{"v": value} for value in values]


def _ret(name):
    return "MATCH (a) RETURN a['%s'] AS v ORDER BY v" % name


def _cnt(name):
    return "MATCH (a)-[r]->(b) RETURN count(a['%s']) AS v" % name


# (query, pandas/cuDF answer, Polars answer). Every RETURN row is the answer `develop` gives, per engine family (they differ
# from each other on develop for a['b'] and a['__gfql_node_id__']); every count over a[...] raised "AST evaluator
# unsupported" on pandas/cuDF and declined on Polars before.
_SPECIAL_NAMES = [
    (_ret("id"), _v(1, 2, 3), _v(1, 2, 3)),
    (_ret("type"), _v("T1", "T2", "T3"), _v("T1", "T2", "T3")),
    (_ret("_x"), _v(10, 20, 30), _v(10, 20, 30)),
    (_ret("my name"), _v("m1", "m2", "m3"), _v("m1", "m2", "m3")),
    (_ret("a.b"), _v(1, 2, 3), _v(1, 2, 3)),
    (_ret("a-b"), _v(4, 5, 6), _v(4, 5, 6)),
    (_ret("1x"), _v(7, 8, 9), _v(7, 8, 9)),
    (_ret("h\u00e9llo"), _v("u", "v", "w"), _v("u", "v", "w")),
    (_ret(""), _v("e1", "e2", "e3"), _v("e1", "e2", "e3")),
    (_ret("missing"), _v(None, None, None), _v(None, None, None)),
    (_ret("b"), _v("B1", "B2", "B3"), _v(1, 2, 3)),  # a property named like the bound alias b
    (_ret("__gfql_node_id__"), _v(None, None, None), _v(True, True, True)),
    (_ret("__gfql_edge_index_0__"), _v(None, None, None), _v(None, None, None)),
    (_ret("__gfql_bindings_join__"), _v(None, None, None), _v(None, None, None)),
    ("MATCH (a)-[r]->(b) RETURN b['b'] AS v ORDER BY v", _v(None, None), _v("B2", "B3")),
    ("MATCH (a)-[r]->(b) RETURN r['s'] AS v ORDER BY v", _v(1, 2), _v(1, 2)),
    ("MATCH (a)-[r]->(b) RETURN r['d'] AS v ORDER BY v", _v(2, 3), _v(2, 3)),
    ("MATCH (a)-[r]->(b) RETURN r['type'] AS v ORDER BY v", _v("k1", "k2"), _v("k1", "k2")),
    ("MATCH (a)-[r]->(b) RETURN r['__gfql_edge_index_0__'] AS v ORDER BY v", _v(0, 1), _NIE),
    ("MATCH (a)-[r]->(b) RETURN count(r['s']) AS v", _v(0), _NIE),
    (_cnt("id"), _v(2), _v(2)),
    (_cnt("type"), _v(2), _v(2)),
    (_cnt("_x"), _v(2), _v(2)),
    (_cnt("my name"), _v(2), _v(2)),
    (_cnt("a.b"), _v(2), _v(2)),
    (_cnt("a-b"), _v(2), _v(2)),
    (_cnt("1x"), _v(2), _v(2)),
    (_cnt("h\u00e9llo"), _v(2), _v(2)),
    (_cnt(""), _v(2), _v(2)),
    (_cnt("b"), _v(2), _v(2)),
    (_cnt("missing"), _v(0), _NIE),
    (_cnt("__gfql_node_id__"), _v(0), _v(2)),  # develop's Polars RETURN reads identity for this name; pandas/cuDF do not
    (_cnt("__gfql_edge_index_0__"), _v(0), _NIE),
    (_cnt("__gfql_bindings_join__"), _v(0), _NIE),
]


def _names_rows(engine, query):
    out = _names_graph(engine).gfql(query, engine=engine)._nodes
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return out.to_dict("records")


@pytest.mark.parametrize("query,pandas_answer,polars_answer", _SPECIAL_NAMES, ids=[case[0] for case in _SPECIAL_NAMES])
def test_special_property_names_in_a_subscript(engine, query, pandas_answer, polars_answer):
    expected = polars_answer if engine in _POLARS else pandas_answer
    if expected == _NIE:
        with pytest.raises(NotImplementedError):
            _names_rows(engine, query)
        return
    try:
        rows = _names_rows(engine, query)
    except (NotImplementedError, GFQLUnsupportedError) as exc:
        assert engine == "polars-gpu" and any(known in str(exc) for known in _GPU_DECLINES), str(exc)
        return
    assert rows == expected
