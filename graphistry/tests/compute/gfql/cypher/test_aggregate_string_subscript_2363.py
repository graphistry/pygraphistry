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
    """Rows of the query. The one decline today: polars-gpu on an executor without GPU group-by (cudf-polars 25.02)
    raises GFQLUnsupportedError; that error, for that engine, is returned as None and nothing else is tolerated."""
    try:
        return _run(engine, query)
    except GFQLUnsupportedError as exc:
        assert engine == "polars-gpu" and "cannot execute this aggregation" in str(exc), str(exc)
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


# The subscript spelling of a non-entity value (a map, a list, a scalar, null, an UNWIND variable) must answer exactly as
# it did before the parser rewrote an entity subscript to a property read. Outcomes below were measured on `develop`:
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
    # Two outcomes that DO differ from `develop` (both toward the Cypher answer and toward pandas/cuDF agreement):
    ("WITH [1, 2] AS l RETURN count(l['name']) AS v", _ERR, None),  # develop: pandas answered 0, cuDF raised
    ("WITH null AS z RETURN count(z['name']) AS v", _ROWS, [{"v": 0}]),  # develop: pandas answered 0, cuDF raised
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
