"""An aggregate over a string subscript (count(a['name'])) answers exactly like the dotted spelling (#2363)."""
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


def _rows(engine, query):
    """Rows of the query, or None when a Polars engine declines it (parity-or-error)."""
    try:
        out = _graph(engine).gfql(query, engine=engine)._nodes
    except NotImplementedError:
        assert engine in {"polars", "polars-gpu"}, "only the Polars engines may decline"
        return None
    if hasattr(out, "to_pandas"):
        out = out.to_pandas()
    return [{key: (list(value) if hasattr(value, "tolist") and not isinstance(value, (int, float, str)) else value) for key, value in row.items()} for row in out.to_dict("records")]


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
