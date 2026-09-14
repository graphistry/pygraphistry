"""A Cypher variable named like a frame column is served through every lane on every engine.

The native chain's alias/column collision class (#2039) has a Cypher twin: a MATCH variable
whose name equals a property the pattern filters on, an edge column, or the node binding.
Pins: the seeded typed-hop and node-lookup lanes, the whole-entity return, the two-hop
count lane and the lowered full path all return the pandas full-path rows on pandas, cuDF
and polars, with and without resident indexes; the native rows route with a colliding alias
does too.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n, rows

NODES = pd.DataFrame({"key": [1, 2, 3, 4], "id": [10, 20, 30, 40], "type": ["p", "p", "m", "m"], "w": [1, 2, 3, 4]})
EDGES = pd.DataFrame({"s": [3, 3, 4, 1], "d": [1, 2, 1, 4], "type": ["HAS_CREATOR", "OTHER", "HAS_CREATOR", "OTHER"], "eid": [100, 101, 102, 103], "w": [5, 6, 7, 8]})
ENGINES = ["pandas", "cudf", "polars"]


def _graph(engine, indexed):
    nodes, edges = NODES, EDGES
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid")
    return g.gfql_index_all(engine=engine).gfql_index_node_props(["id"], engine=engine) if indexed else g


def _rows(res):
    df = res._nodes if hasattr(res, "_nodes") and not hasattr(res, "columns") else res
    df = df.to_pandas() if hasattr(df, "to_pandas") else df
    df = df.reindex(sorted(df.columns), axis=1)

    def canon(v):
        try:
            f = float(v)
            return "nan" if f != f else f
        except (TypeError, ValueError):
            return str(v)
    return sorted(tuple(canon(v) for v in r) for r in df.values.tolist())


QUERIES = {
    "edge variable = filtered edge column": "MATCH (m {id: 30})-[type:HAS_CREATOR]->(p) RETURN p.id AS pid",
    "edge variable = filtered column, projected": "MATCH (m {id: 30})-[type:HAS_CREATOR]->(p) RETURN p.id AS pid, type.w AS w",
    "seed variable = its filtered property": "MATCH (id {id: 30})-[:HAS_CREATOR]->(p) RETURN p.id AS pid",
    "destination variable = its filtered property": "MATCH (m {id: 30})-[:HAS_CREATOR]->(type {type: 'p'}) RETURN type.id AS pid",
    "node-only variable = its filtered property": "MATCH (id {id: 30}) RETURN id.key AS k",
    "whole entity with a colliding edge variable": "MATCH (m {id: 30})-[type:HAS_CREATOR]->(p) RETURN p",
    "two-hop count with a colliding edge variable": "MATCH (a)-[type:HAS_CREATOR]->(b)-[e2]->(c) RETURN count(*) AS c",
}


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
@pytest.mark.parametrize("name", list(QUERIES))
def test_colliding_cypher_variables_match_the_pandas_full_path(engine, indexed, name):
    q = QUERIES[name]
    oracle = _rows(_graph("pandas", False).gfql(q, engine="pandas", index_policy="off"))
    got = _rows(_graph(engine, indexed).gfql(q, engine=engine, index_policy="use" if indexed else "off"))
    assert got == oracle


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("indexed", [False, True], ids=["scan", "indexed"])
def test_rows_route_with_a_colliding_alias_matches_the_pandas_full_path(engine, indexed):
    ops = [n({"id": 30}, name="m"), e_forward({"type": "HAS_CREATOR"}, name="type"), n(name="p"), rows(source="p")]
    oracle = _rows(_graph("pandas", False).gfql(ops, engine="pandas", index_policy="off"))
    got = _rows(_graph(engine, indexed).gfql(ops, engine=engine, index_policy="use" if indexed else "off"))
    assert got == oracle
