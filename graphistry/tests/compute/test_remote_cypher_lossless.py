"""Remote compilation must preserve semantics or decline before upload/HTTP."""

import os
from unittest.mock import patch

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import series_to_pylist
from graphistry.compute import e_forward, let, n
from graphistry.compute.ast import ASTRef, from_json
from graphistry.compute.chain import Chain
from graphistry.compute.chain_remote import _compiled_to_let_json
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.lowering import compile_cypher_query
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.tests.compute.test_chain_remote_v2 import _mock_plottable


def _graph():
    return graphistry.nodes(
        pd.DataFrame({"id": ["u", "v", "x", "y"], "t": [1, 2, 5, 3]}), "id"
    ).edges(pd.DataFrame({"s": ["u", "x"], "d": ["v", "y"], "eid": ["k", "m"]}), "s", "d").bind(edge="eid")


@pytest.mark.parametrize("validate", [True, False])
@pytest.mark.parametrize(
    "query,field",
    [
        ("GRAPH g = GRAPH { MATCH (n) WHERE n.t > 3 OR n.t < 2 } USE g MATCH (n) RETURN n.id", "where"),
        ("GRAPH { MATCH (n) WHERE n.t > 3 OR n.t < 2 }", "where"),
        ("GRAPH g = GRAPH { MATCH (a)-[r]->(b) WHERE a.t < b.t } USE g MATCH (n) RETURN n.id", "where"),
        ("GRAPH g0 = GRAPH { MATCH (n) } GRAPH g1 = GRAPH { USE g0 MATCH (a)-[r]->(b) WHERE a.t < b.t } USE g1 MATCH (n) RETURN n.id", "where"),
        ("CALL graphistry.degree.write()", "call"),
        ("GRAPH { CALL graphistry.degree.write() }", "call"),
        ("CALL graphistry.degree()", "call"),
        ("CALL graphistry.degree() YIELD nodeId, degree RETURN nodeId, degree", "call"),
        ("GRAPH g = GRAPH { CALL graphistry.degree.write() } USE g MATCH (n) RETURN n.id", "call"),
    ],
)
def test_unrepresentable_cypher_declines_before_io(query, field, validate):
    g = _mock_plottable(dataset_id=None)
    with patch("graphistry.compute.chain_remote.requests.post") as post:
        with pytest.raises(GFQLValidationError) as exc:
            from graphistry.compute.chain_remote import chain_remote_generic

            chain_remote_generic(g, query, validate=validate)
        assert exc.value.context["field"] == field
    g.upload.assert_not_called()
    g._pygraphistry.refresh.assert_not_called()
    post.assert_not_called()


@pytest.mark.parametrize(
    "query",
    [
        "GRAPH Cand = GRAPH { MATCH (a)-[r]->(b) WHERE a.t > 2 } USE cand MATCH (a)-[r]->(b) RETURN r.eid AS eid",
        "GRAPH G0 = GRAPH { MATCH (a)-[r]->(b) } GRAPH Cand = GRAPH { USE g0 MATCH (a)-[r]->(b) WHERE a.t > 2 } USE CAND MATCH (a)-[r]->(b) RETURN r.eid AS eid",
    ],
)
def test_compiled_graph_refs_keep_declared_case(query):
    g = _graph()
    wire = _compiled_to_let_json(compile_cypher_query(parse_cypher(query)))
    out = g.gfql(from_json(wire, validate=True))
    assert series_to_pylist(out._nodes["eid"]) == series_to_pylist(g.gfql(query)._nodes["eid"]) == ["m"]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("engine", ["pandas", "cudf"])
@pytest.mark.parametrize("nested", [False, True])
def test_local_let_executes_chain_predicate(serialized, engine, nested):
    g = _graph()
    if engine == "cudf":
        if os.environ.get("TEST_CUDF") != "1":
            pytest.skip("cuDF coverage requires TEST_CUDF=1")
        cudf = pytest.importorskip("cudf")
        g = g.nodes(cudf.from_pandas(g._nodes)).edges(cudf.from_pandas(g._edges))
    query = let({"filtered": Chain(
        [n(name="a"), e_forward(name="r"), n(name="b")],
        where=[compare(col("a", "t"), "<", col("b", "t"))],
    )})
    if nested:
        query = let({"inner": query})
    if serialized:
        query = from_json(query.to_json(), validate=True)
    out = g.gfql(query, engine=engine)
    assert series_to_pylist(out._edges["eid"]) == ["k"]


def test_native_let_names_remain_case_sensitive():
    g = _graph()
    out = g.gfql(let({"Cand": n({"id": "u"}), "cand": n({"id": "v"})}), output="cand")
    assert series_to_pylist(out._nodes["id"]) == ["v"]


@pytest.mark.parametrize("serialized", [False, True])
@pytest.mark.parametrize("nested", [False, True, "ref"])
def test_remote_native_let_where_declines_before_http(serialized, nested):
    query = let({"filtered": Chain(
        [n(name="a"), e_forward(name="r"), n(name="b")],
        where=[compare(col("a", "t"), "<", col("b", "t"))],
    )})
    if nested == "ref":
        query = let({"source": n(), "filtered": ASTRef("source", [query])})
    elif nested:
        query = let({"inner": query})
    if serialized:
        query = query.to_json()
    from graphistry.compute.chain_remote import chain_remote_generic

    with patch("graphistry.compute.chain_remote.requests.post") as post:
        with pytest.raises(GFQLValidationError) as exc:
            chain_remote_generic(_mock_plottable(), query, validate=False)
    assert exc.value.context["field"] == "where"
    post.assert_not_called()
