"""AUTO resolves same-path WHERE execution from the cuDF graph, like explicit cuDF."""

import os

import pandas as pd
import pytest

import graphistry
from graphistry.Engine import EngineAbstract, series_to_pylist
from graphistry.compute import e_forward, n
from graphistry.compute.gfql.same_path_types import col, compare


pytestmark = pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="requires TEST_CUDF=1")


@pytest.mark.parametrize("engine", [None, "auto", EngineAbstract.AUTO, "cudf"])
@pytest.mark.parametrize("query_form", ["cypher", "chain"])
@pytest.mark.parametrize("matched", [True, False])
def test_auto_cross_alias_where_keeps_cudf(engine, query_form, matched):
    cudf = pytest.importorskip("cudf")
    nodes = cudf.from_pandas(pd.DataFrame({"id": ["u", "v"], "t": [1, 2] if matched else [2, 1]}))
    edges = cudf.DataFrame({"s": ["u"], "d": ["v"], "eid": ["k"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d").bind(edge="eid")
    kwargs = {} if engine is None else {"engine": engine}
    if query_form == "cypher":
        out = g.gfql("MATCH (a)-[r]->(b) WHERE a.t < b.t RETURN r.eid AS eid", **kwargs)
        result_frame = out._nodes
    else:
        out = g.gfql(
            [n(name="a"), e_forward(name="r"), n(name="b")],
            where=[compare(col("a", "t"), "<", col("b", "t"))],
            **kwargs,
        )
        result_frame = out._edges
    assert isinstance(result_frame, cudf.DataFrame)
    assert series_to_pylist(result_frame["eid"]) == (["k"] if matched else [])
    assert nodes["t"].to_arrow().to_pylist() == ([1, 2] if matched else [2, 1])
