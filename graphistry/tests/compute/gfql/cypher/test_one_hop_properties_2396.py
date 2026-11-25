"""properties()/keys() of one-hop aliases render only that alias's own fields (#2396)."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLUnsupportedError


def _graph(engine, edge_id=None):
    nodes = pd.DataFrame({"id": [1, 2], "name": ["x", "y"]})
    edges = pd.DataFrame({"s": [1], "d": [2], "kind": ["k1"]})
    if edge_id is not None:
        edges[edge_id] = [7]
    if engine == "cudf":
        import cudf

        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d", edge_id)


def _records(frame):
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
    ("MATCH (n)-[e]->(m) RETURN properties(e) AS p", {"p": "{kind: 'k1'}"}),
    (
        "MATCH (n)-[e]->(m) RETURN properties(n) AS a, properties(e) AS p, properties(m) AS b",
        {"a": "{name: 'x'}", "p": "{kind: 'k1'}", "b": "{name: 'y'}"},
    ),
    ("MATCH (n)-[e]->(m) RETURN keys(e) AS k", {"k": ["kind"]}),
    ("MATCH (n)-[e]->(m) WITH e RETURN properties(e) AS p", {"p": "{kind: 'k1'}"}),
    ("MATCH ()-[e]->() RETURN properties(e) AS p", {"p": "{kind: 'k1'}"}),
])
def test_one_hop_alias_properties_are_alias_scoped(engine, query, expected):
    graph = _graph(engine)
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(query, engine=engine)
        return
    assert _records(graph.gfql(query, engine=engine)._nodes) == [expected]


def test_one_hop_edge_properties_match_anonymous_form_with_edge_id(engine):
    graph = _graph(engine, edge_id="eid")
    one_hop = "MATCH (n)-[e]->(m) RETURN properties(e) AS p"
    anonymous = "MATCH ()-[e]->() RETURN properties(e) AS p"
    if engine in {"polars", "polars-gpu"}:
        with pytest.raises((NotImplementedError, GFQLUnsupportedError)):
            graph.gfql(one_hop, engine=engine)
        return
    rows = _records(graph.gfql(one_hop, engine=engine)._nodes)
    assert rows == _records(graph.gfql(anonymous, engine=engine)._nodes)
    assert rows == [{"p": "{kind: 'k1', eid: 7}"}]
