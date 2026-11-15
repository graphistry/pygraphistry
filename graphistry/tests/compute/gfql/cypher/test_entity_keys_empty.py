"""Ordinary entity keys preserve empty lists across native nullable rows."""
import os

import pandas as pd
import pytest

import graphistry


@pytest.mark.parametrize("engine", [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="needs TEST_CUDF=1")),
])
@pytest.mark.parametrize("table", ["nodes", "edges"])
@pytest.mark.parametrize("values,expected", [
    ([None, "x"], [[], ["p"]]),
    ([None, None], [[], []]),
    (["x", "y"], [["p"], ["p"]]),
])
def test_entity_keys_with_empty_property_rows(engine, table, values, expected):
    nodes = pd.DataFrame({"id": ["a", "b"], "p": values})
    edges = pd.DataFrame({"src": ["a", "b"], "dst": ["b", "a"], "edge_id": [0, 1], "type": ["R", "R"], "p": values})
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
    before = nodes.to_arrow().to_pylist() if engine == "cudf" else nodes.to_dict("records")
    source = graphistry.nodes(nodes, "id")
    if table == "edges":
        source = source.edges(edges, "src", "dst", edge="edge_id")
        query = "MATCH ()-[r]->() RETURN r.edge_id AS identity, keys(r) AS k"
        identities = [0, 1]
    else:
        query = "MATCH (n) RETURN n.id AS identity, keys(n) AS k"
        identities = ["a", "b"]
    edges_before = edges.to_arrow().to_pylist() if engine == "cudf" else edges.to_dict("records")
    result = source.gfql(query, engine=engine)
    rows = result._nodes.to_arrow().to_pylist() if engine == "cudf" else result._nodes.to_dict("records")
    assert sorted(rows, key=lambda row: row["identity"]) == [
        {"identity": identity, "k": keys} for identity, keys in zip(identities, expected)
    ]
    assert type(result._nodes).__module__.split(".")[0] == engine
    after = nodes.to_arrow().to_pylist() if engine == "cudf" else nodes.to_dict("records")
    assert after == before
    assert source._nodes is nodes and source._node == "id"
    edges_after = edges.to_arrow().to_pylist() if engine == "cudf" else edges.to_dict("records")
    assert edges_after == edges_before
    if table == "edges":
        assert source._edges is edges and source._edge == "edge_id"
