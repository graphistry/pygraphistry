import pandas as pd

import graphistry
from graphistry import let, query, ref


def _mk_graph():
    nodes = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "kind": ["person", "person", "company"],
        }
    )
    edges = pd.DataFrame(
        {
            "s": ["a", "b"],
            "d": ["b", "c"],
        }
    )
    return graphistry.edges(edges, "s", "d").nodes(nodes, "id")


def test_top_level_query_executes_locally():
    result = _mk_graph().gfql(query("MATCH (n {id: 'b'}) RETURN n.id AS id"))

    assert result._nodes is not None
    assert sorted(result._nodes["id"].tolist()) == ["b"]
    assert result._edges is not None
    assert len(result._edges) == 0


def test_top_level_query_supports_params():
    result = _mk_graph().gfql(
        query("MATCH (n {id: $node_id}) RETURN n.id AS id", params={"node_id": "c"})
    )

    assert result._nodes is not None
    assert sorted(result._nodes["id"].tolist()) == ["c"]


def test_ref_query_executes_against_binding():
    workflow = let(
        {
            "path": query("MATCH (n {id: 'b'}) RETURN n.id AS id"),
            "names": ref(
                "path",
                query("MATCH (n) RETURN n.id AS id ORDER BY id"),
            ),
        }
    )

    result = _mk_graph().gfql(workflow, output="names")

    assert result._nodes is not None
    assert result._nodes.to_dict("records") == [{"id": "b"}]
    assert result._edges is not None
    assert len(result._edges) == 0
