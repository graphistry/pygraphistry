import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from types import SimpleNamespace

from graphistry.compute import n, e_forward
from graphistry.gfql.ref.enumerator import (
    OracleCaps,
    col,
    compare,
    enumerate_chain,
)


def make_plottable(nodes_df, edges_df):
    return SimpleNamespace(
        _nodes=nodes_df,
        _edges=edges_df,
        _node="id",
        _source="src",
        _destination="dst",
        _edge="edge_id",
    )


def test_enumerator_simple_chain():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "user1", "type": "owns"},
        ]
    )

    g = make_plottable(nodes, edges)
    ops = [n({"type": "account"}, name="a"), e_forward({"type": "txn"}), n(name="b")]

    result = enumerate_chain(g, ops, caps=OracleCaps(max_nodes=20, max_edges=20))
    assert set(result.nodes["id"]) == {"acct1", "acct2"}
    assert set(result.edges["edge_id"]) == {"e1"}


def test_enumerator_same_path_where():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1"},
            {"id": "acct2", "type": "account", "owner_id": "user2"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"edge_id": "e1", "src": "acct1", "dst": "user1", "type": "owns"},
            {"edge_id": "e2", "src": "acct2", "dst": "user1", "type": "owns"},
        ]
    )
    g = make_plottable(nodes, edges)
    ops = [n({"type": "account"}, name="a"), e_forward({"type": "owns"}), n({"type": "user"}, name="c")]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

    result = enumerate_chain(
        g, ops, where=where, caps=OracleCaps(max_nodes=20, max_edges=20), include_paths=True
    )
    assert set(result.nodes["id"]) == {"acct1", "user1"}
    assert set(result.edges["edge_id"]) == {"e1"}
    assert result.paths == [{"a": "acct1", "c": "user1"}]


def test_enumerator_null_semantics():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": None},
            {"id": "user1", "type": "user"},
        ]
    )
    edges = pd.DataFrame([{"edge_id": "e1", "src": "acct1", "dst": "user1"}])
    g = make_plottable(nodes, edges)
    ops = [n(name="a"), e_forward(), n(name="c")]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

    result = enumerate_chain(g, ops, where=where, caps=OracleCaps(max_nodes=20, max_edges=20))
    assert result.nodes.empty
    assert result.edges.empty


def test_enumerator_triangle_where_filters_bad_paths():
    nodes = pd.DataFrame(
        [
            {"id": "acct_good", "type": "account", "owner_id": "user1"},
            {"id": "acct_bad", "type": "account", "owner_id": "user2"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"edge_id": "e_good", "src": "acct_good", "dst": "user1", "type": "owns"},
            {"edge_id": "e_bad_match", "src": "acct_bad", "dst": "user2", "type": "owns"},
            {"edge_id": "e_bad_wrong", "src": "acct_bad", "dst": "user1", "type": "owns"},
        ]
    )
    g = make_plottable(nodes, edges)
    ops = [
        n({"type": "account"}, name="a"),
        e_forward({"type": "owns"}, name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

    result = enumerate_chain(
        g, ops, where=where, include_paths=True, caps=OracleCaps(max_nodes=20, max_edges=20)
    )

    assert set(result.nodes["id"]) == {"acct_good", "acct_bad", "user1", "user2"}
    assert set(result.edges["edge_id"]) == {"e_good", "e_bad_match"}
    assert result.tags["a"] == {"acct_good", "acct_bad"}
    assert result.tags["c"] == {"user1", "user2"}
    assert result.tags["r"] == {"e_good", "e_bad_match"}
    assert result.paths == [
        {"a": "acct_good", "c": "user1", "r": "e_good"},
        {"a": "acct_bad", "c": "user2", "r": "e_bad_match"},
    ]


def test_enumerator_alias_reuse_with_inequality():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "u1"},
            {"id": "acct2", "type": "account", "owner_id": "u2"},
            {"id": "acct3", "type": "account", "owner_id": "u1"},
            {"id": "u1", "type": "user"},
            {"id": "u2", "type": "user"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"edge_id": "t1", "src": "acct1", "dst": "acct2", "type": "transfer"},
            {"edge_id": "t2", "src": "acct3", "dst": "acct1", "type": "transfer"},
            {"edge_id": "o1", "src": "acct2", "dst": "u2", "type": "owns"},
            {"edge_id": "o2", "src": "acct1", "dst": "u1", "type": "owns"},
        ]
    )
    g = make_plottable(nodes, edges)
    ops = [
        n({"type": "account"}, name="a"),
        e_forward({"type": "transfer"}, name="t"),
        n({"type": "account"}, name="b"),
        e_forward({"type": "owns"}, name="o"),
        n({"type": "user"}, name="c"),
    ]
    where = [
        compare(col("a", "id"), "!=", col("b", "id")),
        compare(col("a", "owner_id"), "==", col("c", "id")),
    ]

    result = enumerate_chain(
        g, ops, where=where, include_paths=True, caps=OracleCaps(max_nodes=20, max_edges=20)
    )

    assert set(result.nodes["id"]) == {"acct3", "acct1", "u1"}
    assert set(result.edges["edge_id"]) == {"t2", "o2"}
    assert result.paths == [{"a": "acct3", "b": "acct1", "c": "u1", "o": "o2", "t": "t2"}]


def test_enumerator_cap_enforcement():
    nodes = pd.DataFrame([{"id": f"n{i}"} for i in range(15)])
    edges = pd.DataFrame([{"edge_id": f"e{i}", "src": "n0", "dst": "n1"} for i in range(2)])
    g = make_plottable(nodes, edges)
    ops = [n(name="a")]
    with pytest.raises(ValueError):
        enumerate_chain(g, ops, caps=OracleCaps(max_nodes=10, max_edges=10))


def test_enumerator_supports_to_pandas_frames():
    class DummyFrame:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df.copy()

    nodes = DummyFrame(pd.DataFrame([{"id": "n1", "type": "user"}]))
    edges = DummyFrame(pd.DataFrame([{"edge_id": "e1", "src": "n1", "dst": "n1"}]))
    g = make_plottable(nodes, edges)
    ops = [n(name="a")]
    result = enumerate_chain(g, ops, caps=OracleCaps(max_nodes=20, max_edges=20))
    assert set(result.nodes["id"]) == {"n1"}


NODE_NAMES = [f"n{i}" for i in range(6)]
EDGE_NAMES = [f"e{i}" for i in range(8)]


@st.composite
def small_graph_cases(draw):
    node_ids = draw(st.lists(st.sampled_from(NODE_NAMES), min_size=2, max_size=4, unique=True))
    nodes = []
    for node_id in node_ids:
        nodes.append({"id": node_id, "value": draw(st.integers(0, 3))})

    edge_ids = draw(
        st.lists(st.sampled_from(EDGE_NAMES), min_size=1, max_size=5, unique=True)
    )
    edges = []
    for edge_id in edge_ids:
        src = draw(st.sampled_from(node_ids))
        dst = draw(st.sampled_from(node_ids))
        edges.append({"edge_id": edge_id, "src": src, "dst": dst})

    where_enabled = draw(st.booleans())
    where_clauses = []
    if where_enabled:
        op = draw(st.sampled_from(["==", "!="]))
        where_clauses = [compare(col("a", "value"), op, col("c", "value"))]

    return {
        "nodes": pd.DataFrame(nodes),
        "edges": pd.DataFrame(edges),
        "where": where_clauses,
    }


@given(small_graph_cases())
@settings(deadline=None, max_examples=50)
def test_enumerator_paths_cover_outputs(case):
    g = make_plottable(case["nodes"], case["edges"])
    ops = [
        n(name="a"),
        e_forward(name="rel"),
        n(name="c"),
    ]
    result = enumerate_chain(
        g,
        ops,
        where=case["where"],
        include_paths=True,
        caps=OracleCaps(max_nodes=10, max_edges=10, max_length=4, max_partial_rows=10_000),
    )

    path_nodes = set()
    path_edges = set()
    for binding in result.paths or []:
        for alias in ("a", "c"):
            if alias in binding:
                path_nodes.add(binding[alias])
        if "rel" in binding:
            path_edges.add(binding["rel"])

    assert set(result.nodes["id"]) <= path_nodes
    if not result.edges.empty:
        assert set(result.edges["edge_id"]) <= path_edges
