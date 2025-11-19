import pandas as pd
import pytest
from hypothesis import given, settings, strategies as st
from typing import Set
from types import SimpleNamespace

from graphistry.compute import n, e_forward, e_undirected
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.gfql.same_path_types import col, compare


def _plottable(nodes, edges):
    return SimpleNamespace(
        _nodes=nodes,
        _edges=edges,
        _node="id",
        _source="src",
        _destination="dst",
        _edge="edge_id",
    )


def _col_set(df: pd.DataFrame, column: str) -> Set[str]:
    return set(df[column]) if column in df.columns else set()


CASES = [
    {
        "id": "simple",
        "nodes": [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ],
        "edges": [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "user1", "type": "owns"},
        ],
        "ops": [n({"type": "account"}, name="a"), e_forward({"type": "txn"}),
        n(name="b")],
        "expect": {"nodes": {"acct1", "acct2"}, "edges": {"e1"}},
    },
    {
        "id": "where",
        "nodes": [
            {"id": "acct_good", "type": "account", "owner_id": "user1"},
            {"id": "acct_bad", "type": "account", "owner_id": "user2"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
        ],
        "edges": [
            {"edge_id": "e_good", "src": "acct_good", "dst": "user1", "type": "owns"},
            {"edge_id": "e_bad_match", "src": "acct_bad", "dst": "user2", "type":
            "owns"},
            {"edge_id": "e_bad_wrong", "src": "acct_bad", "dst": "user1", "type":
            "owns"},
        ],
        "ops": [
            n({"type": "account"}, name="a"),
            e_forward({"type": "owns"}, name="r"),
            n({"type": "user"}, name="c"),
        ],
        "where": [compare(col("a", "owner_id"), "==", col("c", "id"))],
        "include_paths": True,
        "expect": {
            "nodes": {"acct_good", "acct_bad", "user1", "user2"},
            "edges": {"e_good", "e_bad_match"},
            "tags": {"a": {"acct_good", "acct_bad"}, "r": {"e_good", "e_bad_match"},
            "c": {"user1", "user2"}},
            "paths": [
                {"a": "acct_good", "c": "user1", "r": "e_good"},
                {"a": "acct_bad", "c": "user2", "r": "e_bad_match"},
            ],
        },
    },
    {
        "id": "null",
        "nodes": [
            {"id": "acct1", "type": "account", "owner_id": None},
            {"id": "user1", "type": "user"},
        ],
        "edges": [{"edge_id": "e1", "src": "acct1", "dst": "user1"}],
        "ops": [n(name="a"), e_forward(), n(name="c")],
        "where": [compare(col("a", "owner_id"), "==", col("c", "id"))],
        "expect": {"nodes": set(), "edges": set()},
    },
    {
        "id": "alias",
        "nodes": [
            {"id": "acct1", "type": "account", "owner_id": "u1"},
            {"id": "acct2", "type": "account", "owner_id": "u2"},
            {"id": "acct3", "type": "account", "owner_id": "u1"},
            {"id": "u1", "type": "user"},
            {"id": "u2", "type": "user"},
        ],
        "edges": [
            {"edge_id": "t1", "src": "acct1", "dst": "acct2", "type": "transfer"},
            {"edge_id": "t2", "src": "acct3", "dst": "acct1", "type": "transfer"},
            {"edge_id": "o1", "src": "acct2", "dst": "u2", "type": "owns"},
            {"edge_id": "o2", "src": "acct1", "dst": "u1", "type": "owns"},
        ],
        "ops": [
            n({"type": "account"}, name="a"),
            e_forward({"type": "transfer"}, name="t"),
            n({"type": "account"}, name="b"),
            e_forward({"type": "owns"}, name="o"),
            n({"type": "user"}, name="c"),
        ],
        "where": [
            compare(col("a", "id"), "!=", col("b", "id")),
            compare(col("a", "owner_id"), "==", col("c", "id")),
        ],
        "include_paths": True,
        "expect": {
            "nodes": {"acct3", "acct1", "u1"},
            "edges": {"t2", "o2"},
            "paths": [{"a": "acct3", "b": "acct1", "c": "u1", "o": "o2", "t": "t2"}],
        },
    },
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case["id"])
def test_enumerator_scenarios(case):
    g = _plottable(pd.DataFrame(case["nodes"]), pd.DataFrame(case["edges"]))
    result = enumerate_chain(
        g,
        case["ops"],
        where=case.get("where"),
        include_paths=case.get("include_paths", False),
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    expect = case["expect"]
    if "nodes" in expect:
        assert _col_set(result.nodes, "id") == expect["nodes"]
    if "edges" in expect:
        assert _col_set(result.edges, "edge_id") == expect["edges"]
    if "paths" in expect:
        assert sorted(result.paths or [], key=str) == sorted(expect["paths"], key=str)
    if "tags" in expect:
        assert result.tags == expect["tags"]


def test_enumerator_cap_enforcement():
    nodes = pd.DataFrame({"id": [f"n{i}" for i in range(15)]})
    edges = pd.DataFrame({"edge_id": ["e1"], "src": ["n0"], "dst": ["n1"]})
    g = _plottable(nodes, edges)
    with pytest.raises(ValueError):
        enumerate_chain(g, [n(name="a")], caps=OracleCaps(max_nodes=10, max_edges=10))


def test_enumerator_supports_to_pandas_frames():
    class Dummy:
        def __init__(self, df):
            self._df = df

        def to_pandas(self):
            return self._df.copy()

    g = _plottable(Dummy(pd.DataFrame([{"id": "n1"}])), Dummy(pd.DataFrame([{"edge_id":
    "e1", "src": "n1", "dst": "n1"}])))
    result = enumerate_chain(g, [n(name="a")], caps=OracleCaps(max_nodes=20,
    max_edges=20))
    assert _col_set(result.nodes, "id") == {"n1"}


def test_where_type_mismatch_inequality_is_false():
    nodes = pd.DataFrame([{"id": "a", "x": "foo"}, {"id": "c", "x": 10}])
    edges = pd.DataFrame([{"edge_id": "e", "src": "a", "dst": "c"}])
    g = _plottable(nodes, edges)
    result = enumerate_chain(
        g,
        [n(name="a"), e_forward(name="r"), n(name="c")],
        where=[compare(col("a", "x"), "<", col("c", "x"))],
        caps=OracleCaps(max_nodes=10, max_edges=10),
    )
    assert result.nodes.empty and result.edges.empty


def test_undirected_self_loop_no_double():
    nodes = pd.DataFrame([{"id": "u"}])
    edges = pd.DataFrame([{"edge_id": "loop", "src": "u", "dst": "u"}])
    g = _plottable(nodes, edges)
    result = enumerate_chain(
        g,
        [n(name="a"), e_undirected(name="r"), n(name="c")],
        include_paths=True,
        caps=OracleCaps(max_nodes=10, max_edges=10),
    )
    assert not result.paths or len(result.paths) == 1


def test_paths_are_deterministically_sorted():
    nodes = pd.DataFrame([{"id": "n1"}, {"id": "n2"}, {"id": "n3"}])
    edges = pd.DataFrame(
        [
            {"edge_id": "e1", "src": "n1", "dst": "n2"},
            {"edge_id": "e2", "src": "n1", "dst": "n3"},
        ]
    )
    g = _plottable(nodes, edges)
    result = enumerate_chain(
        g,
        [n(name="a"), e_forward(name="r"), n(name="c")],
        include_paths=True,
        caps=OracleCaps(max_nodes=10, max_edges=10),
    )
    bindings = result.paths or []
    tuples = [tuple(binding.get(k) for k in sorted(binding)) for binding in bindings]
    assert tuples == sorted(tuples)


def test_enumerator_min_max_three_branch_unlabeled():
    nodes = pd.DataFrame(
        [
            {"id": "a"},
            {"id": "b1"},
            {"id": "c1"},
            {"id": "d1"},
            {"id": "e1"},
            {"id": "b2"},
            {"id": "c2"},
        ]
    )
    edges = pd.DataFrame(
        [
            {"edge_id": "e1", "src": "a", "dst": "b1"},
            {"edge_id": "e2", "src": "b1", "dst": "c1"},
            {"edge_id": "e3", "src": "c1", "dst": "d1"},
            {"edge_id": "e4", "src": "d1", "dst": "e1"},
            {"edge_id": "e5", "src": "a", "dst": "b2"},
            {"edge_id": "e6", "src": "b2", "dst": "c2"},
        ]
    )
    g = _plottable(nodes, edges)
    result = enumerate_chain(
        g,
        [n({"id": "a"}), e_forward(min_hops=3, max_hops=3), n()],
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert _col_set(result.nodes, "id") == {"a", "b1", "c1", "d1"}
    assert _col_set(result.edges, "edge_id") == {"e1", "e2", "e3"}


NODE_POOL = [f"n{i}" for i in range(6)]
EDGE_POOL = [f"e{i}" for i in range(8)]


@st.composite
def small_graph_cases(draw):
    nodes = draw(st.lists(st.sampled_from(NODE_POOL), min_size=2, max_size=4,
    unique=True))
    node_rows = [{"id": node, "value": draw(st.integers(0, 3))} for node in nodes]
    edges = draw(st.lists(st.tuples(st.sampled_from(nodes), st.sampled_from(nodes)),
    min_size=1, max_size=5))
    edge_rows = [
        {"edge_id": EDGE_POOL[i % len(EDGE_POOL)], "src": src, "dst": dst}
        for i, (src, dst) in enumerate(edges)
    ]
    where = []
    if draw(st.booleans()):
        where = [
            compare(
                col("a", "value"),
                draw(st.sampled_from(["==", "!=", "<", "<=", ">", ">="])),
                col("c", "value"),
            )
        ]
    return {
        "nodes": pd.DataFrame(node_rows),
        "edges": pd.DataFrame(edge_rows),
        "where": where,
    }


@given(small_graph_cases())
@settings(deadline=None, max_examples=25)
def test_enumerator_paths_cover_outputs(case):
    g = _plottable(case["nodes"], case["edges"])
    result = enumerate_chain(
        g,
        [n(name="a"), e_forward(name="rel"), n(name="c")],
        where=case["where"],
        include_paths=True,
        caps=OracleCaps(max_nodes=10, max_edges=10, max_length=4,
        max_partial_rows=10_000),
    )

    path_nodes = {
        binding[alias]
        for binding in result.paths or []
        for alias in ("a", "c")
        if alias in binding
    }
    path_edges = {binding["rel"] for binding in result.paths or [] if "rel" in binding}

    assert _col_set(result.nodes, "id") <= path_nodes
    if not result.edges.empty:
        assert set(result.edges["edge_id"]) <= path_edges
