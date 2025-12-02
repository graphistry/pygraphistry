import pandas as pd
import pytest

from graphistry.compute import e_forward, e_reverse, e_undirected, n
from graphistry.compute.ast import ASTEdge, ASTNode
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull


def _to_pandas(df):
    if df is None:
        return None
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _alias_bindings(df, id_col, alias):
    if df is None or alias not in df.columns:
        return set()
    return set(df.loc[df[alias].astype(bool), id_col])


def _run_parity_case(nodes, edges, ops):
    g = (
        CGFull()
        .nodes(pd.DataFrame(nodes), "id")
        .edges(pd.DataFrame(edges), "src", "dst", edge="edge_id")
    )
    gfql_result = g.gfql(ops)
    oracle = enumerate_chain(g, ops, caps=OracleCaps(max_nodes=50, max_edges=50))

    gfql_nodes = _to_pandas(gfql_result._nodes)
    gfql_edges = _to_pandas(gfql_result._edges)

    assert gfql_nodes is not None
    assert set(gfql_nodes[g._node]) == set(oracle.nodes[g._node])

    if g._edge is not None and gfql_edges is not None and not gfql_edges.empty:
        assert set(gfql_edges[g._edge]) == set(oracle.edges[g._edge])
    else:
        assert oracle.edges.empty

    for op in ops:
        alias = getattr(op, "_name", None)
        if not alias:
            continue
        if isinstance(op, ASTNode):
            assert oracle.tags.get(alias, set()) == _alias_bindings(gfql_nodes, g._node, alias)
        elif isinstance(op, ASTEdge):
            assert oracle.tags.get(alias, set()) == _alias_bindings(gfql_edges, g._edge, alias)


CASES = [
    (
        "forward",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "acct3", "type": "txn"},
            {"edge_id": "e3", "src": "acct3", "dst": "acct1", "type": "txn"},
        ],
        [n({"type": "account"}, name="start"), e_forward({"type": "txn"}, name="hop"), n({"type": "account"}, name="end")],
    ),
    (
        "reverse",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ],
        [
            {"edge_id": "owns1", "src": "acct1", "dst": "user1", "type": "owns"},
            {"edge_id": "owns2", "src": "acct2", "dst": "user1", "type": "owns"},
        ],
        [n({"type": "user"}, name="u"), e_reverse({"type": "owns"}, name="owns_rev"), n({"type": "account"}, name="acct")],
    ),
    (
        "two_hop",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
        ],
        [
            {"edge_id": "txn1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "owns1", "src": "acct2", "dst": "user1", "type": "owns"},
            {"edge_id": "owns2", "src": "acct2", "dst": "user2", "type": "owns"},
        ],
        [
            n({"type": "account"}, name="acct_start"),
            e_forward({"type": "txn"}, name="txn"),
            n({"type": "account"}, name="acct_mid"),
            e_forward({"type": "owns"}, name="owns"),
            n({"type": "user"}, name="user_end"),
        ],
    ),
    (
        "undirected",
        [
            {"id": "n1", "type": "node"},
            {"id": "n2", "type": "node"},
            {"id": "n3", "type": "node"},
        ],
        [
            {"edge_id": "e12", "src": "n1", "dst": "n2", "type": "path"},
            {"edge_id": "e23", "src": "n2", "dst": "n3", "type": "path"},
        ],
        [n({"type": "node"}, name="start"), e_undirected({"type": "path"}, name="hop"), n({"type": "node"}, name="end")],
    ),
    (
        "empty",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
        ],
        [{"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"}],
        [n({"type": "user"}, name="start"), e_forward({"type": "txn"}, name="hop"), n({"type": "user"}, name="end")],
    ),
    (
        "cycle",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
        ],
        [
            {"edge_id": "e12", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e21", "src": "acct2", "dst": "acct1", "type": "txn"},
        ],
        [
            n({"type": "account"}, name="start"),
            e_forward({"type": "txn"}, name="hop1"),
            n({"type": "account"}, name="mid"),
            e_forward({"type": "txn"}, name="hop2"),
            n({"type": "account"}, name="end"),
        ],
    ),
    (
        "branch",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
            {"id": "acct4", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct1", "dst": "acct3", "type": "txn"},
            {"edge_id": "e3", "src": "acct3", "dst": "acct4", "type": "txn"},
        ],
        [n({"type": "account"}, name="root"), e_forward({"type": "txn"}, name="first_hop"), n({"type": "account"}, name="child")],
    ),
    (
        "forward_labels",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "acct3", "type": "account"},
        ],
        [
            {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "e2", "src": "acct2", "dst": "acct3", "type": "txn"},
        ],
        [
            n({"type": "account"}, name="start"),
            e_forward(
                {"type": "txn"},
                name="hop",
                label_node_hops="node_hop",
                label_edge_hops="edge_hop",
                label_seeds=True,
            ),
            n({"type": "account"}, name="end"),
        ],
    ),
    (
        "reverse_two_hop",
        [
            {"id": "acct1", "type": "account"},
            {"id": "acct2", "type": "account"},
            {"id": "user1", "type": "user"},
        ],
        [
            {"edge_id": "txn1", "src": "acct1", "dst": "acct2", "type": "txn"},
            {"edge_id": "owns1", "src": "acct2", "dst": "user1", "type": "owns"},
        ],
        [
            n({"type": "user"}, name="user_end"),
            e_reverse({"type": "owns"}, name="owns_rev"),
            n({"type": "account"}, name="acct_mid"),
            e_reverse({"type": "txn"}, name="txn_rev"),
            n({"type": "account"}, name="acct_start"),
        ],
    ),
]


@pytest.mark.parametrize("_, nodes, edges, ops", CASES, ids=[case[0] for case in CASES])
def test_enumerator_matches_gfql(_, nodes, edges, ops):
    _run_parity_case(nodes, edges, ops)


def test_enumerator_min_max_three_branch_unlabeled():
    nodes = [
        {"id": "a"},
        {"id": "b1"},
        {"id": "c1"},
        {"id": "d1"},
        {"id": "e1"},
        {"id": "b2"},
        {"id": "c2"},
    ]
    edges = [
        {"edge_id": "e1", "src": "a", "dst": "b1"},
        {"edge_id": "e2", "src": "b1", "dst": "c1"},
        {"edge_id": "e3", "src": "c1", "dst": "d1"},
        {"edge_id": "e4", "src": "d1", "dst": "e1"},
        {"edge_id": "e5", "src": "a", "dst": "b2"},
        {"edge_id": "e6", "src": "b2", "dst": "c2"},
    ]
    ops = [
        n({"id": "a"}),
        e_forward(min_hops=3, max_hops=3),
        n(),
    ]
    _run_parity_case(nodes, edges, ops)
