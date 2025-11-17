import pandas as pd

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
    mask = df[alias].astype(bool)
    return set(df.loc[mask, id_col].tolist())


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

    # Compare alias bindings via boolean columns
    for op in ops:
        alias = getattr(op, "_name", None)
        if not alias:
            continue
        if isinstance(op, ASTNode):
            assert oracle.tags.get(alias, set()) == _alias_bindings(gfql_nodes, g._node, alias)
        elif isinstance(op, ASTEdge):
            assert oracle.tags.get(alias, set()) == _alias_bindings(gfql_edges, g._edge, alias)


def test_enumerator_matches_gfql_simple_chain():
    nodes = [
        {"id": "acct1", "type": "account"},
        {"id": "acct2", "type": "account"},
        {"id": "acct3", "type": "account"},
    ]
    edges = [
        {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
        {"edge_id": "e2", "src": "acct2", "dst": "acct3", "type": "txn"},
        {"edge_id": "e3", "src": "acct3", "dst": "acct1", "type": "txn"},
    ]
    ops = [
        n({"type": "account"}, name="start"),
        e_forward({"type": "txn"}, name="hop"),
        n({"type": "account"}, name="end"),
    ]
    _run_parity_case(nodes, edges, ops)


def test_enumerator_matches_gfql_reverse_edges():
    nodes = [
        {"id": "acct1", "type": "account"},
        {"id": "acct2", "type": "account"},
        {"id": "user1", "type": "user"},
    ]
    edges = [
        {"edge_id": "owns1", "src": "acct1", "dst": "user1", "type": "owns"},
        {"edge_id": "owns2", "src": "acct2", "dst": "user1", "type": "owns"},
    ]
    ops = [
        n({"type": "user"}, name="u"),
        e_reverse({"type": "owns"}, name="owns_rev"),
        n({"type": "account"}, name="acct"),
    ]
    _run_parity_case(nodes, edges, ops)


def test_enumerator_matches_gfql_two_hop_chain():
    nodes = [
        {"id": "acct1", "type": "account"},
        {"id": "acct2", "type": "account"},
        {"id": "user1", "type": "user"},
        {"id": "user2", "type": "user"},
    ]
    edges = [
        {"edge_id": "txn1", "src": "acct1", "dst": "acct2", "type": "txn"},
        {"edge_id": "owns1", "src": "acct2", "dst": "user1", "type": "owns"},
        {"edge_id": "owns2", "src": "acct2", "dst": "user2", "type": "owns"},
    ]
    ops = [
        n({"type": "account"}, name="acct_start"),
        e_forward({"type": "txn"}, name="txn"),
        n({"type": "account"}, name="acct_mid"),
        e_forward({"type": "owns"}, name="owns"),
        n({"type": "user"}, name="user_end"),
    ]
    _run_parity_case(nodes, edges, ops)


def test_enumerator_matches_gfql_undirected_edges():
    nodes = [
        {"id": "n1", "type": "node"},
        {"id": "n2", "type": "node"},
        {"id": "n3", "type": "node"},
    ]
    edges = [
        {"edge_id": "e12", "src": "n1", "dst": "n2", "type": "path"},
        {"edge_id": "e23", "src": "n2", "dst": "n3", "type": "path"},
    ]
    ops = [
        n({"type": "node"}, name="start"),
        e_undirected({"type": "path"}, name="hop"),
        n({"type": "node"}, name="end"),
    ]
    _run_parity_case(nodes, edges, ops)


def test_enumerator_matches_gfql_empty_result():
    nodes = [
        {"id": "acct1", "type": "account"},
        {"id": "acct2", "type": "account"},
    ]
    edges = [
        {"edge_id": "e1", "src": "acct1", "dst": "acct2", "type": "txn"},
    ]
    ops = [
        n({"type": "user"}, name="start"),
        e_forward({"type": "txn"}, name="hop"),
        n({"type": "user"}, name="end"),
    ]
    _run_parity_case(nodes, edges, ops)
