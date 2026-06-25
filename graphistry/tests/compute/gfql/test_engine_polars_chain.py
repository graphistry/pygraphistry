"""Differential parity: native polars chain() == pandas chain().

Phase 1 of the GFQL polars engine. Pandas is the oracle; polars must produce
identical node/edge sets and alias columns. See plans/gfql-polars-engine.
"""
import random

import pandas as pd
import pytest

import graphistry
from graphistry import gt, lt, ge, le, eq, ne, between, is_in, contains, startswith, endswith
from graphistry.compute.ast import n, e, e_forward, e_reverse, e_undirected

pl = pytest.importorskip("polars")


def _nset(g):
    df = g._nodes
    if df is None:
        return set()
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    return set(df[g._node].tolist())


def _eset(g):
    df = g._edges
    if df is None or len(df) == 0:
        return set()
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    return set(zip(df[g._source].tolist(), df[g._destination].tolist()))


def _named(g, col):
    df = g._nodes if col in (g._nodes.columns if g._nodes is not None else []) else g._edges
    if df is None or col not in df.columns:
        return None
    if "polars" in type(df).__module__:
        df = df.to_pandas()
    key = g._node if g._node in df.columns else g._source
    return set(df[df[col].fillna(False).astype(bool)][key].tolist())


NODES = pd.DataFrame({
    "id": ["a", "b", "c", "d", "e", "f", "g"],
    "kind": ["x", "y", "y", "z", "x", "z", "y"],
    "name": ["alice", "bob", "carol", "dave", "erin", "frank", "grace"],
    "score": [10, 20, 30, 40, 50, 60, 70],
})
EDGES = pd.DataFrame({
    "s": ["a", "a", "b", "c", "d", "e", "b", "g", "c"],
    "d": ["b", "c", "c", "d", "e", "f", "d", "a", "g"],
    "rel": ["r1", "r2", "r1", "r2", "r1", "r2", "r1", "r2", "r1"],
    "w": [1, 2, 3, 4, 5, 6, 7, 8, 9],
})
BASE = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")

CHAINS = {
    "n-e-n": [n(), e_forward(), n()],
    "filter src": [n({"kind": "x"}), e_forward(), n()],
    "filter dst": [n(), e_forward(), n({"kind": "z"})],
    "edge_match": [n(), e_forward({"rel": "r1"}), n()],
    "pred gt": [n({"score": gt(15)}), e_forward(), n()],
    "pred lt": [n(), e_forward(), n({"score": lt(45)})],
    "pred ge": [n({"score": ge(30)}), e_forward(), n()],
    "pred le": [n(), e_forward(), n({"score": le(30)})],
    "pred eq": [n({"score": eq(30)}), e_forward(), n()],
    "pred ne": [n({"score": ne(30)}), e_forward(), n()],
    "pred between": [n({"score": between(20, 50)}), e_forward(), n()],
    "pred is_in": [n({"kind": is_in(["x", "z"])}), e_forward(), n()],
    "pred contains": [n({"name": contains("a")}), e_forward(), n()],
    "pred startswith": [n({"name": startswith("a")}), e_forward(), n()],
    "pred endswith": [n({"name": endswith("e")}), e_forward(), n()],
    "reverse": [n(), e_reverse(), n()],
    "undirected": [n({"kind": "y"}), e_undirected(), n()],
    "two-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n()],
    "three-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n(), e_forward(), n()],
    "edge+dst": [n(), e_forward({"rel": "r1"}), n({"kind": "y"})],
    "src+edge+dst": [n({"kind": "x"}), e_forward({"rel": "r2"}), n({"kind": "y"})],
    "named nodes": [n({"kind": "x"}, name="srcs"), e_forward(), n(name="dsts")],
    "named edge": [n(), e_forward(name="hop1"), n()],
    "named both": [n({"kind": "y"}, name="ys"), e_forward(name="h"), n(name="tgt")],
    "named reverse mid-filter": [n(name="a"), e_reverse(name="e1"), n({"kind": "y"}, name="b")],
    "node only": [n({"kind": "y"})],
    "reverse two-hop": [n({"kind": "z"}), e_reverse(), n(), e_reverse(), n()],
    "undirected filter": [n(), e_undirected({"rel": "r1"}), n({"score": gt(25)})],
    "empty result": [n({"kind": "x"}), e_forward({"rel": "nope"}), n()],
}


@pytest.mark.parametrize("cname", list(CHAINS))
def test_polars_chain_parity(cname):
    ch = CHAINS[cname]
    gp = BASE.chain(ch, engine="pandas")
    gl = BASE.chain(ch, engine="polars")
    assert "polars" in type(gl._nodes).__module__
    assert _nset(gp) == _nset(gl), f"node mismatch [{cname}]"
    assert _eset(gp) == _eset(gl), f"edge mismatch [{cname}]"
    for op in ch:
        nm = getattr(op, "_name", None)
        if nm:
            assert _named(gp, nm) == _named(gl, nm), f"alias[{nm}] mismatch [{cname}]"


# ---- Randomized differential fuzzer (the CHANGELOG-advertised fuzzer) ----

def _rand_node(rng):
    r = rng.random()
    if r < 0.4:
        return n()
    if r < 0.6:
        return n({"kind": rng.choice(["x", "y", "z"])})
    if r < 0.8:
        return n({"score": gt(rng.randint(0, 80))})
    return n({"score": lt(rng.randint(20, 100))})


def _rand_edge(rng):
    ctor = rng.choice([e_forward, e_reverse, e_undirected])
    if rng.random() < 0.5:
        return ctor({"rel": rng.choice(["r1", "r2", "r3"])})
    return ctor()


def _rand_chain(rng):
    ops = [_rand_node(rng)]
    for _ in range(rng.randint(1, 3)):
        ops.append(_rand_edge(rng))
        ops.append(_rand_node(rng))
    return ops


def _rand_graph(rng):
    nn = rng.randint(4, 12)
    ids = [f"n{i}" for i in range(nn)]
    nodes = pd.DataFrame({
        "id": ids,
        "kind": [rng.choice(["x", "y", "z"]) for _ in ids],
        "score": [rng.randint(0, 100) for _ in ids],
    })
    ne = rng.randint(3, 24)
    edges = pd.DataFrame({
        "s": [rng.choice(ids) for _ in range(ne)],
        "d": [rng.choice(ids) for _ in range(ne)],
        "rel": [rng.choice(["r1", "r2", "r3"]) for _ in range(ne)],
    })
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@pytest.mark.parametrize("seed", range(60))
def test_polars_chain_fuzz_parity(seed):
    rng = random.Random(seed)
    g = _rand_graph(rng)
    ch = _rand_chain(rng)
    gp = g.chain(ch, engine="pandas")
    try:
        gl = g.chain(ch, engine="polars")
    except NotImplementedError:
        pytest.skip("deferred surface")
    assert _nset(gp) == _nset(gl), f"node mismatch seed={seed} chain={ch}"
    assert _eset(gp) == _eset(gl), f"edge mismatch seed={seed} chain={ch}"


# ---- Deferred-surface guards (must raise, never silently wrong) ----

@pytest.mark.parametrize("ch", [
    [n(), e(hops=2), n()],
    [n(), e(to_fixed_point=True), n()],
    [n(), e_undirected(), n(), e_undirected(), n()],  # undirected multi-edge
])
def test_polars_chain_deferred_raises(ch):
    with pytest.raises(NotImplementedError):
        BASE.chain(ch, engine="polars")


def test_polars_chain_node_query_raises():
    with pytest.raises(NotImplementedError):
        BASE.chain([n(query="score > 10"), e_forward(), n()], engine="polars")


# ---- Edge cases: empty graph, duplicate edges (multiplicity), edges-only ----

def test_polars_chain_empty_graph():
    g = graphistry.nodes(pd.DataFrame({"id": []}), "id").edges(
        pd.DataFrame({"s": [], "d": []}), "s", "d")
    gp = g.chain([n(), e_forward(), n()], engine="pandas")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    assert _nset(gp) == _nset(gl) == set()
    assert _eset(gp) == _eset(gl) == set()


def test_polars_chain_duplicate_edges_multiplicity():
    edges = pd.DataFrame({"s": ["a", "a", "a", "b"], "d": ["b", "b", "c", "c"]})  # (a,b) x2
    g = graphistry.nodes(pd.DataFrame({"id": ["a", "b", "c"]}), "id").edges(edges, "s", "d")
    gp = g.chain([n(), e_forward(), n()], engine="pandas")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    # assert on edge COUNT (set(zip) would hide a dropped parallel edge)
    assert len(gl._edges) == len(gp._edges)


def test_polars_chain_edges_only_runs():
    # No node table / binding: pandas is degenerate here (nan node, _node=None);
    # polars returns clean materialized nodes. Edges must still match and it must
    # not crash (the prior BLOCKER).
    g = graphistry.edges(pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]}), "s", "d")
    gp = g.chain([n(), e_forward(), n()], engine="pandas")
    gl = g.chain([n(), e_forward(), n()], engine="polars")
    assert _eset(gp) == _eset(gl)
    assert _nset(gl) == {0, 1, 2}


def test_polars_chain_pandas_start_nodes():
    sn = pd.DataFrame({"id": ["a"]})
    gp = BASE.chain([n(), e_forward(), n()], engine="pandas", start_nodes=sn)
    gl = BASE.chain([n(), e_forward(), n()], engine="polars", start_nodes=sn)
    assert _nset(gp) == _nset(gl)
    assert _eset(gp) == _eset(gl)
