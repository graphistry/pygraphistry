"""Differential parity: native polars chain() == pandas chain().

Phase 1 of the GFQL polars engine. Pandas is the oracle; polars must produce
identical node/edge sets and alias columns. See plans/gfql-polars-engine.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected
from graphistry.compute.predicates.numeric import gt, lt

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
    "reverse": [n(), e_reverse(), n()],
    "undirected": [n({"kind": "y"}), e_undirected(), n()],
    "two-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n()],
    "three-hop": [n({"kind": "x"}), e_forward(), n(), e_forward(), n(), e_forward(), n()],
    "edge+dst": [n(), e_forward({"rel": "r1"}), n({"kind": "y"})],
    "src+edge+dst": [n({"kind": "x"}), e_forward({"rel": "r2"}), n({"kind": "y"})],
    "named nodes": [n({"kind": "x"}, name="srcs"), e_forward(), n(name="dsts")],
    "named edge": [n(), e_forward(name="hop1"), n()],
    "named both": [n({"kind": "y"}, name="ys"), e_forward(name="h"), n(name="tgt")],
    "node only": [n({"kind": "y"})],
    "reverse two-hop": [n({"kind": "z"}), e_reverse(), n(), e_reverse(), n()],
    "undirected filter": [n(), e_undirected({"rel": "r1"}), n({"score": gt(25)})],
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


def test_polars_chain_multihop_unsupported():
    from graphistry.compute.ast import e
    with pytest.raises(NotImplementedError):
        BASE.chain([n(), e(hops=2), n()], engine="polars")
