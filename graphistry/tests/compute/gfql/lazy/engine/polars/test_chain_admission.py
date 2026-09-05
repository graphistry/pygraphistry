"""``polars_plain_single_hop_admits`` is the polars chain's own gate for its plain single-hop
branches.

Pins: the decision table over the shared corpus (``"seeded-index"`` for a directed seeded hop
without a destination filter, ``"skip-combine"`` for the other plain single hops except the
filtered undirected one, None otherwise), a seeded chain (``start_nodes``) declines, and every
admitted shape stays value-identical to the pandas full path.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.gfql.lazy.engine.polars.chain import polars_plain_single_hop_admits
from graphistry.tests.compute.gfql.routes.corpus import CORPUS, EDGES, NODES, by_name

pl = pytest.importorskip("polars")

EXPECTED = {
    "plain single hop, unseeded": "skip-combine",
    "plain single hop, seeded": "seeded-index",
    "plain single hop, seeded, reverse": "seeded-index",
    "plain single hop, seeded, destination filter": "skip-combine",
    "plain single hop, undirected, unconstrained": "skip-combine",
    "plain single hop, undirected, seeded": None,
    "single hop, prune to endpoints": "seeded-index",
}


def test_every_corpus_shape_has_a_verdict():
    for e in CORPUS:
        assert polars_plain_single_hop_admits(e.ops(), None) == EXPECTED.get(e.name), e.name


@pytest.mark.parametrize("name", list(EXPECTED))
def test_decision_table(name):
    assert polars_plain_single_hop_admits(by_name()[name].ops(), None) == EXPECTED[name]


def test_start_nodes_decline():
    assert polars_plain_single_hop_admits([n({"key": 1}), e_forward(), n()], pd.DataFrame({"key": [1]})) is None


def _sig(res):
    nn = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    ee = res._edges.to_pandas() if hasattr(res._edges, "to_pandas") else res._edges
    return sorted(nn["key"].tolist()), sorted(map(tuple, ee[["s", "d"]].values.tolist()))


@pytest.mark.parametrize("name", [k for k, v in EXPECTED.items() if v is not None])
def test_admitted_shapes_match_the_pandas_full_path(name, request):
    if name == "single hop, prune to endpoints":
        request.applymarker(pytest.mark.xfail(strict=True, reason="graphistry/pygraphistry#2053"))
    ops = by_name()[name].ops()
    g_pd = graphistry.nodes(NODES, "key").edges(EDGES, "s", "d", "eid")
    g_pl = graphistry.nodes(pl.from_pandas(NODES), "key").edges(pl.from_pandas(EDGES), "s", "d", "eid")
    assert _sig(g_pl.gfql(ops, engine="polars")) == _sig(g_pd.gfql(ops, engine="pandas", policy={"preload": lambda ctx: None}))
