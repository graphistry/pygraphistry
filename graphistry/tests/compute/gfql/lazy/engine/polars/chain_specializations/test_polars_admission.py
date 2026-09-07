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
import graphistry.compute.gfql.lazy.engine.polars.chain as pchain
import graphistry.compute.gfql.lazy.engine.polars.chain_specializations.hotpaths as hot
from graphistry.compute.ast import e_forward, n
from graphistry.compute.gfql.lazy.engine.polars.chain_specializations.admission import polars_plain_single_hop_admits, polars_seeded_lane_admits
from graphistry.tests.compute.gfql.routes.corpus import CORPUS, EDGES, NODES, by_name

pl = pytest.importorskip("polars")

EXPECTED = {
    "plain single hop, unseeded": "skip-combine",
    "plain single hop, seeded": "seeded-index",
    "plain single hop, seeded, reverse": "seeded-index",
    "plain single hop, seeded, destination filter": "skip-combine",
    "plain single hop, undirected, unconstrained": "skip-combine",
    "plain single hop, undirected, seeded": None,
    "single hop, prune to endpoints": None,
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
    ops = by_name()[name].ops()
    g_pd = graphistry.nodes(NODES, "key").edges(EDGES, "s", "d", "eid")
    g_pl = graphistry.nodes(pl.from_pandas(NODES), "key").edges(pl.from_pandas(EDGES), "s", "d", "eid")
    assert _sig(g_pl.gfql(ops, engine="polars")) == _sig(g_pd.gfql(ops, engine="pandas", policy={"preload": lambda ctx: None}))


SEEDED_LANE_ADMITS = {
    "plain single hop, seeded", "plain single hop, seeded, reverse", "plain single hop, seeded, destination filter",
    "typed single hop, seeded", "typed single hop, seeded, named",
    "single hop, edge alias = filtered column", "single hop, destination alias = its filtered column",
    "single hop, node and edge alias share a name",
}


def test_seeded_lane_decision_table_over_the_corpus():
    got = {e.name for e in CORPUS if polars_seeded_lane_admits(e.ops())}
    assert got == SEEDED_LANE_ADMITS


def _indexed_polars_graph():
    g = graphistry.nodes(pl.from_pandas(NODES), "key").edges(pl.from_pandas(EDGES), "s", "d", "eid")
    return g.gfql_index_all(engine="polars").gfql_index_node_props(["id"], engine="polars")


@pytest.mark.parametrize("name", [e.name for e in CORPUS])
def test_seeded_lane_never_serves_a_shape_it_does_not_admit(name):
    ops = by_name()[name].ops()
    g = _indexed_polars_graph()
    real = pchain._try_seeded_chain_polars
    hit = {"n": 0}

    def spy(*a, **k):
        r = real(*a, **k)
        hit["n"] += r is not None
        return r
    pchain._try_seeded_chain_polars = spy
    try:
        g.gfql(ops, engine="polars", index_policy="use")
    except Exception:
        pass
    finally:
        pchain._try_seeded_chain_polars = real
    assert hit["n"] == 0 or polars_seeded_lane_admits(ops), f"{name}: served without admission"


SEEDED_LANE_SERVES_DIRECTLY = SEEDED_LANE_ADMITS - {
    # admitted by shape, declined by the body's alias-collision rule
    "single hop, edge alias = filtered column", "single hop, destination alias = its filtered column",
    "single hop, node and edge alias share a name",
}


@pytest.mark.route_engaged("polars-seeded")
@pytest.mark.parametrize("name", sorted(SEEDED_LANE_ADMITS))
def test_seeded_lane_called_directly_serves_every_admitted_non_colliding_shape(name):
    ops = by_name()[name].ops()
    res = hot._try_seeded_chain_polars(_indexed_polars_graph(), ops)
    assert (res is not None) == (name in SEEDED_LANE_SERVES_DIRECTLY)


@pytest.mark.parametrize("ops_name", ["single hop, prune to endpoints"])
def test_prune_to_endpoints_is_a_typed_decline_on_polars_and_served_on_pandas(ops_name):
    """A single-hop edge has no hop labels to prune by on polars, so prune_to_endpoints there
    raises the engine's NotImplementedError (variable-length hops keep their native pruning);
    pandas answers the shape."""
    ops = by_name()[ops_name].ops()
    g_pl = graphistry.nodes(pl.from_pandas(NODES), "key").edges(pl.from_pandas(EDGES), "s", "d", "eid")
    with pytest.raises(NotImplementedError, match="prune_to_endpoints"):
        g_pl.gfql(ops, engine="polars")
    unseeded = [n(), e_forward(prune_to_endpoints=True), n()]
    with pytest.raises(NotImplementedError, match="prune_to_endpoints"):
        g_pl.gfql(unseeded, engine="polars")
    g_pd = graphistry.nodes(NODES, "key").edges(EDGES, "s", "d", "eid")
    assert _sig(g_pd.gfql(ops, engine="pandas"))[0] == [2, 3]
    assert polars_plain_single_hop_admits(ops, None) is None
    assert polars_plain_single_hop_admits([n({"key": 1}), e_forward(), n()], None) == "seeded-index"
