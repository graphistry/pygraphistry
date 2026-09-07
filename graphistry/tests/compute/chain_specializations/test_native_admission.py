"""``native_fast_path_admits`` is the pandas/cuDF chain fast path's own gate.

Pins: for every corpus shape the predicate's verdict equals whether ``_try_chain_fast_path``
served it (admits ⇔ served, on pandas and cuDF), the decision table is stable per shape, a
seeded chain (``start_nodes``) and a non pandas/cuDF engine always decline, and every served
shape stays value-identical to the full path.
"""
import pandas as pd
import pytest

import graphistry
import graphistry.compute.chain as chain_mod
from graphistry.Engine import Engine
from graphistry.compute.ast import e_forward, n
from graphistry.compute.chain_specializations.admission import native_fast_path_admits
from graphistry.tests.compute.gfql.routes.corpus import CORPUS, EDGES, NODES, by_name

ENGINES = ["pandas", "cudf"]


def _graph(engine):
    nodes, edges = NODES, EDGES
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "key").edges(edges, "s", "d", "eid")


def _served(g, ops, engine):
    real = chain_mod._try_chain_fast_path
    hit = {"n": 0}

    def spy(*a, **k):
        r = real(*a, **k)
        hit["n"] += r is not None
        return r
    chain_mod._try_chain_fast_path = spy
    try:
        res = g.gfql(ops, engine=engine)
    finally:
        chain_mod._try_chain_fast_path = real
    return res, hit["n"] == 1


def _sig(res):
    nn = res._nodes.to_pandas() if hasattr(res._nodes, "to_pandas") else res._nodes
    ee = res._edges.to_pandas() if hasattr(res._edges, "to_pandas") else res._edges
    return sorted(nn["key"].tolist()), sorted(map(tuple, ee[["s", "d"]].values.tolist()))


EXPECTED = {
    "single node, scalar filter": "single-node", "single node, named": "single-node",
    "single node, predicate filter": "single-node", "single node, no filter": "single-node",
    "plain single hop, unseeded": "seeded-hop", "plain single hop, seeded": "seeded-hop",
    "plain single hop, seeded, reverse": "seeded-hop", "plain single hop, seeded, destination filter": "seeded-hop",
    "plain single hop, undirected, unconstrained": "seeded-hop", "plain single hop, undirected, seeded": None,
    "typed single hop, seeded": "seeded-hop", "typed single hop, seeded, named": "seeded-hop",
    "typed single hop, seeded, named, undirected": None, "single hop, node and edge alias share a name": None,
    "single hop, edge alias = filtered column": "seeded-hop", "single hop, destination alias = its filtered column": "seeded-hop",
    "single hop, source node match": None, "single hop, prune to endpoints": None,
    "hops=2, seeded": None, "hops=2, seeded, typed, named": None, "to_fixed_point, seeded": None, "two single hops": None,
}


def test_every_corpus_shape_has_an_expected_verdict():
    assert set(EXPECTED) == {e.name for e in CORPUS}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_decision_table(name):
    assert native_fast_path_admits(by_name()[name].ops(), Engine.PANDAS, None) == EXPECTED[name]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("name", list(EXPECTED))
def test_admits_iff_served(engine, name):
    g = _graph(engine)
    ops = by_name()[name].ops()
    admitted = native_fast_path_admits(ops, Engine(engine), None) is not None
    res, served = _served(g, ops, engine)
    assert served == admitted, f"{name}: predicate={admitted} served={served}"
    if served:
        full = g.gfql(ops, engine=engine, policy={"preload": lambda ctx: None})
        assert _sig(res) == _sig(full)


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.DASK])
def test_other_engines_decline(engine):
    assert native_fast_path_admits([n({"id": 30})], engine, None) is None


def test_start_nodes_decline():
    assert native_fast_path_admits([n({"key": 1}), e_forward(), n()], Engine.PANDAS, pd.DataFrame({"key": [1]})) is None
