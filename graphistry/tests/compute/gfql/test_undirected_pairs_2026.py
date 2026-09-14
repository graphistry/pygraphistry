"""Undirected hop builds both edge orientations without deduplicating the doubled frame.

Every consumer of the doubled frame dedups on the edge id, so the whole-frame dedup was
redundant. Pins: parity with an independent oracle on a graph with self-loops, parallel
edges and hub seeds on every engine; self-loops kept once through the multi-hop path; and
categorical endpoint columns with differing category sets keep working.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_undirected, n
from graphistry.compute.predicates.is_in import is_in

ENGINES = ["pandas", "polars", "cudf"]


def _frames(seed=0):
    rng = np.random.default_rng(seed)
    N, E = 3000, 12000
    edges = pd.DataFrame({"s": rng.integers(0, N, E), "d": rng.integers(0, N, E)})
    edges.loc[:40, "d"] = edges.loc[:40, "s"]  # self-loops, some on hub seeds
    edges = pd.concat([edges, edges.iloc[:30]], ignore_index=True)  # parallel edges
    nodes = pd.DataFrame({"id": np.arange(N)})
    return nodes, edges


def _graph(engine):
    nodes, edges = _frames()
    if engine == "polars":
        pl = pytest.importorskip("polars")
        nodes, edges = pl.from_pandas(nodes), pl.from_pandas(edges)
    elif engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


def _oracle_ball(edges, seeds, hops):
    """Node set within `hops` undirected steps of the seeds, by plain set expansion."""
    adj = {}
    for s, d in zip(edges["s"].tolist(), edges["d"].tolist()):
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)
    frontier, seen = set(seeds), set(seeds)
    for _ in range(hops):
        nxt = set()
        for u in frontier:
            nxt |= adj.get(u, set())
        frontier = nxt - seen
        seen |= nxt
    return seen


def _ids(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    return set(int(v) for v in df["id"].tolist())


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", [1, 2])
def test_undirected_hop_matches_set_oracle_with_self_loops_and_parallel_edges(engine, hops):
    nodes, edges = _frames()
    deg = pd.concat([edges["s"], edges["d"]]).value_counts()
    seeds = [int(v) for v in deg.index[:5]]
    seeds.append(int(edges.loc[0, "s"]))  # a seed carrying a self-loop
    g = _graph(engine)
    got = g.gfql([n({"id": is_in(seeds)}), e_undirected(hops=hops), n()], engine=engine)
    expected = _oracle_ball(edges, seeds, hops)
    # the wavefront keeps a seed only when an edge reaches it; the oracle includes every seed
    assert _ids(got) - set(seeds) == expected - set(seeds)
    assert _ids(got) <= expected


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", [1, 2])
def test_undirected_edges_keep_every_self_loop_once(engine, hops):
    """hops=2 goes through the doubled-frame build; hops=1 is served by the seeded lane."""
    nodes, edges = _frames()
    loop_seed = int(edges.loc[0, "s"])
    g = _graph(engine)
    out = g.gfql([n({"id": is_in([loop_seed])}), e_undirected(hops=hops), n()], engine=engine)
    e = out._edges
    e = e.to_pandas() if hasattr(e, "to_pandas") else pd.DataFrame(e)
    loops = e[(e["s"] == loop_seed) & (e["d"] == loop_seed)]
    expected = edges[(edges["s"] == loop_seed) & (edges["d"] == loop_seed)]
    assert len(loops) == len(expected)
    # every returned edge row is one input edge row (parallel edges included): the
    # reverse orientation never surfaces as an extra row
    counts = e.groupby(["s", "d"]).size()
    source = edges.groupby(["s", "d"]).size()
    assert all(counts[k] <= source[k] for k in counts.index)


@pytest.mark.parametrize("engine", ["pandas"])  # cuDF cannot concat categoricals of differing category sets
def test_categorical_endpoints_with_different_category_sets(engine):
    edges = pd.DataFrame({"s": ["a", "b", "b", "a"], "d": ["b", "c", "b", "d"]}).astype(
        {"s": "category", "d": "category"})
    nodes = pd.DataFrame({"id": list("abcd")})
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        nodes, edges = cudf.from_pandas(nodes), cudf.from_pandas(edges)
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    seeds = pd.DataFrame({"id": ["a"]})
    if engine == "cudf":
        import cudf
        seeds = cudf.from_pandas(seeds)
    out = g.hop(nodes=seeds, hops=2, direction="undirected")
    assert _ids_str(out) == {"a", "b", "c", "d"}
    assert len(out._edges) == 4


def _ids_str(res):
    nodes = res._nodes
    df = nodes.to_pandas() if hasattr(nodes, "to_pandas") else pd.DataFrame(nodes)
    return set(str(v) for v in df["id"].tolist())
