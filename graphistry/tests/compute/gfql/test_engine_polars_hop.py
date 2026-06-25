"""Differential parity: native polars hop() == pandas hop().

Phase 1 (core BFS) of the GFQL polars engine. The pandas engine is the
correctness oracle; the polars lane must produce identical node-id and edge
sets. See plans/gfql-polars-engine.
"""
import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")


def _node_set(g):
    n = g._nodes
    if n is None:
        return set()
    if "polars" in type(n).__module__:
        n = n.to_pandas()
    return set(n[g._node].tolist())


def _edge_set(g):
    e = g._edges
    if e is None or len(e) == 0:
        return set()
    if "polars" in type(e).__module__:
        e = e.to_pandas()
    return set(zip(e[g._source].tolist(), e[g._destination].tolist()))


GRAPHS = {
    "line5": pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "e"]}),
    "cycle4": pd.DataFrame({"s": ["a", "b", "c", "d"], "d": ["b", "c", "d", "a"]}),
    "branch": pd.DataFrame({"s": ["a", "a", "b", "c", "x"], "d": ["b", "c", "d", "d", "y"]}),
}

CASES = [
    dict(hops=1, direction="forward"),
    dict(hops=2, direction="forward"),
    dict(hops=3, direction="forward"),
    dict(hops=1, direction="reverse"),
    dict(hops=2, direction="reverse"),
    dict(hops=1, direction="undirected"),
    dict(hops=2, direction="undirected"),
    dict(to_fixed_point=True, direction="forward"),
    dict(to_fixed_point=True, direction="undirected"),
    dict(hops=1, direction="forward", return_as_wave_front=True),
    dict(hops=2, direction="forward", return_as_wave_front=True),
]

SEEDS = [None, ["a"], ["a", "c"], ["z"]]  # z = absent


@pytest.mark.parametrize("gname", list(GRAPHS))
@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("seed", SEEDS)
def test_polars_hop_parity(gname, case, seed):
    base = graphistry.edges(GRAPHS[gname], "s", "d").materialize_nodes()
    nodes = None if seed is None else pd.DataFrame({base._node: seed})

    gp = base.hop(nodes=nodes, engine="pandas", **case)
    gl = base.hop(nodes=nodes, engine="polars", **case)

    assert "polars" in type(gl._nodes).__module__
    assert _node_set(gp) == _node_set(gl), f"node mismatch {gname} {case} seed={seed}"
    assert _edge_set(gp) == _edge_set(gl), f"edge mismatch {gname} {case} seed={seed}"


def test_polars_hop_unsupported_raises():
    base = graphistry.edges(GRAPHS["line5"], "s", "d").materialize_nodes()
    with pytest.raises(NotImplementedError):
        base.hop(hops=1, label_node_hops="h", engine="polars")
    with pytest.raises(NotImplementedError):
        base.hop(hops=1, source_node_query="s == 'a'", engine="polars")
