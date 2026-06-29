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


@pytest.mark.parametrize("kw", [
    {"label_node_hops": "h"},
    {"label_edge_hops": "h"},
    {"label_seeds": True},
    {"min_hops": 2},
    {"output_min_hops": 1},
    {"output_max_hops": 2},
    {"source_node_query": "s == 'a'"},
    {"destination_node_query": "s == 'a'"},
    {"edge_query": "s == 'a'"},
    {"include_zero_hop_seed": True},
])
def test_polars_hop_unsupported_raises(kw):
    base = graphistry.edges(GRAPHS["line5"], "s", "d").materialize_nodes()
    with pytest.raises(NotImplementedError):
        base.hop(hops=1, engine="polars", **kw)


def test_polars_hop_min_hops_1_allowed():
    # min_hops=1 is the default and must NOT raise (boundary guard).
    base = graphistry.edges(GRAPHS["line5"], "s", "d").materialize_nodes()
    base.hop(hops=1, min_hops=1, engine="polars")


def test_polars_hop_edges_only_parity():
    g = graphistry.edges(pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]}), "s", "d")
    gp = g.hop(hops=1, engine="pandas")
    gl = g.hop(hops=1, engine="polars")
    assert _node_set(gp) == _node_set(gl)
    assert _edge_set(gp) == _edge_set(gl)


def test_polars_hop_dtype_mismatch_parity():
    # node ids float (with a null), edge endpoints int — pandas coerces; polars
    # must align join-key dtypes rather than crash.
    nodes = pd.DataFrame({"id": [0.0, 1.0, 2.0, None], "k": ["x", "y", "z", "x"]})
    edges = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    gp = g.hop(hops=1, engine="pandas")
    gl = g.hop(hops=1, engine="polars")
    assert _node_set(gp) == _node_set(gl)
    assert _edge_set(gp) == _edge_set(gl)


def test_polars_hop_predicate_matches_parity():
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d"], "score": [10, 20, 30, 40]})
    edges = pd.DataFrame({"s": ["a", "b", "c"], "d": ["b", "c", "d"], "rel": ["r1", "r2", "r1"]})
    g = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    from graphistry import gt
    # all-nodes hop (no custom seed) so node-match resolves against the full
    # node table for both engines.
    for kw in (
        {"edge_match": {"rel": "r1"}},
        {"source_node_match": {"score": gt(15)}},
        {"destination_node_match": {"score": gt(15)}},
    ):
        gp = g.hop(hops=1, engine="pandas", **kw)
        gl = g.hop(hops=1, engine="polars", **kw)
        assert _node_set(gp) == _node_set(gl), kw
        assert _edge_set(gp) == _edge_set(gl), kw


def test_polars_filter_by_dict_exotic_predicate_declines():
    # NO-CHEATING: an exotic predicate with no native polars lowering must raise
    # NotImplementedError, NOT silently evaluate the column via pandas (the old
    # bridge misrepresented pandas semantics as polars).
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_by_dict_polars, predicate_to_expr

    class IsOdd(ASTPredicate):
        def __call__(self, s):
            return (s % 2) == 1

    df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "v": [1, 2, 3, 4, 5]})
    assert predicate_to_expr("v", IsOdd()) is None  # not lowered
    with pytest.raises(NotImplementedError):
        filter_by_dict_polars(df, {"v": IsOdd()})
