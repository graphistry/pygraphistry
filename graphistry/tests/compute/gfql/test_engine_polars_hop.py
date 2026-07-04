"""Differential parity: native polars hop() == pandas hop().

Phase 1 (core BFS) of the GFQL polars engine. The pandas engine is the
correctness oracle; the polars lane must produce identical node-id and edge
sets. See plans/gfql-polars-engine.
"""
import pandas as pd
import pytest

import graphistry

pl = pytest.importorskip("polars")

from graphistry.tests.compute.gfql.polars_test_utils import (  # noqa: E402
    node_id_set as _node_set,
    edge_pair_set as _edge_set,
    node_attr_map as _node_attrs_hop,
)


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
    {"min_hops": 2},   # min_hops>1 is native in chain()/gfql() only; a DIRECT hop() stays NIE
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


@pytest.mark.parametrize("direction", ["forward", "reverse"])
@pytest.mark.parametrize("seed_sel", [["a"], ["c"], ["a", "d"]])
def test_polars_hop_min_hops_labeled_policy_unit_parity(direction, seed_sel):
    # Unit test of the CHAIN-CONTEXT min_hops node policy (the surface the chain actually uses):
    # drive the internal polars hop with `min_hops_label_policy=True` and compare the FULL node
    # frame (incl non-id attrs, null-aware) against pandas' LABELED hop (`label_node_hops=...`, the
    # track_node_hops path the chain's ASTEdge.execute takes). This asserts the null-attribute /
    # seed-strip / endpoint contract at the unit level (seed-48 class) without building a chain.
    # NOTE: a *direct* base.hop(engine='polars', min_hops>1) intentionally stays NIE (see
    # test_polars_hop_unsupported_raises) — only the chain wires up this policy.
    from graphistry.compute.gfql.lazy.engine.polars.hop_eager import hop_polars
    from graphistry.Engine import Engine, df_to_engine
    nodes = pd.DataFrame({"id": ["a", "b", "c", "d", "e"],
                          "kind": ["x", "y", "y", "z", "x"], "score": [1, 2, 3, 4, 5]})
    edges = pd.DataFrame({"s": ["a", "b", "c", "d", "a", "c"], "d": ["b", "c", "d", "e", "c", "a"]})
    base = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    sel = base._nodes[base._nodes["id"].isin(seed_sel)]
    gp = base.hop(nodes=sel, min_hops=2, max_hops=3, direction=direction, return_as_wave_front=True,
                  label_node_hops="__gfql_output_node_hop__", label_edge_hops="__gfql_output_edge_hop__",
                  engine="pandas")
    gpol = base.nodes(df_to_engine(base._nodes, Engine.POLARS), "id").edges(
        df_to_engine(base._edges, Engine.POLARS), "s", "d")
    gl = hop_polars(gpol, nodes=df_to_engine(sel, Engine.POLARS), min_hops=2, max_hops=3,
                    direction=direction, return_as_wave_front=True, min_hops_label_policy=True)
    assert _node_set(gp) == _node_set(gl)
    assert _node_attrs_hop(gp) == _node_attrs_hop(gl), f"labeled-policy attr divergence {direction} {seed_sel}"


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
