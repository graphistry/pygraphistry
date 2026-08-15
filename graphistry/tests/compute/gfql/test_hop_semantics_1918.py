"""Round-011 direct-``hop()`` semantics pins (#1918).

Companion to ``test_hop_semantics_pins.py`` (#1892). Every case here reproduces a finding
from the round-011 re-probe of the direct ``hop()`` surface, with hand-computed oracles
written against the documented semantics in ``graphistry/compute/hop.py``'s docstring:

* F1 -- an edges-only graph returned the FULL materialized node table next to correctly
  filtered edges (the node-output block was gated on ``self._nodes is not None``).
* F2 -- any ``track_hops`` flag dropped the seed from the node output and the endpoint
  backfill re-added it id-only, NaN-ing its attributes and upcasting int64 -> float64.
* F3 -- an undirected edge traversed in both directions inside one wavefront produced TWO
  output edge rows under hop labeling.
* F4 -- ``to_fixed_point`` disagreed with the saturated bounded hop on any acyclic
  single-seed component (also widens the #1892 pin in ``test_hop_semantics_pins.py``).
* F5 -- ``label_seeds``, a label-column flag, changed the returned node SET.
* F6/F7 -- polars validated no bounds at all; ``min_hops=-1, hops=1`` returned an ANSWER.
* F8 -- a directed cycle with ``min_hops=4, max_hops=5`` returned empty (#1787-adjacent).

Parameterization rule for this file (the mistake that hid F4): a pin must cover the arms
that FAIL, not only the arms that pass. Filters, seed counts, directions and engines are
crossed rather than fixed wherever the finding admits more than one arm.
"""
import pandas as pd
import pytest

import graphistry

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


# ---------------------------------------------------------------------------- helpers
def _pd(df):
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def _frame(engine, df):
    return pl.from_pandas(df) if engine == "polars" else df


def node_ids(g, col="id"):
    return sorted(_pd(g._nodes)[col].tolist())


def edge_ids(g, s="s", d="d"):
    return sorted(map(tuple, _pd(g._edges)[[s, d]].itertuples(index=False)))


def _seeds(engine, ids):
    return _frame(engine, pd.DataFrame({"id": pd.Series(ids, dtype="int64")}))


# ------------------------------------------------------------------------- fixtures
#: attributed graph: 0(a,10) -x-> 1(b,20) -y-> 2(a,30). ``w`` is int64 on purpose -- F2's
#: dtype upcast is only observable on a non-float attribute column.
def _attr_graph(engine):
    ndf = pd.DataFrame({"id": [0, 1, 2], "type": ["a", "b", "a"],
                        "w": pd.Series([10, 20, 30], dtype="int64")})
    edf = pd.DataFrame({"s": [0, 1], "d": [1, 2], "rel": ["x", "y"]})
    return graphistry.nodes(_frame(engine, ndf), "id").edges(_frame(engine, edf), "s", "d")


#: undirected-topology fixtures for F4. Each is ACYCLIC except ``tri``; the acyclic ones with
#: a single seed are exactly the shapes the old heuristics dropped.
_TOPOLOGIES = {
    "path2": [(0, 1)],                       # 2-node path
    "path3": [(0, 1), (1, 2)],               # 3-node path (seed 1 sits in the middle)
    "star": [(0, 1), (0, 2), (0, 3)],        # star (seed 0 is the hub, 1..3 are leaves)
    "tri": [(0, 1), (1, 2), (2, 0)],         # cycle -- the arm that already worked
}


def _topology(engine, name):
    edges = _TOPOLOGIES[name]
    ids = sorted({v for e in edges for v in e})
    ndf = pd.DataFrame({"id": ids})
    edf = pd.DataFrame(edges, columns=["s", "d"])
    return graphistry.nodes(_frame(engine, ndf), "id").edges(_frame(engine, edf), "s", "d")


def _edges_only(engine, rel=None):
    """No node table bound -- hop() must synthesize AND then filter it (F1)."""
    edf = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4]})
    if rel is not None:
        edf["rel"] = rel
    return graphistry.edges(_frame(engine, edf), "s", "d")


# ============================================================================ F4
# to_fixed_point == saturated bounded, on every undirected topology x every single seed.
# Hand oracle: hop() is WALK-based (the docstring's "paths", no relationship isomorphism),
# so hop 2 may return along the edge hop 1 arrived on. Therefore on a connected undirected
# component EVERY seed is re-encountered by hop 2 and the whole component is returned --
# including the acyclic single-seed cases the old heuristics truncated.

_F4_CASES = [
    ("path2", 0, [0, 1]),
    ("path2", 1, [0, 1]),
    ("path3", 0, [0, 1, 2]),
    ("path3", 1, [0, 1, 2]),   # seed in the MIDDLE of an acyclic component
    ("path3", 2, [0, 1, 2]),
    ("star", 0, [0, 1, 2, 3]),  # hub
    ("star", 1, [0, 1, 2, 3]),  # leaf
    ("star", 3, [0, 1, 2, 3]),
    ("tri", 0, [0, 1, 2]),      # cyclic: correct BEFORE the fix too; must stay correct
    ("tri", 2, [0, 1, 2]),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("topo,seed,expect", _F4_CASES,
                         ids=[f"{t}-seed{s}" for t, s, _ in _F4_CASES])
def test_f4_tfp_equals_saturated_bounded_single_seed(engine, topo, seed, expect):
    g = _topology(engine, topo)
    kw = dict(nodes=_seeds(engine, [seed]), direction="undirected",
              return_as_wave_front=True, engine=engine)
    bounded = g.hop(hops=6, to_fixed_point=False, **kw)   # 6 >> diameter: saturated
    fixed = g.hop(hops=6, to_fixed_point=True, **kw)
    # the invariant ...
    assert node_ids(fixed) == node_ids(bounded)
    assert edge_ids(fixed) == edge_ids(bounded)
    # ... and the VALUE, so neither side can satisfy it by being empty
    assert node_ids(fixed) == expect
    assert edge_ids(fixed) == sorted(_TOPOLOGIES[topo])


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("topo", list(_TOPOLOGIES), ids=list(_TOPOLOGIES))
def test_f4_tfp_equals_saturated_bounded_under_filters(engine, topo):
    """The #1892 F-02 leak fix must survive the F4 widening: with a destination filter the
    invariant still holds, i.e. the reached-set rule prunes filtered-out seeds too."""
    g = _topology(engine, topo)
    kw = dict(nodes=_seeds(engine, [0]), direction="undirected", return_as_wave_front=True,
              destination_node_match={"id": 1}, engine=engine)
    bounded = g.hop(hops=6, to_fixed_point=False, **kw)
    fixed = g.hop(hops=6, to_fixed_point=True, **kw)
    assert node_ids(fixed) == node_ids(bounded)
    assert edge_ids(fixed) == edge_ids(bounded)
    # every hop is gated to destination id==1, so the seed is never re-encountered
    assert node_ids(fixed) == [1]


# ============================================================================ F1
# Edges-only graph: the traversal result must be APPLIED to the synthesized node table.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kw,expect_nodes,expect_edges", [
    (dict(hops=1), [0, 1], [(0, 1)]),
    (dict(hops=2), [0, 1, 2], [(0, 1), (1, 2)]),
    (dict(hops=1, direction="undirected"), [0, 1], [(0, 1)]),
    (dict(to_fixed_point=True), [0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)]),
], ids=["h1", "h2", "h1-undirected", "tfp"])
def test_f1_edges_only_graph_applies_the_traversal_to_nodes(engine, kw, expect_nodes,
                                                            expect_edges):
    # 0->1->2->3->4, no node table. Pre-fix pandas returned nodes [0,1,2,3,4] for EVERY
    # case (the whole materialized table) alongside correctly filtered edges.
    g = _edges_only(engine)
    r = g.hop(nodes=_seeds(engine, [0]), engine=engine, **kw)
    assert node_ids(r) == expect_nodes
    assert edge_ids(r) == expect_edges


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("kw", [
    dict(nodes_ids=[99], hops=2),                    # seed absent from the graph
    dict(nodes_ids=[0], hops=2, edge_match={"rel": "nope"}),  # filter kills every edge
], ids=["absent-seed", "no-edge-matches"])
def test_f1_edges_only_empty_traversal_returns_no_nodes(engine, kw):
    kw = dict(kw)
    ids = kw.pop("nodes_ids")
    g = _edges_only(engine, rel=["x", "x", "x", "x"])
    r = g.hop(nodes=_seeds(engine, ids), engine=engine, **kw)
    assert node_ids(r) == []
    assert edge_ids(r) == []


@pytest.mark.parametrize("engine", ENGINES)
def test_f1_edges_only_result_is_self_consistent(engine):
    """The defining property: the node table must be exactly the edge endpoints (no node
    exists without an incident retained edge, on a graph whose nodes ARE its endpoints)."""
    g = _edges_only(engine)
    r = g.hop(nodes=_seeds(engine, [1]), hops=1, engine=engine)
    endpoints = sorted({v for e in edge_ids(r) for v in e})
    assert node_ids(r) == endpoints == [1, 2]


# ============================================================================ F2
# Every track_hops flag must leave a traversed SEED's attributes and dtypes intact.

_TRACK_FLAGS = [
    ("label_node_hops", dict(label_node_hops="nh")),
    ("label_edge_hops", dict(label_edge_hops="eh")),
    ("output_min_hops", dict(output_min_hops=1)),
    ("output_max_hops", dict(output_max_hops=2)),
    ("label_seeds", dict(label_node_hops="nh", label_seeds=True)),
]


@pytest.mark.parametrize("flag_id,flags", _TRACK_FLAGS, ids=[i for i, _ in _TRACK_FLAGS])
@pytest.mark.parametrize("direction", ["forward", "undirected"])
def test_f2_tracked_seed_keeps_its_attributes_and_dtype(flag_id, flags, direction):
    """pandas oracle: seed 0 is a traversed node, so it comes back with its OWN row from the
    node table -- type 'a', w 10 -- and ``w`` stays int64. Pre-fix the seed was dropped by
    the label inner-merge and re-synthesized id-only by the endpoint backfill: type/w NaN and
    w upcast to float64. (polars declines all of these but label_node_hops; see below.)"""
    g = _attr_graph("pandas")
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, direction=direction, engine="pandas",
              **flags)
    nodes = _pd(r._nodes)
    assert 0 in nodes["id"].tolist(), "the seed is a traversed node and must be returned"
    row = nodes.loc[nodes["id"] == 0].iloc[0]
    assert row["type"] == "a"
    assert row["w"] == 10
    assert str(nodes["w"].dtype) == "int64", "attr dtype must not upcast via a NaN backfill"


def test_f2_min_hops_forward_seed_is_still_attribute_less_residue():
    """RESIDUE PIN (#1918 F2, deliberately NOT fixed) -- recorded as a value test, not a
    comment, so it cannot rot and so the day it changes is a loud test failure.

    ``min_hops>=2`` is the one track_hops flag whose seed still returns attribute-less. There
    the hop-label set is REBUILT from the retained-path backward walk and the node-output
    restriction to it is load-bearing: it is how a dead-end source-side branch leaves the
    answer. Widening it admits nodes with no retained incident edge and breaks the 400-case
    polars chain min_hops parity, whose node output deliberately mirrors this pandas stub
    (``hop_eager.py:_min_hops_labeled_node_output``). Lifting it is a decision about the
    CHAIN contract, not a local hop() bug fix.

    The seed's ROW is present (it is a retained-edge endpoint); only its attributes are NaN.
    Scoped to FORWARD (and reverse): the retained-path label rebuild runs for directed hops
    only, so UNDIRECTED min_hops takes the plain label-restriction path and is clean -- pinned
    positively just below, which also fixes the boundary of this residue in place.
    """
    g = _attr_graph("pandas")
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=2, max_hops=2, engine="pandas")
    nodes = _pd(r._nodes)
    row = nodes.loc[nodes["id"] == 0].iloc[0]
    assert pd.isna(row["type"]) and pd.isna(row["w"]), "residue: still id-only backfilled"


def test_f2_min_hops_undirected_seed_keeps_attributes():
    """The other side of the residue boundary: undirected min_hops re-reaches the seed, so it
    is hop-LABELED and comes back with its real attributes and an unupcast int64 column."""
    g = _attr_graph("pandas")
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=2, max_hops=2,
              direction="undirected", engine="pandas")
    nodes = _pd(r._nodes)
    row = nodes.loc[nodes["id"] == 0].iloc[0]
    assert (row["type"], row["w"]) == ("a", 10)
    assert str(nodes["w"].dtype) == "int64"


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "undirected"])
def test_f2_label_node_hops_seed_attributes_cross_engine(engine, direction):
    """The one track_hops flag polars serves: pandas and polars must agree on the seed row."""
    g = _attr_graph(engine)
    r = g.hop(nodes=_seeds(engine, [0]), hops=2, direction=direction,
              label_node_hops="nh", engine=engine)
    nodes = _pd(r._nodes)
    row = nodes.loc[nodes["id"] == 0].iloc[0]
    assert (row["type"], row["w"]) == ("a", 10)
    assert str(nodes["w"].dtype) == "int64"


def test_f2_dtype_matches_the_untracked_hop():
    """Differential form: turning a LABEL on must not change any non-label column's dtype."""
    g = _attr_graph("pandas")
    plain = g.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, engine="pandas")
    labeled = g.hop(nodes=pd.DataFrame({"id": [0]}), hops=2, label_node_hops="nh",
                    engine="pandas")
    for col in ("id", "type", "w"):
        assert str(_pd(plain._nodes)[col].dtype) == str(_pd(labeled._nodes)[col].dtype), col


# ============================================================================ F3
# An undirected edge traversed in both directions within one wavefront is still ONE edge.

@pytest.mark.parametrize("flags", [
    dict(label_edge_hops="eh"),
    dict(label_node_hops="nh"),
    dict(output_max_hops=1),
    dict(label_edge_hops="eh", label_node_hops="nh"),
], ids=["edge-label", "node-label", "output-window", "both-labels"])
def test_f3_undirected_edge_is_not_duplicated_under_hop_tracking(flags):
    # seeds {0,1} both incident to edge (0,1): it is traversed 0->1 AND 1->0 in hop 1.
    g = _attr_graph("pandas")
    r = g.hop(nodes=pd.DataFrame({"id": [0, 1]}), hops=1, direction="undirected",
              engine="pandas", **flags)
    edges = _pd(r._edges)
    assert len(edges) == 2, f"one row per EDGE, got {len(edges)}: {edges.to_dict('records')}"
    assert edge_ids(r) == [(0, 1), (1, 2)]


def test_f3_matches_the_unlabeled_edge_output():
    """Differential form: a labeling flag must not change the edge multiset."""
    g = _attr_graph("pandas")
    kw = dict(nodes=pd.DataFrame({"id": [0, 1]}), hops=1, direction="undirected",
              engine="pandas")
    assert edge_ids(g.hop(**kw, label_edge_hops="eh")) == edge_ids(g.hop(**kw))


# ============================================================================ F5
# label_seeds is a LABEL contract: it must not change membership.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "undirected"])
def test_f5_label_seeds_does_not_change_the_node_set(engine, direction):
    g = _attr_graph(engine)
    kw = dict(nodes=_seeds(engine, [0]), hops=2, direction=direction,
              return_as_wave_front=True, label_node_hops="nh", engine=engine)
    assert node_ids(g.hop(**kw, label_seeds=True)) == node_ids(g.hop(**kw, label_seeds=False))


def test_f5_wavefront_node_set_value():
    """Non-vacuity + the polars-agreeing value: forward wavefront from 0 on 0->1->2 reaches
    {1,2}; seed 0 is never re-encountered so it is stripped, label_seeds or not."""
    g = _attr_graph("pandas")
    kw = dict(nodes=pd.DataFrame({"id": [0]}), hops=2, return_as_wave_front=True,
              label_node_hops="nh", engine="pandas")
    assert node_ids(g.hop(**kw, label_seeds=True)) == [1, 2]
    assert node_ids(g.hop(**kw, label_seeds=False)) == [1, 2]


# ============================================================================ F6
# Contradictory bounds are a typed ValueError on EVERY engine.

_BAD_BOUNDS = [
    ("min-gt-max", dict(min_hops=2, max_hops=1)),
    ("min1-max0", dict(min_hops=1, max_hops=0)),
    ("negative-hops", dict(hops=-1)),
    ("negative-min", dict(min_hops=-1, hops=1)),   # polars used to ANSWER this one
    ("negative-max", dict(max_hops=-1)),
    ("negative-output-min", dict(hops=2, output_min_hops=-1)),
    ("negative-output-max", dict(hops=2, output_max_hops=-1)),
    ("output-min-gt-max", dict(hops=3, output_min_hops=2, output_max_hops=1)),
    ("output-min-gt-traversal", dict(hops=1, output_min_hops=2)),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("case_id,kw", _BAD_BOUNDS, ids=[i for i, _ in _BAD_BOUNDS])
def test_f6_contradictory_bounds_raise_value_error(engine, case_id, kw):
    g = _attr_graph(engine)
    with pytest.raises(ValueError):
        g.hop(nodes=_seeds(engine, [0]), engine=engine, **kw)


@pytest.mark.parametrize("engine", ENGINES)
def test_f6_negative_min_hops_is_not_silently_ignored(engine):
    """The worst F6 arm, called out on its own: polars consulted ``min_hops`` only when >1,
    so ``min_hops=-1, hops=1`` returned the ordinary 1-hop answer [0, 1] instead of raising."""
    g = _attr_graph(engine)
    with pytest.raises(ValueError, match="min_hops"):
        g.hop(nodes=_seeds(engine, [0]), min_hops=-1, hops=1, engine=engine)


# ============================================================================ F7
# hops=None is legal for Optional[int] and means "unbounded max" == run-to-closure, on BOTH
# engines. pandas was already run-to-closure; polars raised ValueError. Contract: pandas.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_f7_hops_none_runs_to_closure(engine, direction):
    g = _attr_graph(engine)
    r = g.hop(nodes=_seeds(engine, [0]), hops=None, direction=direction, engine=engine)
    ref = g.hop(nodes=_seeds(engine, [0]), to_fixed_point=True, direction=direction,
                engine=engine)
    assert node_ids(r) == node_ids(ref)
    assert edge_ids(r) == edge_ids(ref)


@pytest.mark.parametrize("engine", ENGINES)
def test_f7_hops_none_value(engine):
    """Non-vacuity: forward from 0 on 0->1->2 reaches the whole path, unlike hops=1."""
    g = _attr_graph(engine)
    assert node_ids(g.hop(nodes=_seeds(engine, [0]), hops=None, engine=engine)) == [0, 1, 2]
    assert node_ids(g.hop(nodes=_seeds(engine, [0]), hops=1, engine=engine)) == [0, 1]


# ============================================================================ F8
# A lower bound must not be starved by the reachable-set closure break.

def _cycle(engine, n):
    ndf = pd.DataFrame({"id": list(range(n))})
    edf = pd.DataFrame({"s": list(range(n)), "d": [(i + 1) % n for i in range(n)]})
    return graphistry.nodes(_frame(engine, ndf), "id").edges(_frame(engine, edf), "s", "d")


@pytest.mark.parametrize("min_hops,max_hops", [(4, 5), (4, 4), (5, 6), (6, 7)],
                         ids=["4-5", "4-4", "5-6", "6-7"])
def test_f8_directed_cycle_min_hops_beyond_closure(min_hops, max_hops):
    """Hand oracle on the directed 3-cycle 0->1->2->0 seeded at 0: directed WALKS of every
    length exist (the cycle re-enters), so a window whose lower bound exceeds the cycle
    length is satisfiable and the whole cycle is retained. Pre-fix the traversal broke at
    hop 3 -- when the reachable NODE SET stopped growing -- freezing max_reached_hop at 3, so
    the ``max_reached_hop < min_hops`` gate emptied every one of these windows.
    polars declines direct min_hops>1, so this is a pandas-lane pin."""
    g = _cycle("pandas", 3)
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=min_hops, max_hops=max_hops,
              engine="pandas")
    assert node_ids(r) == [0, 1, 2]
    assert edge_ids(r) == [(0, 1), (1, 2), (2, 0)]


def test_f8_does_not_widen_a_satisfiable_window():
    """Guard against over-fixing: windows whose bound the BFS already reached are UNCHANGED
    (the fix only defers the break while the bound is unmet). On the 4-path 0->1->2->3,
    min_hops=2 keeps the 2- and 3-edge paths and drops nothing extra."""
    ndf = pd.DataFrame({"id": [0, 1, 2, 3]})
    edf = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    g = graphistry.nodes(ndf, "id").edges(edf, "s", "d")
    r = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=2, max_hops=3, engine="pandas")
    assert node_ids(r) == [0, 1, 2, 3]
    assert edge_ids(r) == [(0, 1), (1, 2), (2, 3)]
    r2 = g.hop(nodes=pd.DataFrame({"id": [0]}), min_hops=4, max_hops=5, engine="pandas")
    assert node_ids(r2) == [] and edge_ids(r2) == [], "acyclic: no 4-walk exists, still empty"
