"""Round-003 hop()/chain() flag-matrix semantics pins (#1892).

Two contract invariants this file exists to hold:

1. FILTER-DOMAIN INVARIANCE: a filter's meaning must not depend on the hops
   value. ``source_node_match``/``destination_node_match`` match against the
   graph's node attribute table; which table they read must not flip when
   ``hops`` goes 1 -> 2, and a seed frame carrying a same-named attr column is
   never authoritative.
2. FIXED-POINT == SATURATED BOUNDED: ``to_fixed_point=True`` must return
   exactly what a bounded hop returns once ``hops`` exceeds the graph
   diameter, on both engines.

Boundary coverage for both invariants lives in
``test_hop_boundary_matrix.py`` (hand-computed oracles).

Layout mirrors the round-003 agent-02 proposal (T-01..T-07):
- value pins (T-01/02/03/04) -- the two HIGH classes
- NIE contract pins (T-05/06) -- pin today's typed declines so support can
  only land via a conscious flip to a value test, and pin the arms that
  already answer so they cannot regress to NIE
- green invariant pins (T-07) -- bright spots from the 984-cell sweep

Provenance: plans/gfql-release-amplification/rounds/round-003/findings/agent-02/
(findings.md F-01..F-04 + bright spots; fixtures reproduced verbatim).
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.predicates.is_in import IsIn

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

polars_only = pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")

ENGINES = ["pandas", pytest.param("polars", marks=polars_only)]


# ---------------------------------------------------------------- fixture
# F-01/F-02 repro graph: 0(a) -x-> 1(b) -y-> 2(a)
def _graph(engine: str):
    ndf = pd.DataFrame({"id": [0, 1, 2], "type": ["a", "b", "a"]})
    edf = pd.DataFrame({"s": [0, 1], "d": [1, 2], "rel": ["x", "y"]})
    if engine == "polars":
        return graphistry.nodes(pl.from_pandas(ndf), "id").edges(
            pl.from_pandas(edf), "s", "d")
    return graphistry.nodes(ndf, "id").edges(edf, "s", "d")


def _frame(engine: str, df: pd.DataFrame):
    return pl.from_pandas(df) if engine == "polars" else df


def _pd(df) -> pd.DataFrame:
    return df.to_pandas() if hasattr(df, "to_pandas") else df


def node_ids(g):
    return sorted(_pd(g._nodes)["id"].tolist())


def edge_ids(g):
    return sorted(map(tuple, _pd(g._edges)[["s", "d"]].itertuples(index=False)))


# ================================================================ T-01 (F-01a)
# Seeded source filter reads the NODE TABLE at every hops value; an id-only
# seed frame is the documented public shape ("id column matching g._node").
# Oracle: 0(a) -x-> 1(b); source filter type=='a' admits 0, so hop from seed
# {0} yields {0, 1} at both hops values.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", [1, 2])
def test_hop_seeded_source_match_id_only_seeds(engine, hops):
    g = _graph(engine)
    r = g.hop(
        nodes=_frame(engine, pd.DataFrame({"id": [0]})), hops=hops,
        source_node_match={"type": "a"}, engine=engine,
    )
    assert node_ids(r) == [0, 1]


# ================================================================ T-02 (F-01b)
# The node table is authoritative even when the seed frame shadows an attr
# column: seed says type 'b' (stale), node table says 'a'. Oracle: the table's
# 'a' admits node 0, so both hops values yield {0, 1}.

@pytest.mark.parametrize("engine", ENGINES)
def test_hop_seeded_source_match_domain_hops_invariant(engine):
    g = _graph(engine)
    seeds = pd.DataFrame({"id": [0], "type": ["b"]})  # stale attr; table says 'a'
    r1 = g.hop(nodes=_frame(engine, seeds), hops=1,
               source_node_match={"type": "a"}, engine=engine)
    r2 = g.hop(nodes=_frame(engine, seeds), hops=2,
               source_node_match={"type": "a"}, engine=engine)
    assert node_ids(r1) == node_ids(r2) == [0, 1]


# ================================================================ T-03 (F-02)
# to_fixed_point must equal the saturated bounded hop (engine-local, so it
# catches a seed leak without any cross-engine comparison). Graph diameter
# < 3, so hops=3 bounded is saturated and MUST equal tfp.
#
# Hand oracle for 0(a) -x-> 1(b) -y-> 2(a), undirected, wavefront, seeds {0, 1}:
#   source type=='a' -> sources may only be {0, 2}. From seed 0, edge x reaches
#     1; seed 1 is type 'b' so it cannot depart. Node 2 is never reached, so
#     edge y is never used. Encountered {1}, edges {(0,1)}. Seed 0 is not
#     re-encountered (returning needs edge x twice) -> nodes [1].
#   dest type=='b' -> destinations may only be {1}. From 0: x -> 1 (kept). From
#     1: x back to 0 and y to 2 are both type 'a' (rejected). Same answer:
#     nodes [1], edges [(0,1)].

# #1918 F4 WIDENING. The original parameterization was `filt` in {source_node_match,
# destination_node_match} x seeds fixed at [0, 1] -- and that is precisely what HID the
# disagreement: BOTH omitted arms (no filter, and a SINGLE seed) were the broken ones, where
# tfp answered [1,2] and bounded [0,1,2] for seeds=[0].
#
# WHICH ARM WAS WRONG: the BOUNDED one. tfp applied the edge-disjointness condition (see the
# hand oracle below); bounded returned whatever its BFS reached, and the BFS re-enters a seed
# by walking back along the edge it left by. An earlier pass at #1918 read the probe's
# "bounded=[0,1,2] vs tfp=[1,2]" as evidence against tfp and moved tfp -- making both arms
# agree on the wrong answer and breaking tests/compute/test_hop.py. The equivalence pin alone
# cannot catch that, which is why the value pin below is derived on paper instead.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("filt", [
    {},                                              # #1918 F4: the arm that was missing
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
], ids=["unfiltered", "src-match", "dst-match"])
@pytest.mark.parametrize("seeds", [[0], [1], [2], [0, 1]],
                         ids=["seed-0", "seed-1", "seed-2", "seeds-0-1"])
def test_hop_undirected_tfp_wavefront_matches_saturated_bounded(engine, filt, seeds):
    g = _graph(engine)
    kw = dict(
        nodes=_frame(engine, pd.DataFrame({"id": seeds})),
        direction="undirected", return_as_wave_front=True, engine=engine, **filt,
    )
    bounded = g.hop(hops=3, to_fixed_point=False, **kw)  # saturated: diameter < 3
    fixed = g.hop(hops=3, to_fixed_point=True, **kw)
    assert node_ids(fixed) == node_ids(bounded)
    assert edge_ids(fixed) == edge_ids(bounded)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("filt", [
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
], ids=["src-match", "dst-match"])
def test_hop_undirected_tfp_wavefront_filtered_values(engine, filt):
    kw = dict(
        nodes=_frame(engine, pd.DataFrame({"id": [0, 1]})),
        direction="undirected", return_as_wave_front=True, engine=engine, **filt,
    )
    g = _graph(engine)
    bounded = g.hop(hops=3, to_fixed_point=False, **kw)
    fixed = g.hop(hops=3, to_fixed_point=True, **kw)
    assert node_ids(fixed) == node_ids(bounded) == [1]
    assert edge_ids(fixed) == edge_ids(bounded) == [(0, 1)]


# The invariant above is self-checking (engine-local equality), so it could in principle be
# satisfied vacuously -- or, worse, by both arms agreeing on a WRONG answer, which is exactly
# what happened once here. So the literals below come from a HAND ORACLE, not from either arm.
#
# ORACLE (derived on paper, edge-disjoint-walk semantics). Fixture read undirected is the path
# 0 -x- 1 -y- 2. ``return_as_wave_front=True`` returns ENCOUNTERED nodes, and returning along
# the edge you departed on is the trip home, not an encounter -- so a walk may not REUSE an
# edge. Enumerating from each seed:
#   seed {0}: 0-x-1 (len 1), 0-x-1-y-2 (len 2). Getting back to 0 would need x twice.  -> {1,2}
#   seed {1}: 1-x-0 (len 1), 1-y-2 (len 1). Back to 1 would need x or y twice.          -> {0,2}
#   seed {2}: 2-y-1 (len 1), 2-y-1-x-0 (len 2).                                          -> {0,1}
#   seeds {0,1}: as above, plus 0 is reached from seed 1 over x and 1 from seed 0 over x --
#                one edge each, nothing reused, so BOTH seeds are genuinely encountered. -> {0,1,2}
# The path is acyclic, so a lone seed is never re-encountered; a second seed in the component
# changes that. Both are one rule: a seed stays iff an edge-disjoint walk reaches it.
@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("seeds,expect_nodes", [
    ([0], [1, 2]),
    ([1], [0, 2]),
    ([2], [0, 1]),
    ([0, 1], [0, 1, 2]),
], ids=["seed-0", "seed-1", "seed-2", "seeds-0-1"])
def test_hop_undirected_tfp_wavefront_unfiltered_values(engine, seeds, expect_nodes):
    g = _graph(engine)
    r = g.hop(nodes=_frame(engine, pd.DataFrame({"id": seeds})), hops=3,
              to_fixed_point=True, direction="undirected", return_as_wave_front=True,
              engine=engine)
    assert node_ids(r) == expect_nodes
    assert edge_ids(r) == [(0, 1), (1, 2)]


# ================================================================ T-04 (F-02)
# Cross-engine parity on the same shape: guards against the heuristic
# re-diverging after any fix, whichever engine the fix lands in first.

@polars_only
@pytest.mark.parametrize("filt", [
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
])
def test_hop_undirected_tfp_wavefront_cross_engine_parity(filt):
    kw = dict(direction="undirected", return_as_wave_front=True,
              to_fixed_point=True, hops=3, **filt)
    r_pd = _graph("pandas").hop(
        nodes=pd.DataFrame({"id": [0, 1]}), engine="pandas", **kw)
    r_pl = _graph("polars").hop(
        nodes=pl.from_pandas(pd.DataFrame({"id": [0, 1]})), engine="polars", **kw)
    assert node_ids(r_pd) == node_ids(r_pl)
    assert edge_ids(r_pd) == edge_ids(r_pl)


# ================================================================ T-05 (F-03)
# NIE contract pins: chain-polars declines e_undirected(to_fixed_point=True)
# with a typed NotImplementedError (no silent fallback on an explicit engine).
# Pin the CURRENT decline so support can only land by consciously flipping
# this to a value test -- and pin the neighboring arms that DO answer so they
# cannot regress to NIE.

@polars_only
def test_chain_polars_undirected_tfp_declines_typed_nie():
    g = _graph("polars")
    with pytest.raises(NotImplementedError, match="polars chain engine"):
        g.gfql([n(), e_undirected(to_fixed_point=True), n()], engine="polars")


@polars_only
def test_hop_polars_undirected_tfp_answers_and_matches_pandas():
    kw = dict(direction="undirected", to_fixed_point=True)
    r_pd = _graph("pandas").hop(
        nodes=pd.DataFrame({"id": [0]}), engine="pandas", **kw)
    r_pl = _graph("polars").hop(
        nodes=pl.from_pandas(pd.DataFrame({"id": [0]})), engine="polars", **kw)
    assert node_ids(r_pl) == node_ids(r_pd) == [0, 1, 2]
    assert edge_ids(r_pl) == edge_ids(r_pd)


@polars_only
@pytest.mark.parametrize("edge_op", [e_forward, e_reverse])
def test_chain_polars_directed_tfp_answers(edge_op):
    ops = [n(), edge_op(to_fixed_point=True), n()]
    r_pd = _graph("pandas").gfql(ops, engine="pandas")
    r_pl = _graph("polars").gfql(ops, engine="polars")
    assert node_ids(r_pl) == node_ids(r_pd) == [0, 1, 2]
    assert edge_ids(r_pl) == edge_ids(r_pd)


# ================================================================ T-06 (F-04)
# Cypher expressibility + decline-message audit pins.

@pytest.mark.parametrize("engine", ENGINES)
def test_cypher_whole_path_return_declines_typed(engine):
    # The canonical "give me the matched paths" statement mirroring
    # chain([n(), e_forward(), n()]) has no cypher spelling today; pin the
    # typed decline (with its #1273 pointer) on both engines.
    g = _graph(engine)
    with pytest.raises(GFQLValidationError) as exc_info:
        g.gfql("MATCH (a)-[e]->(b) RETURN a, e, b", engine=engine)
    msg = str(exc_info.value)
    assert "unsupported-cypher-query" in msg
    assert "#1273" in msg


def test_cypher_whole_entity_return_pandas_answers():
    out = _graph("pandas").gfql("MATCH (a)-[e]->(b) RETURN a, b", engine="pandas")
    assert len(_pd(out._nodes)) > 0


@polars_only
def test_cypher_whole_entity_return_polars_current_nie():
    # AUDIT NOTE (F-04): this graph is all-int64/str, yet the decline message
    # blames "float/temporal/nested/label/multi-entity columns" -- the gate
    # fires on data its message does not describe. When the gate is fixed or
    # narrowed, flip this pin to a row-level parity assertion vs pandas.
    g = _graph("polars")
    with pytest.raises(NotImplementedError, match="cypher result projection"):
        g.gfql("MATCH (a)-[e]->(b) RETURN a, b", engine="polars")


# ================================================================ T-07 greens
# Bright-spot invariants that held across the full 984-cell sweep.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("filt", [{}, {"edge_match": {"rel": "x"}}])
def test_tfp_hops_invariance(engine, direction, filt):
    # to_fixed_point=True must ignore the hops value entirely (invariant 2's
    # green half): identical node AND edge sets at hops=1 vs hops=3.
    g = _graph(engine)
    kw = dict(nodes=_frame(engine, pd.DataFrame({"id": [0, 2]})),
              direction=direction, to_fixed_point=True, engine=engine, **filt)
    r1 = g.hop(hops=1, **kw)
    r3 = g.hop(hops=3, **kw)
    assert node_ids(r1) == node_ids(r3)
    assert edge_ids(r1) == edge_ids(r3)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("direction", ["forward", "undirected"])
@pytest.mark.parametrize("seed_ids", [[], [99]], ids=["empty", "not-in-graph"])
def test_degenerate_seeds_clean(engine, direction, seed_ids):
    # Degenerate seeds return len-0 nodes AND edges with unchanged column sets
    # -- correct schema, no phantom rows.
    g = _graph(engine)
    seeds = pd.DataFrame({"id": pd.Series(seed_ids, dtype="int64")})
    r = g.hop(nodes=_frame(engine, seeds), hops=2, direction=direction,
              engine=engine)
    assert len(_pd(r._nodes)) == 0
    assert len(_pd(r._edges)) == 0
    assert list(_pd(r._nodes).columns) == ["id", "type"]
    assert list(_pd(r._edges).columns) == ["s", "d", "rel"]


@pytest.mark.parametrize("engine", ENGINES)
def test_hop_equals_chain(engine):
    # Cross-surface contract: direct hop(seeds, 2, wavefront=False) equals
    # chain([n(id-isin), e_forward(2), n()]) node and edge sets.
    g = _graph(engine)
    r_hop = g.hop(nodes=_frame(engine, pd.DataFrame({"id": [0]})), hops=2,
                  return_as_wave_front=False, engine=engine)
    r_chain = g.gfql(
        [n({"id": IsIn(options=[0])}), e_forward(hops=2), n()], engine=engine)
    assert node_ids(r_hop) == node_ids(r_chain)
    assert edge_ids(r_hop) == edge_ids(r_chain)
