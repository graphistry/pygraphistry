"""Round-003 hop()/chain() flag-matrix semantics pins (#1892).

Two contract invariants this file exists to hold:

1. FILTER-DOMAIN INVARIANCE: a filter's meaning must not depend on the hops
   value. ``source_node_match``/``destination_node_match`` are documented as
   matching against the graph's node attribute table; which table they read
   must not flip when ``hops`` goes 1 -> 2 (F-01: today the seeded hops==1
   special case filters the SEED frame instead -- an error on id-only seeds,
   a silent wrong answer when the seed frame shadows an attr column).
2. FIXED-POINT == SATURATED BOUNDED: ``to_fixed_point=True`` must return
   exactly what a bounded hop returns once ``hops`` exceeds the graph
   diameter (F-02: today the pandas undirected+tfp+wavefront seed-trim
   heuristic leaks never-re-encountered seeds, diverging from pandas' own
   bounded arms and from polars).

Layout mirrors the round-003 agent-02 proposal (T-01..T-07):
- strict xfail pins (T-01/02/03/04) -- the two HIGH classes; flipping one
  means fixing that #1892 instance (adjust the expectation, don't delete)
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
from graphistry.compute.exceptions import GFQLSchemaError, GFQLValidationError
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
# Seeded single-hop source filter must read the NODE TABLE, not the seed frame.
# An id-only seed frame is the documented public shape ("id column matching
# g._node"); today hops==1 resolves the filter against it and raises.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("hops", [
    pytest.param(1, marks=pytest.mark.xfail(
        strict=True, raises=GFQLSchemaError,
        reason="#1892: seeded hops==1 filters the id-only seed frame, not the "
               "node table -> [column-not-found] on 'type'")),
    2,
])
def test_hop_seeded_source_match_id_only_seeds(engine, hops):
    g = _graph(engine)
    r = g.hop(
        nodes=_frame(engine, pd.DataFrame({"id": [0]})), hops=hops,
        source_node_match={"type": "a"}, engine=engine,
    )
    assert node_ids(r) == [0, 1]


# ================================================================ T-02 (F-01b)
# Filter domain must not flip with hops when the seed frame shadows an attr
# column: seed says type 'b' (stale), node table says 'a'. Today hops=1 reads
# the seed frame (empty result), hops=2 reads the node table ([0, 1]) -- a
# SILENT wrong answer on one side. If the fix decides seed-frame attrs are
# authoritative instead, flip the expected value but KEEP the hops-invariance
# equality -- the invariance is this test's point.

@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#1892: filter domain flips with hops -- hops=1 consults the "
           "shadowing seed frame (empty), hops=2 the node table ([0, 1])")
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
# catches the seed-leak without any cross-engine comparison). Graph diameter
# < 3, so hops=3 bounded is saturated and MUST equal tfp. Today pandas tfp
# leaks seed 0 (never re-encountered under the filter: hopping from node 1,
# type 'b', is forbidden); pandas bounded and both polars arms agree on [1].

ENGINES_T03 = [
    pytest.param("pandas", marks=pytest.mark.xfail(
        strict=True, raises=AssertionError,
        reason="#1892: pandas undirected+tfp+wavefront seed-trim heuristic is "
               "topology-only and leaks filtered-out seeds")),
    pytest.param("polars", marks=polars_only),
]


@pytest.mark.parametrize("engine", ENGINES_T03)
@pytest.mark.parametrize("filt", [
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
])
def test_hop_undirected_tfp_wavefront_matches_saturated_bounded(engine, filt):
    g = _graph(engine)
    kw = dict(
        nodes=_frame(engine, pd.DataFrame({"id": [0, 1]})),
        direction="undirected", return_as_wave_front=True, engine=engine, **filt,
    )
    bounded = g.hop(hops=3, to_fixed_point=False, **kw)  # saturated: diameter < 3
    fixed = g.hop(hops=3, to_fixed_point=True, **kw)
    assert node_ids(fixed) == node_ids(bounded)
    assert edge_ids(fixed) == edge_ids(bounded)


# ================================================================ T-04 (F-02)
# Cross-engine parity on the same shape: guards against the heuristic
# re-diverging after any fix, whichever engine the fix lands in first.

@polars_only
@pytest.mark.parametrize("filt", [
    {"source_node_match": {"type": "a"}},
    {"destination_node_match": {"type": "b"}},
])
@pytest.mark.xfail(
    strict=True, raises=AssertionError,
    reason="#1892: pandas tfp keeps leaked seed 0, polars does not")
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
