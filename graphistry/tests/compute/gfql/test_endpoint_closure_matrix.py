"""#1888 endpoint closure, as an executable contract matrix.

THE CONTRACT (one rule; every surface, every engine, every policy state):

    With a node table BOUND, a pattern edge matches only if BOTH of its endpoints
    resolve to node rows. A node table SYNTHESIZED from the edges is VACUOUSLY
    CLOSED -- the gate must remove nothing there.

This file is the systematic form of that rule. ``test_known_cross_engine_divergences``
holds the original per-finding pins (F-01/F-02/F-03); this one sweeps the shape space:

  POSITIVE  closure holds on chain (1-hop / 2-hop / var-length), direct hop() in all
            three directions seeded and unseeded, Cypher property rows, Cypher
            count(*), and the #1658 index-backed seeded route.
  NEGATIVE  dangling source, dangling destination, both endpoints missing, ALL edges
            dangling, an EMPTY node table over non-empty edges, a self-loop on a
            missing node, and the vacuously-closed synthesized table that must NOT
            be gated.
  ENGINES   every case on pandas and polars; cudf / polars-gpu parametrized and
            SKIPPED with a named reason off-GPU (the dgx sweep is their gate).

ORACLES ARE HAND-COMPUTED from the fixture tables below and written as literals.
Engine agreement is NOT evidence and is never used as the expected value: a rule
applied wrongly in two places agrees with itself.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import n, e_forward, e_undirected

from .polars_test_utils import (
    edge_pair_set, gpu_environment_reason, node_id_set, to_pandas_any,
)

pl = pytest.importorskip("polars")


ALL_ENGINES = ["pandas", "polars", "cudf", "polars-gpu"]


def _require_engine(engine: str) -> None:
    """Skip with a NAMED reason so an absent GPU stack is visible in the report, not silent."""
    if engine == "cudf":
        pytest.importorskip("cudf", reason="cudf engine lane requires a GPU box (--gpus all)")
    if engine == "polars-gpu":
        pytest.importorskip("cudf", reason="polars-gpu lane requires a GPU box (--gpus all)")
        import importlib.util
        if importlib.util.find_spec("cudf_polars") is None:
            pytest.skip("polars-gpu lane requires cudf_polars (RAPIDS 26.02+ image)")


def _frame(pdf: pd.DataFrame, engine: str):
    if engine in ("polars", "polars-gpu"):
        return pl.from_pandas(pdf)
    if engine == "cudf":
        import cudf
        return cudf.from_pandas(pdf)
    return pdf


def _bind(engine: str, nodes: pd.DataFrame, edges: pd.DataFrame):
    return (graphistry
            .nodes(_frame(nodes, engine), "id")
            .edges(_frame(edges, engine), "s", "d"))


def _edges_only(engine: str, edges: pd.DataFrame):
    """A graph with NO node table bound -- nodes get synthesized from the edges."""
    return graphistry.edges(_frame(edges, engine), "s", "d")


def _seed(engine: str, ids):
    return _frame(pd.DataFrame({"id": list(ids)}), engine)


# --- FIXTURES + their hand-computed closure tables ------------------------------------------
#
# MIXED -- the workhorse. nodes {0,1,2}; the five edges cover every endpoint-miss shape:
#     (0,1)  both endpoints resolve      -> CLOSED, matches
#     (1,2)  both endpoints resolve      -> CLOSED, matches
#     (2,7)  destination 7 has no row    -> dangling DST, dropped
#     (8,0)  source 8 has no row         -> dangling SRC, dropped
#     (5,6)  neither endpoint has a row  -> BOTH missing, dropped
#   => closed edge set = {(0,1), (1,2)};  reachable node set = {0,1,2}
MIXED_NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
MIXED_EDGES = pd.DataFrame({"s": [0, 1, 2, 8, 5], "d": [1, 2, 7, 0, 6]})
MIXED_CLOSED_EDGES = {(0, 1), (1, 2)}

# CLEAN -- positive control: nothing dangles, so the gate must be a pure no-op.
CLEAN_NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
CLEAN_EDGES = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 0]})
CLEAN_CLOSED_EDGES = {(0, 1), (1, 2), (2, 0)}

# ALL_DANGLING -- every edge references ids with no node row => nothing survives.
ALLDANG_NODES = pd.DataFrame({"id": [0, 1], "v": [10, 20]})
ALLDANG_EDGES = pd.DataFrame({"s": [5, 7], "d": [6, 8]})

# EMPTY_NODES -- an empty (but bound, and correctly typed) node table over real edges.
# Bound-and-empty is still BOUND, so the closed answer is zero edges.
EMPTY_NODES = pd.DataFrame({"id": pd.Series([], dtype="int64"), "v": pd.Series([], dtype="int64")})
EMPTY_NODES_EDGES = pd.DataFrame({"s": [0, 1], "d": [1, 2]})

# SELF_LOOP -- (0,0) is closed; (9,9) is a self-loop on a node that does not exist.
# A self-loop is the case where "both endpoints" is one id, so a one-sided gate passes it.
SELFLOOP_NODES = pd.DataFrame({"id": [0], "v": [10]})
SELFLOOP_EDGES = pd.DataFrame({"s": [0, 9], "d": [0, 9]})

# SYNTHESIZED -- no node table bound. Vacuously closed: ALL five MIXED edges must match
# and the node set is every endpoint id. This is the case the gate must NOT touch.
SYNTH_ALL_EDGES = {(0, 1), (1, 2), (2, 7), (8, 0), (5, 6)}
SYNTH_ALL_NODES = {0, 1, 2, 5, 6, 7, 8}


def _rows(g):
    return to_pandas_any(g._nodes)


# --- POSITIVE: closure holds across every traversal surface ----------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_chain_single_hop_matches_only_closed_edges(engine):
    """chain n()-e_forward()-n(): the two closed edges, and no phantom node rows."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).gfql([n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == MIXED_CLOSED_EDGES
    assert node_id_set(out) == {0, 1, 2}


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_chain_two_hop_matches_only_closed_edges(engine):
    """Multi-hop: the only closed 2-path is 0->1->2, so the same two edges."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).gfql(
        [n(), e_forward(), n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == MIXED_CLOSED_EDGES
    assert node_id_set(out) == {0, 1, 2}


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_chain_var_length_matches_only_closed_edges(engine):
    """Variable-length arm: bounded expansion cannot reach through a dangling endpoint."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).gfql(
        [n(), e_forward(to_fixed_point=True), n()], engine=engine)
    assert edge_pair_set(out) == MIXED_CLOSED_EDGES
    assert node_id_set(out) == {0, 1, 2}


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_hop_unseeded_matches_only_closed_edges(engine, direction):
    """Direct hop(), all three directions, unseeded: every seed is a real node, and both
    closed edges are reachable in each direction, so the answer is direction-invariant."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(direction=direction, engine=engine)
    assert edge_pair_set(out) == MIXED_CLOSED_EDGES
    assert node_id_set(out) == {0, 1, 2}


# Seeded hop oracle table: (seed, direction) -> (closed edges, reached node ids).
# Hand-derived from MIXED's closed set {(0,1),(1,2)}; the empty rows are the load-bearing
# ones -- pre-gate, seed 2 forward returned (2,7) and seed 0 reverse returned (8,0).
_SEEDED_HOP_ORACLE = [
    (0, "forward", {(0, 1)}, {0, 1}),
    (2, "forward", set(), set()),          # only out-edge is the dangling (2,7)
    (0, "reverse", set(), set()),          # only in-edge is the dangling (8,0)
    (2, "reverse", {(1, 2)}, {1, 2}),
    (1, "undirected", MIXED_CLOSED_EDGES, {0, 1, 2}),
    (2, "undirected", {(1, 2)}, {1, 2}),   # (2,7) dropped, (1,2) kept
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("seed,direction,want_edges,want_nodes", _SEEDED_HOP_ORACLE)
def test_hop_seeded_matches_only_closed_edges(engine, seed, direction, want_edges, want_nodes):
    _require_engine(engine)
    g = _bind(engine, MIXED_NODES, MIXED_EDGES)
    out = g.hop(nodes=_seed(engine, [seed]), hops=1, direction=direction, engine=engine)
    assert edge_pair_set(out) == want_edges
    assert node_id_set(out) <= want_nodes | {seed}


# A SEED id with no node row is itself an unresolvable endpoint, so every edge incident to it
# is unclosed -- the source side of the rule, which a destination-only reading would pass.
# MIXED's phantom ids are {5,6,7,8}; each is an endpoint of exactly one edge.
_PHANTOM_SEED_ORACLE = [(8, "forward"), (8, "undirected"), (5, "forward"), (6, "undirected"),
                        (7, "reverse"), (0, "forward")]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("seed,direction", _PHANTOM_SEED_ORACLE)
def test_hop_seeded_on_an_id_with_no_node_row(engine, seed, direction):
    """Seed 0 is the positive control: a real node still traverses its closed edge."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(
        nodes=_seed(engine, [seed]), hops=1, direction=direction, engine=engine)
    want = {(0, 1)} if seed == 0 else set()
    assert edge_pair_set(out) == want
    assert node_id_set(out) == ({0, 1} if seed == 0 else set())


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_cypher_property_rows_only_closed_edges(engine):
    """Cypher row surface: one row per closed edge, carrying the REAL endpoint attributes."""
    _require_engine(engine)
    g = _bind(engine, MIXED_NODES, MIXED_EDGES)
    rows = _rows(g.gfql("MATCH (a)-[r]->(b) RETURN a.v AS av, b.v AS bv ORDER BY av", engine=engine))
    assert rows.to_dict("records") == [{"av": 10, "bv": 20}, {"av": 20, "bv": 30}]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_cypher_count_agrees_with_closed_edge_count(engine):
    """Cypher aggregate surface: count(*) is the SAME closed answer as the row surface."""
    _require_engine(engine)
    g = _bind(engine, MIXED_NODES, MIXED_EDGES)
    cnt = _rows(g.gfql("MATCH (a)-[]->(b) RETURN count(*) AS c", engine=engine))
    assert int(cnt["c"].iloc[0]) == len(MIXED_CLOSED_EDGES)


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_index_backed_seeded_hop_serves_the_closed_answer(engine):
    """#1658 index-backed route: the CSR gather is built from the RAW edge frame, so it must
    decline (and let the scan answer) rather than emit an edge whose endpoint has no node row."""
    _require_engine(engine)
    from graphistry.compute.gfql.index import gfql_index_edges

    try:
        g = gfql_index_edges(_bind(engine, MIXED_NODES, MIXED_EDGES))
        out_open = g.hop(nodes=_seed(engine, [0]), hops=1, direction="forward", engine=engine)
        out_dangling = g.hop(nodes=_seed(engine, [2]), hops=1, direction="forward", engine=engine)
    except Exception as ex:
        # The CSR gather is the first thing here to launch a real device kernel, so a box with
        # cudf importable but no CUDA runtime surfaces its breakage HERE and nowhere above.
        # Narrow classifier on purpose: anything not a known environment marker still FAILS.
        reason = gpu_environment_reason(ex)
        if reason is None:
            raise
        pytest.skip(f"index-backed lane needs a working GPU runtime: {reason}")
    assert edge_pair_set(out_open) == {(0, 1)}
    assert edge_pair_set(out_dangling) == set(), "index route emitted an unclosed edge"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_clean_graph_is_untouched_by_the_gate(engine):
    """POSITIVE CONTROL: with zero dangling endpoints the gate removes nothing and the node
    attributes keep their original dtype (no NaN-stub row, so no int64 -> float64 upcast)."""
    _require_engine(engine)
    out = _bind(engine, CLEAN_NODES, CLEAN_EDGES).gfql([n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == CLEAN_CLOSED_EDGES
    assert node_id_set(out) == {0, 1, 2}
    nodes_pdf = to_pandas_any(out._nodes)
    assert not nodes_pdf["v"].isna().any()
    assert str(nodes_pdf["v"].dtype) == "int64"


# --- NEGATIVE: each endpoint-miss shape, and the one case that must NOT be gated -------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("label,edges,want", [
    ("dangling_dst", pd.DataFrame({"s": [0, 1], "d": [1, 7]}), {(0, 1)}),
    ("dangling_src", pd.DataFrame({"s": [0, 8], "d": [1, 2]}), {(0, 1)}),
    ("both_missing", pd.DataFrame({"s": [0, 5], "d": [1, 6]}), {(0, 1)}),
])
def test_each_endpoint_miss_shape_is_dropped(engine, label, edges, want):
    """One shape per row: a miss on the SOURCE side, on the DESTINATION side, and on both.
    The gate has to be symmetric -- #1808 was exactly a one-sided (source-only) gate."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, edges).gfql([n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == want, label


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_all_edges_dangling_yields_empty_result(engine):
    """Total wipeout: no edge is closed, so both frames come back empty (not a crash,
    and not the ungated input)."""
    _require_engine(engine)
    out = _bind(engine, ALLDANG_NODES, ALLDANG_EDGES).gfql([n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == set()
    assert node_id_set(out) == set()


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_empty_but_bound_node_table_closes_everything_out(engine):
    """Bound-and-EMPTY is still bound: no id resolves, so no edge matches."""
    _require_engine(engine)
    out = _bind(engine, EMPTY_NODES, EMPTY_NODES_EDGES).gfql(
        [n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == set()
    assert node_id_set(out) == set()


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_self_loop_on_missing_node_is_dropped(engine):
    """A self-loop collapses both endpoints onto ONE id, so a gate that checks only one
    side still passes (9,9). Only (0,0) is closed."""
    _require_engine(engine)
    out = _bind(engine, SELFLOOP_NODES, SELFLOOP_EDGES).gfql(
        [n(), e_undirected(), n()], engine=engine)
    assert edge_pair_set(out) == {(0, 0)}
    assert node_id_set(out) == {0}


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_synthesized_node_table_is_vacuously_closed_and_not_gated(engine):
    """THE NEGATIVE OF THE GATE. With no node table bound the nodes are synthesized FROM the
    edges, so every endpoint resolves by construction. Gating here would delete real answers."""
    _require_engine(engine)
    out = _edges_only(engine, MIXED_EDGES).gfql([n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == SYNTH_ALL_EDGES
    assert node_id_set(out) == SYNTH_ALL_NODES


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_synthesized_node_table_hop_is_not_gated(engine):
    """Same vacuous-closure rule on the direct hop() kernel, which gates separately."""
    _require_engine(engine)
    out = _edges_only(engine, MIXED_EDGES).hop(direction="forward", engine=engine)
    assert edge_pair_set(out) == SYNTH_ALL_EDGES


# --- INVARIANCE: the closed answer cannot depend on which lane served it ----------------------

# Attaching a policy disables the fast paths, so the same query is served by a different
# executor. #1888 F-01 was exactly a value FLIP between those two lanes.
_POLICY = {"preload": (lambda ctx: None)}


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_closed_answer_is_invariant_to_the_serving_lane(engine):
    _require_engine(engine)
    g = _bind(engine, MIXED_NODES, MIXED_EDGES)
    ops = [n(), e_forward(), n()]
    fast = g.gfql(ops, engine=engine)
    policied = g.gfql(ops, engine=engine, policy=_POLICY)
    assert edge_pair_set(fast) == MIXED_CLOSED_EDGES
    assert edge_pair_set(policied) == MIXED_CLOSED_EDGES


# =============================================================================================
# ROUND-2 AMPLIFICATION -- the hop.py pos/neg boundary obligation (#1895 review).
#
# The section above sweeps the SHAPES of an endpoint miss on a 1-hop pattern. This section
# sweeps the AXES hop.py actually branches on, because both hop.py changes live on those
# branches: the closure gate sits next to the ``base_target_nodes`` construction (so it
# interacts with hop windows, direction, seeds, filters and the target wavefront), and the
# endpoint-backfill epilogue now runs ONLY for genuinely unbacked ids (so dtype preservation
# and the still-required backfill are both live contracts).
#
# ORACLES ARE HAND-COMPUTED from the fixture tables and written as literals. Engine agreement
# is never the oracle -- every expected value below is derived by walking the edge list by hand.
# =============================================================================================

# LADDER -- a directed path with a dangling id welded onto EACH end, so the gate must bite at
# the head and the tail of a multi-hop walk and at the hop-window saturation boundary.
#   nodes {0,1,2,3}
#   0->1, 1->2, 2->3   both endpoints resolve   -> CLOSED
#   3->9               destination 9 has no row -> dropped (tail)
#   4->0               source 4 has no row      -> dropped (head)
LADDER_NODES = pd.DataFrame({"id": [0, 1, 2, 3], "v": [10, 20, 30, 40]})
LADDER_EDGES = pd.DataFrame({"s": [0, 1, 2, 3, 4], "d": [1, 2, 3, 9, 0]})
LADDER_CLOSED = {(0, 1), (1, 2), (2, 3)}
LADDER_IDS = {0, 1, 2, 3}

# PARALLEL -- duplicate rows for one closed pair and one dangling pair. The gate is a row
# filter, so it must drop BOTH dangling rows and keep BOTH closed rows (row count, not just
# the pair set: a gate that deduplicated would pass a set-only assertion).
PARALLEL_NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
PARALLEL_EDGES = pd.DataFrame({"s": [0, 0, 2, 2], "d": [1, 1, 7, 7]})

# ISLANDS -- two disconnected closed components, one fully-dangling component, one isolated
# node (4) that no edge touches.
ISLANDS_NODES = pd.DataFrame({"id": [0, 1, 2, 3, 4], "v": [10, 20, 30, 40, 50]})
ISLANDS_EDGES = pd.DataFrame({"s": [0, 2, 5], "d": [1, 3, 6]})

# NO_EDGES -- a bound node table over an EMPTY (correctly typed) edge table.
NO_EDGES = pd.DataFrame({"s": pd.Series([], dtype="int64"), "d": pd.Series([], dtype="int64")})


def _assert_no_phantom_node_ids(out, bound_ids):
    """CLOSURE, stated on the OUTPUT: a hop may never surface an id the node table lacks.
    This is the assertion that fails loudest pre-gate -- a dangling endpoint used to arrive
    as a synthesized NaN-attribute node row."""
    assert node_id_set(out) <= bound_ids, f"phantom node ids: {node_id_set(out) - bound_ids}"


def _assert_output_is_endpoint_closed(out):
    """The OTHER half, and the reason the backfill epilogue still has to exist: every endpoint
    of every surviving edge must have a node row. Deleting the backfill breaks this."""
    edges = to_pandas_any(out._edges)
    nodes = to_pandas_any(out._nodes)
    if edges is None or len(edges) == 0:
        return
    ids = set(nodes["id"].tolist())
    assert set(edges["s"].tolist()) <= ids, "edge source has no node row"
    assert set(edges["d"].tolist()) <= ids, "edge destination has no node row"


# --- AXIS: hop window x direction, walked by hand on LADDER ------------------------------------
#
# (seed, direction, hops, expected closed edges). Derived by walking LADDER_EDGES:
#   forward from 0 : hop1 {(0,1)}  hop2 +{(1,2)}  hop3 +{(2,3)}  hop4 saturates -- 3->9 is
#                    dropped, so the 4th hop adds NOTHING (pre-gate it added (3,9) and node 9)
#   reverse from 3 : hop1 {(2,3)}  hop3 the whole path -- 4->0 is dropped, so the walk stops
#                    at 0 instead of continuing to the phantom 4
#   reverse from 0 : EMPTY -- 0's only in-edge is the dangling 4->0
#   undirected     : both ends bite at once
_LADDER_WINDOW_ORACLE = [
    (0, "forward", 1, {(0, 1)}),
    (0, "forward", 2, {(0, 1), (1, 2)}),
    (0, "forward", 3, LADDER_CLOSED),
    (0, "forward", 4, LADDER_CLOSED),          # saturated: the 4th hop would be 3->9
    (3, "reverse", 1, {(2, 3)}),
    (3, "reverse", 3, LADDER_CLOSED),
    (3, "reverse", 4, LADDER_CLOSED),          # saturated: the 4th hop would be 4->0
    (0, "reverse", 1, set()),                  # only in-edge is dangling 4->0
    (3, "forward", 1, set()),                  # only out-edge is dangling 3->9
    (0, "undirected", 1, {(0, 1)}),            # 4->0 dropped, so only the 0-1 side
    (3, "undirected", 1, {(2, 3)}),            # 3->9 dropped, so only the 2-3 side
    (0, "undirected", 3, LADDER_CLOSED),
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("seed,direction,hops,want", _LADDER_WINDOW_ORACLE)
def test_hop_window_never_reaches_through_a_dangling_endpoint(engine, seed, direction, hops, want):
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [seed]), hops=hops, direction=direction, engine=engine)
    assert edge_pair_set(out) == want
    _assert_no_phantom_node_ids(out, LADDER_IDS)
    _assert_output_is_endpoint_closed(out)


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("seed,direction,want", [
    (0, "forward", LADDER_CLOSED),
    (3, "reverse", LADDER_CLOSED),
    (0, "undirected", LADDER_CLOSED),
    (3, "undirected", LADDER_CLOSED),
])
def test_to_fixed_point_stops_at_the_closed_frontier(engine, seed, direction, want):
    """to_fixed_point runs the eager BFS loop (a different arm from the bounded single hop),
    so it needs its own cell: the fixed point must be the CLOSED component, not the raw one."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [seed]), to_fixed_point=True, direction=direction, engine=engine)
    assert edge_pair_set(out) == want
    _assert_no_phantom_node_ids(out, LADDER_IDS)


# --- AXIS: the TRAVERSAL min_hops window (NOT output_min_hops) ---------------------------------
#
# min_hops constrains the final TARGETS; intermediate hops still traverse. On LADDER forward
# from 0 the closed distances are 1->1, 2->2, 3->3, and the only hop-4 edge is the dangling 3->9:
#   min_hops=2 : targets {2,3}; the paths reaching them are the whole closed ladder
#   min_hops=3 : target {3};    same paths
#   min_hops=4 : NO closed target at all -- pre-gate this returned 3->9 and node 9
# Served on the pandas-lane engines; the polars hop declines min_hops with a typed NIE.
_MIN_HOPS_SERVED = ["pandas", "cudf"]
_MIN_HOPS_DECLINED = ["polars", "polars-gpu"]

_MIN_HOPS_ORACLE = [(2, LADDER_CLOSED), (3, LADDER_CLOSED), (4, set())]


@pytest.mark.parametrize("engine", _MIN_HOPS_SERVED)
@pytest.mark.parametrize("min_hops,want", _MIN_HOPS_ORACLE)
def test_min_hops_window_never_lands_on_a_dangling_target(engine, min_hops, want):
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0]), min_hops=min_hops, hops=4, direction="forward", engine=engine)
    assert edge_pair_set(out) == want
    _assert_no_phantom_node_ids(out, LADDER_IDS)


@pytest.mark.parametrize("engine,min_hops", [
    ("pandas", 2), ("pandas", 3), ("pandas", 4),
    ("cudf", 2), ("cudf", 3), ("cudf", 4),
])
def test_min_hops_window_output_excludes_the_seed_and_backs_everything_else(engine, min_hops):
    """#1918 F2 contract: seeds are hop 0, and hop 0 is labeled only under label_seeds, so a
    min_hops>=2 window excludes the seed's node ROW entirely (it does not come back as an
    attribute-less stub, and cuDF no longer diverges by accident). Every NON-seed endpoint of
    a surviving edge is still backed by its full-attribute node row."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0]), min_hops=min_hops, hops=4, direction="forward", engine=engine)
    ids = node_id_set(out)
    assert 0 not in ids, "min_hops>=2 excludes the unlabeled seed row"
    edges = to_pandas_any(out._edges)
    if edges is not None and len(edges) > 0:
        non_seed_endpoints = (set(edges["s"].tolist()) | set(edges["d"].tolist())) - {0}
        assert non_seed_endpoints <= ids, "non-seed endpoint with no node row"
    nodes = to_pandas_any(out._nodes)
    if len(nodes) > 0:
        assert not nodes.drop(columns=["id"]).isna().any().any(), "attribute-less stub row"


@pytest.mark.parametrize("engine", _MIN_HOPS_DECLINED)
def test_polars_lane_declines_min_hops_loudly(engine):
    _require_engine(engine)
    with pytest.raises(NotImplementedError, match="min_hops"):
        _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
            nodes=_seed(engine, [0]), min_hops=2, hops=4, direction="forward", engine=engine)


# --- AXIS: seed cardinality 0 / 1 / many / all -------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("label,seeds,want", [
    ("zero_seeds", [], set()),                                   # nothing to start from
    ("one_seed", [0], {(0, 1)}),
    ("many_seeds", [0, 2], {(0, 1), (2, 3)}),
    ("all_seeds", [0, 1, 2, 3], LADDER_CLOSED),                  # 3->9 still excluded
])
def test_seed_cardinality_boundaries_stay_closed(engine, label, seeds, want):
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, seeds), hops=1, direction="forward", engine=engine)
    assert edge_pair_set(out) == want, label
    _assert_no_phantom_node_ids(out, LADDER_IDS)


# --- AXIS: topology (parallel edges, disconnected, isolated, empty edge table) ------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_parallel_edges_are_gated_row_by_row(engine):
    """BOTH duplicate closed rows survive and BOTH duplicate dangling rows die: the gate is a
    row filter, not a dedup. A set-only assertion cannot tell those apart, so assert the count."""
    _require_engine(engine)
    out = _bind(engine, PARALLEL_NODES, PARALLEL_EDGES).gfql(
        [n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == {(0, 1)}
    assert len(to_pandas_any(out._edges)) == 2, "both parallel closed rows must survive"
    _assert_no_phantom_node_ids(out, {0, 1, 2})


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_disconnected_components_and_isolated_node(engine):
    """Two closed components survive independently, the all-dangling component vanishes, and
    the isolated node 4 never appears in an edge result."""
    _require_engine(engine)
    out = _bind(engine, ISLANDS_NODES, ISLANDS_EDGES).gfql(
        [n(), e_forward(), n()], engine=engine)
    assert edge_pair_set(out) == {(0, 1), (2, 3)}
    assert node_id_set(out) == {0, 1, 2, 3}
    _assert_output_is_endpoint_closed(out)


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_empty_edge_table_with_bound_nodes_is_empty_not_a_crash(engine, direction):
    """Degenerate boundary: the gate indexes an empty edge frame. Must return empty, and must
    not raise on the empty isin/join."""
    _require_engine(engine)
    out = _bind(engine, PARALLEL_NODES, NO_EDGES).hop(
        nodes=_seed(engine, [0]), hops=1, direction=direction, engine=engine)
    assert edge_pair_set(out) == set()


# --- AXIS: endpoint filters (source side / destination side) x closure -------------------------
#
# These are the cells where the gate and the FILTER domain interact. On MIXED, node 2 (v=30) has
# exactly one out-edge and it is the dangling (2,7); node 0 (v=10) has exactly one in-edge and it
# is the dangling (8,0). So each filter selects a real node whose entire adjacency is unclosed --
# the answer is EMPTY, where pre-gate it was the dangling edge itself.
@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("label,kwargs,want", [
    ("source_filter_selects_a_node_whose_only_out_edge_dangles",
     {"source_node_match": {"v": 30}, "direction": "forward"}, set()),
    ("destination_filter_selects_a_node_whose_only_in_edge_dangles",
     {"destination_node_match": {"v": 10}, "direction": "forward"}, set()),
    ("source_filter_on_a_closed_node_still_matches",
     {"source_node_match": {"v": 10}, "direction": "forward"}, {(0, 1)}),
    ("destination_filter_on_a_closed_node_still_matches",
     {"destination_node_match": {"v": 30}, "direction": "forward"}, {(1, 2)}),
])
def test_endpoint_filters_compose_with_closure(engine, label, kwargs, want):
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(hops=1, engine=engine, **kwargs)
    assert edge_pair_set(out) == want, label
    _assert_no_phantom_node_ids(out, {0, 1, 2})


# --- AXIS: WHICH endpoint each node filter binds to, in the loop that carries the gate ---------
#
# The polars single-hop chain fast path applies the closure semi-join and the node filters in
# ONE loop over (source_col, dest_col). The gate is symmetric, so it cannot detect a swapped
# from/to mapping -- but the FILTERS can, and only on `reverse`, where the pattern's left node
# binds to the edge DESTINATION. Round-5 mutation audit: collapsing the swap to `(n0, n2)`
# leaves the whole compute suite green while every reverse cell answers with the mirror edge set.
#
# Fixture RING: ids 0..3, kind 'A' on even ids and 'B' on odd; edges 0->1, 1->2, 2->3, 3->0.
# Hand-walked (a node filter constrains the node it is written on, never the edge direction):
#   [A]<-[e]-[ ]  a is the DESTINATION and must be A => dst in {0,2} => (3,0), (1,2)
#   [ ]<-[e]-[A]  b is the SOURCE and must be A      => src in {0,2} => (0,1), (2,3)
#   [A]<-[e]-[B]  dst in {0,2} AND src in {1,3}      => (3,0), (1,2)
#   [A]-[e]->[ ]  src in {0,2}                       => (0,1), (2,3)
#   [ ]-[e]->[A]  dst in {0,2}                       => (1,2), (3,0)
RING_NODES = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["A", "B", "A", "B"]})
RING_EDGES = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 0]})

_ENDPOINT_BINDING_ORACLE = [
    ("reverse_left_filter", "reverse", {"kind": "A"}, None, {(3, 0), (1, 2)}),
    ("reverse_right_filter", "reverse", None, {"kind": "A"}, {(0, 1), (2, 3)}),
    ("reverse_both_filters", "reverse", {"kind": "A"}, {"kind": "B"}, {(3, 0), (1, 2)}),
    ("forward_left_filter", "forward", {"kind": "A"}, None, {(0, 1), (2, 3)}),
    ("forward_right_filter", "forward", None, {"kind": "A"}, {(1, 2), (3, 0)}),
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("label,direction,left,right,want", _ENDPOINT_BINDING_ORACLE)
def test_node_filter_binds_to_the_endpoint_it_is_written_on(
    engine, label, direction, left, right, want
):
    _require_engine(engine)
    from graphistry.compute.ast import e_reverse

    edge = e_forward() if direction == "forward" else e_reverse()
    ops = [n(left) if left else n(), edge, n(right) if right else n()]
    out = _bind(engine, RING_NODES, RING_EDGES).gfql(ops, engine=engine)
    assert edge_pair_set(out) == want, label
    endpoints = {i for pair in want for i in pair}
    assert node_id_set(out) == endpoints, label


# --- AXIS: the backfill epilogue (the SECOND hop.py change) ------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_direct_hop_keeps_node_attribute_dtypes(engine, direction):
    """THE cell for the backfill change, aimed at the surface it lives on. Pre-gate, direct
    hop() on MIXED returned a NaN-attribute row for id 7 and upcast v from int64 to float64
    (the chain surface filtered its own node frame and so never showed this). Closure plus
    backfill-only-when-actually-missing means the attribute column comes back untouched."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(
        hops=1, direction=direction, engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert not nodes_pdf["v"].isna().any(), "a phantom NaN-attribute node row survived"
    assert str(nodes_pdf["v"].dtype) == "int64", "attribute dtype was upcast by a stub row"
    _assert_no_phantom_node_ids(out, {0, 1, 2})


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_chain_surface_keeps_node_attribute_dtypes(engine):
    """Same contract on the chain surface. This one already held pre-gate (chain filtered its
    node frame independently), so it is a guard against the gate REGRESSING it, not a fix pin."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).gfql([n(), e_forward(), n()], engine=engine)
    nodes_pdf = to_pandas_any(out._nodes)
    assert not nodes_pdf["v"].isna().any()
    assert str(nodes_pdf["v"].dtype) == "int64"


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
@pytest.mark.parametrize("hops", [1, 2])
def test_every_surviving_edge_still_has_both_node_rows(engine, direction, hops):
    """The backfill epilogue is now conditional, so pin what it still owes: whatever edges come
    back, their endpoints are present in the node frame. Deleting the backfill breaks this even
    though closure alone would not."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0, 3]), hops=hops, direction=direction, engine=engine)
    _assert_output_is_endpoint_closed(out)
    _assert_no_phantom_node_ids(out, LADDER_IDS)


# --- AXIS: zero-hop seed inclusion -------------------------------------------------------------
#
# include_zero_hop_seed is a PANDAS-LANE kwarg: the polars hop declines it with
# NotImplementedError (its documented parity-or-NIE contract). Both halves are pinned below --
# the closed answer where it is served, and the loud decline where it is not. A silent wrong
# answer on the polars lane is exactly what the NIE cell exists to catch.
_ZERO_HOP_SEED_SERVED = ["pandas", "cudf"]
_ZERO_HOP_SEED_DECLINED = ["polars", "polars-gpu"]

# CONTROL fixture: a graph with ZERO dangling endpoints, so the gate cannot be involved.
ISOLATED_SEED_NODES = pd.DataFrame({"id": [0, 1, 4], "v": [10, 20, 50]})
ISOLATED_SEED_EDGES = pd.DataFrame({"s": [0], "d": [1]})


@pytest.mark.parametrize("engine", _ZERO_HOP_SEED_SERVED)
def test_zero_hop_seed_does_not_smuggle_back_a_dangling_edge(engine):
    """include_zero_hop_seed must not reintroduce the seed's unclosed edge. Seed 2's only
    out-edge is the dangling (2,7), so the edge answer stays empty with the flag set."""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(
        nodes=_seed(engine, [2]), hops=1, direction="forward",
        include_zero_hop_seed=True, engine=engine)
    assert edge_pair_set(out) == set()
    _assert_no_phantom_node_ids(out, {0, 1, 2})


@pytest.mark.parametrize("engine", _ZERO_HOP_SEED_SERVED)
@pytest.mark.parametrize("include_zero_hop_seed", [False, True])
def test_unreached_seed_is_dropped_on_a_fully_closed_graph_too(engine, include_zero_hop_seed):
    """CONTROL for the cell above: a seed that reaches nothing comes back with an EMPTY node
    frame on direct hop() even with the flag set, on a graph where NOTHING dangles. Pinned so
    the empty node frame in the dangling case is never misread as closure eating the seed."""
    _require_engine(engine)
    out = _bind(engine, ISOLATED_SEED_NODES, ISOLATED_SEED_EDGES).hop(
        nodes=_seed(engine, [4]), hops=1, direction="forward",
        include_zero_hop_seed=include_zero_hop_seed, engine=engine)
    assert edge_pair_set(out) == set()
    assert node_id_set(out) == set(), "pre-existing hop() behaviour, independent of #1888"


@pytest.mark.parametrize("engine", _ZERO_HOP_SEED_DECLINED)
def test_polars_lane_declines_zero_hop_seed_loudly(engine):
    """Parity-or-NIE: the polars hop must REFUSE the kwarg, not quietly ignore it and return a
    differently-shaped answer than the pandas lane."""
    _require_engine(engine)
    with pytest.raises(NotImplementedError, match="include_zero_hop_seed"):
        _bind(engine, MIXED_NODES, MIXED_EDGES).hop(
            nodes=_seed(engine, [2]), hops=1, direction="forward",
            include_zero_hop_seed=True, engine=engine)


# =============================================================================================
# AMPLIFICATION ROUND 2 -- axes round 1 did not touch.
#
# Round 1 swept seeds / topology / direction / hop count / endpoint filters. Round 2 goes after
# the remaining hop.py branches that sit downstream of the gate and the backfill: the OUTPUT
# hop window, hop LABELLING (which reaches the backfill's node_hop_records merge -- a branch
# that only executes when something is genuinely unbacked), wavefront-mode output, a CYCLE
# (to_fixed_point termination), a non-integer id dtype, an edge-attribute filter, and the
# target wavefront that widens the resolvable-id universe.
# =============================================================================================

# CYCLE -- a closed 3-cycle with a dangling edge hanging off each of two nodes. to_fixed_point
# must terminate on the cycle and must not step onto 9 or admit 8.
CYCLE_NODES = pd.DataFrame({"id": [0, 1, 2], "v": [10, 20, 30]})
CYCLE_EDGES = pd.DataFrame({"s": [0, 1, 2, 2, 8], "d": [1, 2, 0, 9, 0]})
CYCLE_CLOSED = {(0, 1), (1, 2), (2, 0)}

# STRING IDS -- the gate is an id-membership test, so it has to hold on a non-integer id dtype.
STR_NODES = pd.DataFrame({"id": ["a", "b", "c"], "v": [10, 20, 30]})
STR_EDGES = pd.DataFrame({"s": ["a", "b", "c", "yy"], "d": ["b", "c", "zz", "a"]})
STR_CLOSED = {("a", "b"), ("b", "c")}

# TYPED LADDER -- LADDER plus an edge attribute, for the edge_match x closure cell.
TYPED_LADDER_EDGES = pd.DataFrame({
    "s": [0, 1, 2, 3, 4],
    "d": [1, 2, 3, 9, 0],
    "t": ["x", "y", "y", "x", "x"],
})


# --- AXIS: the OUTPUT hop window ---------------------------------------------------------------
#
# Walked by hand on LADDER from seed 0 forward with max_hops=4. Hop k contributes the k-th edge
# of the path: hop1 (0,1), hop2 (1,2), hop3 (2,3), hop4 would be (3,9) -- which the gate deletes,
# so hop 4 contributes NOTHING. output_min_hops=4 is therefore EMPTY where pre-gate it was
# exactly the dangling edge: the sharpest cell in this file.
# Served on the pandas-lane engines only; the polars hop declines output windows with NIE
# (pinned separately below).
_OUTPUT_WINDOW_SERVED = ["pandas", "cudf"]
_OUTPUT_WINDOW_DECLINED = ["polars", "polars-gpu"]

# (kwargs, expected edges, expected node frame). The node frame is the ENDPOINT SET of the
# retained edges -- both sides -- which is what the backfill epilogue is for: the sliced edge's
# SOURCE is not itself a hop-window survivor, so only the backfill puts its node row back.
_OUTPUT_WINDOW_ORACLE = [
    ({"output_max_hops": 1}, {(0, 1)}, {0, 1}),
    ({"output_max_hops": 2}, {(0, 1), (1, 2)}, {0, 1, 2}),
    ({"output_max_hops": 3}, LADDER_CLOSED, LADDER_IDS),
    ({"output_max_hops": 4}, LADDER_CLOSED, LADDER_IDS),   # hop 4 is the dangling 3->9
    ({"output_min_hops": 2}, {(1, 2), (2, 3)}, {1, 2, 3}),
    ({"output_min_hops": 3}, {(2, 3)}, {2, 3}),
    ({"output_min_hops": 4}, set(), set()),                # ONLY the dangling edge lived at hop 4
]


@pytest.mark.parametrize("kwargs,want,want_nodes", _OUTPUT_WINDOW_ORACLE)
def test_output_hop_window_node_frame_is_the_closed_endpoint_set(kwargs, want, want_nodes):
    """PANDAS ONLY, deliberately: this is the cell that pins the endpoint BACKFILL. Under an
    output window the surviving edge's source is not itself in the window, so without the
    backfill the node frame comes back missing it. cuDF does not backfill here -- that
    divergence is pre-existing and pinned in test_known_cross_engine_divergences.py."""
    out = _bind("pandas", LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed("pandas", [0]), max_hops=4, direction="forward", engine="pandas", **kwargs)
    assert edge_pair_set(out) == want
    assert node_id_set(out) == want_nodes
    _assert_output_is_endpoint_closed(out)


@pytest.mark.parametrize("engine", _OUTPUT_WINDOW_SERVED)
@pytest.mark.parametrize("kwargs,want,want_nodes", _OUTPUT_WINDOW_ORACLE)
def test_output_hop_window_slices_only_closed_edges(engine, kwargs, want, want_nodes):
    # NOTE: no _assert_output_is_endpoint_closed here. Under an output window the cudf lane
    # returns the sliced edge WITHOUT backfilling its source node row, where pandas backfills.
    # That divergence is PRE-EXISTING (reproduces identically at this branch's merge-base) and
    # is pinned as a strict xfail in test_known_cross_engine_divergences.py.
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0]), max_hops=4, direction="forward", engine=engine, **kwargs)
    assert edge_pair_set(out) == want
    _assert_no_phantom_node_ids(out, LADDER_IDS)


@pytest.mark.parametrize("engine", _OUTPUT_WINDOW_DECLINED)
@pytest.mark.parametrize("kwarg", ["output_min_hops", "output_max_hops"])
def test_polars_lane_declines_output_hop_windows_loudly(engine, kwarg):
    """Parity-or-NIE: declining loudly is the contract, so a future polars implementation
    cannot land silently returning the ungated window."""
    _require_engine(engine)
    with pytest.raises(NotImplementedError, match=kwarg):
        _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
            nodes=_seed(engine, [0]), max_hops=4, direction="forward",
            engine=engine, **{kwarg: 2})


# --- AXIS: hop labelling (reaches the backfill's node_hop_records merge) ------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_hop_labels_carry_no_phantom_row_and_stay_integral(engine):
    """The backfill epilogue merges node_hop_records into the ids it appends. Pre-gate that
    appended a row for id 9 with a NULL hop label, which also forced the label column off int64.
    With closure there is nothing unbacked to append, so every labelled node is a real one and
    the label column stays integral."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0]), max_hops=4, direction="forward",
        label_node_hops="nh", label_seeds=True, engine=engine)
    assert edge_pair_set(out) == LADDER_CLOSED
    nodes_pdf = to_pandas_any(out._nodes)
    assert "nh" in nodes_pdf.columns
    assert not nodes_pdf["nh"].isna().any(), "a phantom node arrived with a null hop label"
    # INTEGRAL, not a specific spelling: the lanes legitimately land on int64 vs nullable Int64.
    # What the null-stub row used to do is force the column to FLOAT, which this still catches.
    assert pd.api.types.is_integer_dtype(nodes_pdf["nh"]), "null label row forced the column off an integer dtype"
    _assert_no_phantom_node_ids(out, LADDER_IDS)


# --- AXIS: wavefront-mode output ---------------------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("seed,hops,want_edges,want_nodes", [
    (0, 1, {(0, 1)}, {1}),
    # hops=2 reaches 1 then 2; wavefront mode returns the ids REACHED (seed excluded), not
    # only the last ring -- so {1,2}. The closure claim here is that 9 is in neither.
    (0, 2, {(0, 1), (1, 2)}, {1, 2}),
    (3, 1, set(), set()),          # 3's only out-edge is the dangling 3->9
])
def test_wave_front_mode_returns_only_closed_frontier(engine, seed, hops, want_edges, want_nodes):
    """return_as_wave_front drops the seed from the node output, so it is a separate epilogue
    arm from every cell above."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [seed]), hops=hops, direction="forward",
        return_as_wave_front=True, engine=engine)
    assert edge_pair_set(out) == want_edges
    assert node_id_set(out) == want_nodes


# --- AXIS: cycle topology + to_fixed_point termination -----------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "undirected"])
def test_fixed_point_on_a_cycle_terminates_at_the_closed_component(engine, direction):
    """A cycle makes to_fixed_point revisit nodes, which is where an over-eager gate or a
    broken visited-set would loop or under-collect. The closed component is the whole cycle;
    2->9 and 8->0 are not part of it."""
    _require_engine(engine)
    out = _bind(engine, CYCLE_NODES, CYCLE_EDGES).hop(
        nodes=_seed(engine, [0]), to_fixed_point=True, direction=direction, engine=engine)
    assert edge_pair_set(out) == CYCLE_CLOSED
    assert node_id_set(out) == {0, 1, 2}
    _assert_output_is_endpoint_closed(out)


# --- AXIS: id dtype ----------------------------------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction", ["forward", "reverse", "undirected"])
def test_closure_holds_on_string_ids(engine, direction):
    """The gate is an id-membership test; string ids exercise a different comparison path than
    the int64 fixtures above (and a different join-key alignment on the polars lane)."""
    _require_engine(engine)
    out = _bind(engine, STR_NODES, STR_EDGES).hop(direction=direction, engine=engine)
    assert edge_pair_set(out) == STR_CLOSED
    assert node_id_set(out) == {"a", "b", "c"}
    _assert_output_is_endpoint_closed(out)


# --- AXIS: edge-attribute filter x closure -----------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("match,want", [
    ({"t": "x"}, {(0, 1)}),        # t='x' also selects the dangling 3->9 and 4->0 -- both die
    ({"t": "y"}, {(1, 2), (2, 3)}),
])
def test_edge_match_composes_with_closure(engine, match, want):
    """edge_match filters edges BEFORE the gate, so this pins that the two filters compose
    rather than one shadowing the other."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, TYPED_LADDER_EDGES).hop(
        to_fixed_point=True, direction="forward", edge_match=match, engine=engine)
    assert edge_pair_set(out) == want
    _assert_no_phantom_node_ids(out, LADDER_IDS)


# --- AXIS: the target wavefront widens the resolvable-id universe ------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_target_wave_front_inside_the_node_table_keeps_closure(engine):
    """The ordinary chain case: the wavefront is a SUBSET of the node table, so it constrains
    the final target and changes nothing about closure."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [0]), hops=1, direction="forward",
        target_wave_front=_seed(engine, [1]), engine=engine)
    assert edge_pair_set(out) == {(0, 1)}
    _assert_no_phantom_node_ids(out, LADDER_IDS)


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_target_wave_front_cannot_resurrect_an_unrelated_dangling_edge(engine):
    """A wavefront naming id 1 does not make id 9 resolvable: 3->9 stays dropped."""
    _require_engine(engine)
    out = _bind(engine, LADDER_NODES, LADDER_EDGES).hop(
        nodes=_seed(engine, [3]), hops=1, direction="forward",
        target_wave_front=_seed(engine, [1]), engine=engine)
    assert edge_pair_set(out) == set()
    _assert_no_phantom_node_ids(out, LADDER_IDS)


@pytest.mark.parametrize("kwargs,want_edges,want_nodes", [
    ({"output_max_hops": 1}, {(0, 1)}, {0, 1}),
    ({"output_max_hops": 2}, {(0, 1), (1, 2)}, {0, 1, 2}),
])
def test_output_window_on_a_cycle_keeps_only_the_windowed_endpoints(kwargs, want_edges, want_nodes):
    """PANDAS ONLY (see the cuDF divergence note above). A CYCLE is the case where the seed is
    re-reached, which routes the hop-label bookkeeping down a different arm than the acyclic
    LADDER: node 2 is reachable but is NOT an endpoint of any windowed edge, so it must not
    appear. Pins that the hop-label column still reaches the output-window node filter."""
    out = _bind("pandas", CYCLE_NODES, CYCLE_EDGES).hop(
        nodes=_seed("pandas", [0]), max_hops=4, direction="forward", engine="pandas", **kwargs)
    assert edge_pair_set(out) == want_edges
    assert node_id_set(out) == want_nodes
    _assert_output_is_endpoint_closed(out)


# =============================================================================================
# AMPLIFICATION ROUND 4 -- the surfaces master gained AFTER rounds 1-3 were written.
#
# The branch was rebased onto master carrying #1894 (OPTIONAL MATCH null-extension) and #1893
# (hop filter-domain / to_fixed_point saturation). Neither had ever been exercised together
# with the endpoint-closure gate.
#
# ANTI-VACUITY, measured at the merge-base (526976e9): of the 8 cells below that run on this
# box, 5 FAIL there -- both OPTIONAL MATCH cells on pandas and on cudf, and the label cell on
# cudf. The other 3 are named CONTROLS: polars already applied closure on the OPTIONAL MATCH
# surface (it was the correct side of #1808), and pandas never raised on the missing label
# column because pandas .loc creates one.
# =============================================================================================

# --- AXIS: OPTIONAL MATCH x closure ------------------------------------------------------------
#
# openCypher: OPTIONAL MATCH keeps every driving row and NULL-binds the optional aliases when
# the pattern finds no match. Endpoint closure says a pattern edge matches only when BOTH
# endpoints resolve to node rows. Composed on MIXED, the two rules answer one question: a row
# whose only candidate edge is DANGLING is an unmatched row, so it is NULL-extended -- it is
# never bound to the unbacked id.

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_optional_match_null_extends_a_dangling_endpoint(engine):
    """MIXED, hand-walked. Driving rows are the node table {0, 1, 2}. Closed out-edges:
    0 -> (0,1), 1 -> (1,2), 2 -> only the dangling (2,7). So 2 is the unmatched row and binds
    b to NULL; it must NOT bind b to 7."""
    _require_engine(engine)
    rows = _rows(_bind(engine, MIXED_NODES, MIXED_EDGES).gfql(
        "MATCH (a) OPTIONAL MATCH (a)-[r]->(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid",
        engine=engine))
    got = [(int(r["aid"]), None if pd.isna(r["bid"]) else int(r["bid"]))
           for r in rows.to_dict("records")]
    assert got == [(0, 1), (1, 2), (2, None)]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_optional_match_undirected_null_extends_a_dangling_endpoint(engine):
    """Same rule on the UNDIRECTED optional pattern, which reaches an endpoint the forward
    cell never touches: id 0 is the DESTINATION of the dangling (8,0). Closed undirected
    adjacency on MIXED is 0-1 and 1-2, so 0 has neighbour {1} (not 8) and 2 has neighbour {1}
    (not 7)."""
    _require_engine(engine)
    rows = _rows(_bind(engine, MIXED_NODES, MIXED_EDGES).gfql(
        "MATCH (a) OPTIONAL MATCH (a)-[r]-(b) RETURN a.id AS aid, b.id AS bid ORDER BY aid, bid",
        engine=engine))
    got = [(int(r["aid"]), None if pd.isna(r["bid"]) else int(r["bid"]))
           for r in rows.to_dict("records")]
    assert got == [(0, 1), (1, 0), (1, 2), (2, 1)]


# --- AXIS: the hop-label column under the undirected seed strip --------------------------------
#
# The gate made the endpoint backfill CONDITIONAL (it appends only ids the node table does not
# back), and the backfill is also where the hop-label column used to get materialized. On a
# result with no surviving edges there is nothing to backfill, so the undirected seed strip --
# which writes NA into that column for every seed -- is the first writer. pandas creates a
# column on assignment; cuDF raises. Hence the column has to exist before the strip writes it.

@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_undirected_zero_hop_seed_under_an_output_window_labels_before_it_strips(engine):
    """MIXED, seed 0, hops=0 with include_zero_hop_seed and an output window (the window is
    what turns hop LABELLING on without a user-visible label column, so the column is internal
    and gets dropped before this point).

    Hand-walked: hops=0 traverses no edge, so the edge set is empty. The undirected arm then
    sets every seed's hop label to NA, and the output window keeps a node iff its label is in
    the window OR it is an endpoint of a surviving edge -- NA satisfies neither and there are
    no edges, so the node set is empty too. The VALUE is the pandas oracle's; what this cell
    pins is that cuDF reaches it instead of raising on the missing label column.

    (The polars lane declines include_zero_hop_seed with a typed NIE --
    test_polars_lane_declines_zero_hop_seed_loudly.)"""
    _require_engine(engine)
    out = _bind(engine, MIXED_NODES, MIXED_EDGES).hop(
        nodes=_seed(engine, [0]), hops=0, direction="undirected",
        include_zero_hop_seed=True, output_min_hops=0, engine=engine)
    assert edge_pair_set(out) == set()
    assert node_id_set(out) == set()


# =============================================================================================
# THE NULL ENDPOINT CONTRACT -- one rule, every surface, every engine.
#
#     A NULL id is NOT an identity. An edge with a NULL endpoint therefore matches no pattern
#     edge -- on any surface, approached from either side, whether or not the node table holds
#     a NULL-id row. A NULL-id node ROW is still a row: node-only patterns count it.
#
# WHY THIS RULE. openCypher evaluates identity comparison under three-valued logic, so
# ``null = null`` is UNKNOWN, never TRUE; a pattern edge binds two node identities, and an
# endpoint that cannot be shown equal to any node identity binds nothing. The rule is also the
# only one that makes an engine agree with ITSELF: with a NULL endpoint linkable, the pandas
# ``count(*)`` over ``(a)-[x]-(b)`` answered 6 while the same pattern as a chain returned 2
# edges (4 orientations), and the polars hop kept the NULL-endpoint edge while its node output
# dropped the NULL row, so the result frame referenced a node that was not in it.
#
# The rule is NOT "delete NULL-id node rows": ``test_a_null_id_node_row_is_still_a_row`` pins
# that a NULL-id row still exists and is still counted by a node-only pattern, and the
# OPTIONAL MATCH cells above pin that a NULL BINDING is still produced. It constrains edge
# endpoint RESOLUTION only.
#
# The spec statement lives in docs/source/gfql/spec/language.md ("NULL node ids and edge
# endpoints"); this section is its executable form.
#
# FIXTURES. NULLEP puts a NULL on BOTH endpoint sides, so a one-sided implementation fails:
#     nodes  id = [0.0, 1.0, 2.0, NULL]
#     edges  (0,1) (1,2) (NULL,2) (2,NULL)
# Contract answer: the matched edge set is {(0,1),(1,2)} and the reached node set is {0,1,2}.
# NULLFREE is the same SHAPE with 3 substituted for every NULL -- the anti-vacuity control that
# says these cells are not just asserting "small answer" on a broken fixture.
# =============================================================================================

NULL_ENDPOINT_NODES = pd.DataFrame({"id": [0.0, 1.0, 2.0, None]})
NULL_ENDPOINT_EDGES = pd.DataFrame({"s": [0.0, 1.0, None, 2.0], "d": [1.0, 2.0, 2.0, None]})
#: The only edges a pattern may match over NULLEP -- the two with no NULL endpoint.
_NULL_ENDPOINT_MATCHED = {(0.0, 1.0), (1.0, 2.0)}
_NULL_ENDPOINT_REACHED = {0.0, 1.0, 2.0}

#: CONTROL: NULLEP with every NULL replaced by the real id 3. Same shape, every edge matches.
NULL_FREE_TWIN_NODES = pd.DataFrame({"id": [0.0, 1.0, 2.0, 3.0]})
NULL_FREE_TWIN_EDGES = pd.DataFrame({"s": [0.0, 1.0, 3.0, 2.0], "d": [1.0, 2.0, 2.0, 3.0]})
_NULL_FREE_TWIN_MATCHED = {(0.0, 1.0), (1.0, 2.0), (3.0, 2.0), (2.0, 3.0)}

#: A NULL in an OBJECT id column: pandas ``isna`` on object, polars ``is_not_null`` on Utf8 and
#: cuDF's string-column null mask are three different comparison paths from the float fixtures.
NULL_STR_NODES = pd.DataFrame({"id": ["a", "b", "c", None]})
NULL_STR_EDGES = pd.DataFrame({"s": ["a", "b", None, "c"], "d": ["b", "c", "c", None]})
_NULL_STR_MATCHED = {("a", "b"), ("b", "c")}


def _pairs_with_nulls_named(g):
    """``edge_pair_set`` with NULL spelled ``"NULL"`` -- NaN != NaN makes a raw set unusable."""
    df = to_pandas_any(g._edges)
    if df is None or len(df) == 0:
        return set()

    def _v(x):
        return "NULL" if pd.isna(x) else x

    return {(_v(a), _v(b)) for a, b in zip(df[g._source].tolist(), df[g._destination].tolist())}


def _ids_with_nulls_named(g):
    """``node_id_set`` with NULL spelled ``"NULL"`` -- NaN != NaN makes a raw set unusable."""
    df = to_pandas_any(g._nodes)
    if df is None or len(df) == 0:
        return set()
    return {"NULL" if pd.isna(x) else x for x in df[g._node].tolist()}


def _scalar(g, col):
    return to_pandas_any(g._nodes)[col].tolist()


# --- the rule on the traversal kernel ---------------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_a_null_edge_endpoint_matches_nothing_on_a_direct_hop(engine):
    """Hand-walked undirected walk from seed 0 with hops=4 over NULLEP. 0 --(0,1)--> 1
    --(1,2)--> 2, and there the walk stops: 2's remaining incident edges are (NULL,2) and
    (2,NULL), and a NULL endpoint is not an identity, so neither links. This is the cell the
    contract turns on -- an undirected walk reaches those edges through their NON-null side,
    so an implementation that only refuses to KEY on a NULL still traverses them."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).hop(
        nodes=_seed(engine, [0.0]), hops=4, direction="undirected", engine=engine)
    assert _pairs_with_nulls_named(out) == _NULL_ENDPOINT_MATCHED
    assert _ids_with_nulls_named(out) == _NULL_ENDPOINT_REACHED


#: (label, edges, seed, direction, matched) for the SYNTHESIZED-table sweep. This is the only
#: shape where the polars hop's own null drop is load-bearing (with a node table bound, the
#: closure gate's `is_in` already refuses a NULL endpoint), so it is swept, not sampled.
_SYNTH_NULL_ORACLE = [
    ("float_undirected", NULL_ENDPOINT_EDGES, 0.0, "undirected", _NULL_ENDPOINT_MATCHED),
    ("float_forward", NULL_ENDPOINT_EDGES, 0.0, "forward", {(0.0, 1.0), (1.0, 2.0)}),
    # reverse from 2 with a 4-hop window walks the whole chain back: (1,2) then (0,1),
    # while (NULL,2) is refused -- so the answer is the full matchable set, not one edge.
    ("float_reverse_from_2", NULL_ENDPOINT_EDGES, 2.0, "reverse", _NULL_ENDPOINT_MATCHED),
    ("float_forward_from_2", NULL_ENDPOINT_EDGES, 2.0, "forward", set()),
    ("str_undirected", NULL_STR_EDGES, "a", "undirected", _NULL_STR_MATCHED),
    ("str_reverse_from_c", NULL_STR_EDGES, "c", "reverse", _NULL_STR_MATCHED),
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("label,edges,seed,direction,want", _SYNTH_NULL_ORACLE)
def test_a_null_edge_endpoint_matches_nothing_on_the_synthesized_table(
    engine, label, edges, seed, direction, want
):
    """The rule does not come from the bound node table. With NO node table bound the id
    universe is synthesized FROM the endpoints and so holds a NULL -- and the answer is still
    the matchable edges only, because NULL is not an identity to synthesize. Hand-walked from
    the same edge lists as the bound cells: a forward walk from 2 finds only (2,NULL), a
    reverse walk into 2 finds (1,2) and the dropped (NULL,2)."""
    _require_engine(engine)
    out = _edges_only(engine, edges).hop(
        nodes=_seed(engine, [seed]), hops=4, direction=direction, engine=engine)
    assert _pairs_with_nulls_named(out) == want, label
    assert "NULL" not in _ids_with_nulls_named(out), label


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("direction,want", [
    # Seed 2 is the node BOTH null-endpoint edges touch, so each direction isolates one side.
    ("forward", set()),                  # 2's only out-edge is (2,NULL)
    ("reverse", {(1.0, 2.0)}),           # 2's in-edges are (1,2) and the dropped (NULL,2)
])
def test_each_null_endpoint_side_is_dropped_on_its_own(engine, direction, want):
    """SIDE SEPARATION. A source-only or destination-only implementation passes exactly one of
    these two rows: forward from 2 isolates the NULL DESTINATION, reverse from 2 the NULL
    SOURCE. (The undirected cells above cannot tell the two apart.)"""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).hop(
        nodes=_seed(engine, [2.0]), hops=1, direction=direction, engine=engine)
    assert _pairs_with_nulls_named(out) == want


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_a_null_seed_id_reaches_nothing(engine):
    """The seed side of the same rule: a caller may hand hop() a NULL seed id, and it resolves
    to no node, so the walk starts nowhere -- it does NOT pick up the NULL-id node row."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).hop(
        nodes=_seed(engine, [None]), hops=2, direction="undirected", engine=engine)
    assert _pairs_with_nulls_named(out) == set()
    assert _ids_with_nulls_named(out) == set()


# --- the same rule on the chain and Cypher surfaces -------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_the_chain_answers_the_same_null_endpoint_question_as_hop(engine):
    """ONE rule on every surface: an unconstrained undirected chain selects the same two edges
    the direct hop above returns. Before the contract, hop() answered three here and the chain
    answered two on all three engines."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).gfql(
        [n(), e_undirected(), n()], engine=engine)
    assert _pairs_with_nulls_named(out) == _NULL_ENDPOINT_MATCHED
    assert _ids_with_nulls_named(out) == _NULL_ENDPOINT_REACHED


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_naming_the_ops_does_not_change_the_null_endpoint_answer(engine):
    """Binding an ALIAS may not change a row count. Names disable the single-hop chain fast
    path, so the named form is served by the full BFS -- a different executor for an otherwise
    identical query. pandas and cuDF used to answer 2 unnamed and 3 named, because the
    undirected BFS reaches a null-endpoint edge through its non-null side."""
    _require_engine(engine)
    g = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES)
    unnamed = g.gfql([n(), e_undirected(), n()], engine=engine)
    named = g.gfql([n(name="a"), e_undirected(name="x"), n(name="b")], engine=engine)
    assert _pairs_with_nulls_named(named) == _pairs_with_nulls_named(unnamed)
    assert _pairs_with_nulls_named(named) == _NULL_ENDPOINT_MATCHED


#: (label, nodes-or-None, edges, matched). The SYNTHESIZED rows are the only ones where the
#: polars chain fast path's own null drop is load-bearing -- with a node table bound the
#: fast path's closure semi-join already refuses a NULL endpoint -- so both id dtypes and both
#: directions are swept there rather than sampled.
_CHAIN_NULL_ORACLE = [
    ("bound_float", NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES, _NULL_ENDPOINT_MATCHED),
    ("synth_float", None, NULL_ENDPOINT_EDGES, _NULL_ENDPOINT_MATCHED),
    ("bound_str", NULL_STR_NODES, NULL_STR_EDGES, _NULL_STR_MATCHED),
    ("synth_str", None, NULL_STR_EDGES, _NULL_STR_MATCHED),
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("edge_op", [e_forward, e_undirected])
@pytest.mark.parametrize("label,nodes,edges,want", _CHAIN_NULL_ORACLE)
def test_the_chain_gates_a_null_endpoint_bound_or_synthesized(
    engine, edge_op, label, nodes, edges, want
):
    """An unconstrained pattern selects every MATCHABLE edge -- the ones with no NULL endpoint
    -- and that does not depend on whether a node table is bound, on the id dtype, or on the
    edge's direction. The output node frame never carries the NULL either."""
    _require_engine(engine)
    g = (_edges_only(engine, edges) if nodes is None else _bind(engine, nodes, edges))
    out = g.gfql([n(), edge_op(), n()], engine=engine)
    assert _pairs_with_nulls_named(out) == want, label
    assert "NULL" not in _ids_with_nulls_named(out), label


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("query,want", [
    # Two matchable edges: undirected matches each in both orientations, forward in one.
    ("MATCH (a)-[x]-(b) RETURN count(*) AS c", 4),
    ("MATCH (a)-[x]->(b) RETURN count(*) AS c", 2),
])
def test_cypher_count_counts_only_matchable_edges(engine, query, want):
    """The aggregate surface has to agree with the row surface. The undirected row was the
    headline divergence: polars answered 4, pandas and cuDF 6, over the same graph."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).gfql(query, engine=engine)
    assert _scalar(out, "c") == [want]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_a_null_id_node_row_is_still_a_row(engine):
    """THE OTHER HALF OF THE CONTRACT, and the cell that separates it from 'drop NULL nodes':
    a node-only pattern counts all FOUR node rows including the NULL-id one. The rule governs
    endpoint RESOLUTION, not row existence."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).gfql(
        "MATCH (a) RETURN count(*) AS c", engine=engine)
    assert _scalar(out, "c") == [4]


# --- endpoint closure of the OUTPUT frame, under the contract ---------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("surface", ["hop", "chain"])
def test_no_output_frame_references_a_node_it_does_not_carry(engine, surface):
    """OUTPUT-SIDE closure, stated without naming NULL. The polars hop used to keep the
    (NULL,2) edge while its node-output semi-join (``all_nodes.join(needed, how='semi')``)
    never matched NULL to NULL, so the frame carried an edge whose endpoint had no node row --
    wrong under EITHER contract. Asserted on the null-aware id spelling, because a raw
    ``set(...) <= set(...)`` is vacuously true for NaN endpoints (NaN != NaN)."""
    _require_engine(engine)
    g = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES)
    out = (g.hop(nodes=_seed(engine, [0.0]), hops=4, direction="undirected", engine=engine)
           if surface == "hop" else g.gfql([n(), e_undirected(), n()], engine=engine))
    ids = _ids_with_nulls_named(out)
    endpoints = {i for pair in _pairs_with_nulls_named(out) for i in pair}
    assert endpoints <= ids, f"edge endpoints with no node row: {endpoints - ids}"
    assert "NULL" not in endpoints


# --- ANTI-VACUITY control + a second id dtype -------------------------------------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_the_null_free_twin_matches_every_edge(engine):
    """CONTROL. NULLFREE is NULLEP with the id 3 substituted for every NULL, so it has the same
    shape, the same seeds and the same walk -- and every edge matches, the undirected count is
    4*2=8 and the forward count 4. Without this row the cells above would pass just as well
    against an implementation that deleted edges for the wrong reason."""
    _require_engine(engine)
    g = _bind(engine, NULL_FREE_TWIN_NODES, NULL_FREE_TWIN_EDGES)
    out = g.hop(nodes=_seed(engine, [0.0]), hops=4, direction="undirected", engine=engine)
    assert _pairs_with_nulls_named(out) == _NULL_FREE_TWIN_MATCHED
    assert _ids_with_nulls_named(out) == {0.0, 1.0, 2.0, 3.0}
    assert _pairs_with_nulls_named(
        g.gfql([n(), e_undirected(), n()], engine=engine)) == _NULL_FREE_TWIN_MATCHED
    assert _scalar(g.gfql("MATCH (a)-[x]-(b) RETURN count(*) AS c", engine=engine), "c") == [8]
    assert _scalar(g.gfql("MATCH (a)-[x]->(b) RETURN count(*) AS c", engine=engine), "c") == [4]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_the_null_endpoint_contract_holds_on_string_ids(engine):
    """Same rule on an OBJECT id column, which reaches a different null representation on each
    engine (pandas ``None`` in object, polars Utf8 null, cuDF string null mask) than the float
    NaN the fixtures above use."""
    _require_engine(engine)
    g = _bind(engine, NULL_STR_NODES, NULL_STR_EDGES)
    out = g.hop(nodes=_seed(engine, ["a"]), hops=4, direction="undirected", engine=engine)
    assert _pairs_with_nulls_named(out) == _NULL_STR_MATCHED
    assert _ids_with_nulls_named(out) == {"a", "b", "c"}
    assert _pairs_with_nulls_named(
        g.gfql([n(), e_forward(), n()], engine=engine)) == _NULL_STR_MATCHED
    assert _scalar(g.gfql("MATCH (a)-[x]-(b) RETURN count(*) AS c", engine=engine), "c") == [4]


# --- DOCUMENTED BOUNDARY: what the contract deliberately does NOT reach --------------------

@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_get_degrees_counts_raw_edge_rows_not_matchable_edges(engine):
    """DEFERRED, pinned rather than left unwritten. ``get_degrees`` -- and the GFQL CALL that
    exposes it -- is a raw edge-row tally, not a pattern match, so it counts a null-endpoint
    edge on whichever endpoint IS an identity: over NULLEP node 2 has degree_in 2 and
    degree_out 1, while exactly one MATCHABLE edge reaches it in each direction. All three
    engines already agree here, so this is a semantic question about what ``get_degrees``
    means rather than a cross-engine divergence, and moving it would change the value of an
    existing user-facing column. Pinned so the boundary cannot drift either way unnoticed."""
    _require_engine(engine)
    out = _bind(engine, NULL_ENDPOINT_NODES, NULL_ENDPOINT_EDGES).get_degrees()
    rows = to_pandas_any(out._nodes)
    got = {("NULL" if pd.isna(r["id"]) else r["id"]):
           (int(r["degree_in"]), int(r["degree_out"])) for _, r in rows.iterrows()}
    assert got == {0.0: (0, 1), 1.0: (1, 1), 2.0: (2, 1), "NULL": (0, 0)}
