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
