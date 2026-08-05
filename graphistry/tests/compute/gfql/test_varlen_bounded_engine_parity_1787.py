"""Bounded variable-length segments: ONE contract, four engines (#1787).

The contract is **parity with the pandas oracle, or ``NotImplementedError``** — never a
different answer with no error. Three bounded shapes broke it on the native polars row
pipeline, so they are now declined, mirroring what #1781 did for the unbounded case:

  1. directed ``-[*k..m]->`` with ``min_hops >= 3`` (and ``min_hops >= 2`` off a filtered seed);
  2. undirected ``-[*1..1]-`` / ``-[*1]-``, the degenerate window, which HALVED the count;
  3. undirected ``-[*1..k]-`` that does not start from the full node set (filtered seed, or a
     non-first segment), which OVER-counted.

WHY THE DIVERGENCE FROM MASTER IS DELIBERATE, and the one thing no test below can state: on
master these shapes were *served*, so declining them is a visible behaviour change that turns
a silent wrong answer into a loud error. That is the intended direction — pandas is the
oracle, and an engine that cannot reproduce the oracle's edge multiplicity must say so. The
underlying limitation is a reconstruction gap, not a policy: pandas' ``step_pairs`` come from
the variable-length ``edge_op.execute`` hop, whose hop-window pruning (and, when seeded, its
per-seed BFS) changes edge multiplicity in a way a rebuild from the raw matching edge table
does not reproduce. When it becomes reconstructible, the gate should shrink and the
``*_declines_*`` tests below should be rewritten to parity tests.

WHY THIS FILE IS ENGINE-PARAMETRIZED RATHER THAN polars-ONLY: "which shapes are answerable,
and with what value" is engine-agnostic semantics. It is parametrized over the four engines,
and it encodes the INTENDED PER-ENGINE behaviour rather than assuming identity — the polars
engines must DECLINE exactly where the pandas-API engines must ANSWER. Asserting agreement
everywhere would be false; asserting only polars would miss that cuDF is the other half of
the same contract. It already paid for itself: the cuDF divergence pinned at the bottom of
this file was found by adding the cuDF parameter.

COVERAGE BOUNDARY, stated rather than hidden behind a skip: no CI lane runs cuDF or
polars-gpu (``ci-gpu.yml`` is hard-disabled and does not install the ``polars`` extra), so on
CI those two parameters report SKIPPED — visibly, via a runtime probe, never as a silent pass.
They are exercised out of band on the dgx GPU box against
``graphistry/test-rapids-official:26.02-gfql-polars`` (``docker run --gpus all`` — omitting
``--gpus all`` FABRICATES failures rather than skipping). Treat a green CI run as evidence for
pandas + polars only.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, NamedTuple, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable

# Two families, not four independent engines. cuDF runs the pandas-API traversal, so it must
# ANSWER; polars-gpu is the GPU collect target of the SAME lazy polars engine, so it must
# decline wherever polars declines. Kept as fixed tuples rather than an importability probe:
# an engine that cannot run here must show up as a SKIPPED parameter, not vanish from the
# report as though it had never been in scope.
PANDAS_API_ENGINES: Tuple[str, ...] = ("pandas", "cudf")
POLARS_API_ENGINES: Tuple[str, ...] = ("polars", "polars-gpu")
ALL_ENGINES: Tuple[str, ...] = PANDAS_API_ENGINES + POLARS_API_ENGINES


class Shape(NamedTuple):
    """One (graph, query) case plus the pandas-oracle answer, pinned as a literal.

    The oracle count is pinned rather than recomputed so a regression that makes every engine
    equally wrong still fails — cross-engine agreement alone would pass it.
    """

    id: str
    nodes: pd.DataFrame
    edges: pd.DataFrame
    query: str
    oracle: int


# --- fixtures (from the #1787 differential fuzz) -------------------------------------------

BOUNDED_NODES = pd.DataFrame({"id": list(range(7))})
BOUNDED_EDGES = pd.DataFrame(
    [(4, 6), (0, 4), (1, 2), (4, 6), (5, 6), (4, 5), (4, 6)], columns=["s", "d"]
)
UNDIR_NODES = pd.DataFrame({"id": list(range(5))})
UNDIR_EDGES = pd.DataFrame(
    [(1, 2), (1, 2), (0, 2), (1, 0), (4, 2), (3, 2), (3, 4), (0, 3), (1, 2)], columns=["s", "d"]
)
SEEDED_NODES = pd.DataFrame({"id": list(range(7)), "kind": ["a", "b"] * 3 + ["a"]})
SEEDED_EDGES = pd.DataFrame(
    [(0, 3), (6, 3), (4, 2), (1, 3), (6, 3), (3, 1), (3, 5), (4, 1), (3, 3), (0, 3)],
    columns=["s", "d"],
)


def _bounded(id_: str, query: str, oracle: int) -> Shape:
    return Shape(id_, BOUNDED_NODES, BOUNDED_EDGES, query, oracle)


def _undir(id_: str, query: str, oracle: int) -> Shape:
    return Shape(id_, UNDIR_NODES, UNDIR_EDGES, query, oracle)


def _seeded(id_: str, query: str, oracle: int) -> Shape:
    return Shape(id_, SEEDED_NODES, SEEDED_EDGES, query, oracle)


# --- the contract, as data -----------------------------------------------------------------

#: Shapes the polars engines must DECLINE and the pandas-API engines must ANSWER correctly.
DECLINED_BY_POLARS: List[Shape] = [
    _bounded("dir-min3-exact", "MATCH (a)-[*3..3]->(b) RETURN count(*) AS c", 0),
    _bounded("dir-min3-window", "MATCH (a)-[*3..4]->(b) RETURN count(*) AS c", 0),
    _seeded("dir-min2-seeded", "MATCH (a {kind:'a'})-[*2..3]->(b) RETURN count(*) AS c", 32),
    _undir("undir-degenerate-window", "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c", 36),
    _undir("undir-degenerate-exact", "MATCH (a)-[*1]-(b) RETURN count(*) AS c", 36),
    _seeded("undir-seeded", "MATCH (a {kind:'a'})-[*1..2]-(b) RETURN count(*) AS c", 38),
    _seeded("undir-non-first-segment", "MATCH (a)-[]->(b)-[*1..2]-(c) RETURN count(*) AS c", 316),
]

#: Neighbours of every declined shape, on BOTH sides of each gate boundary. These are what
#: prove the gate is a scalpel: a lazily over-broad gate that declined "anything variable
#: length" would fail every one of them.
SERVED_EVERYWHERE: List[Shape] = [
    _bounded("dir-min1", "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c", 12),
    _bounded("dir-min2-unseeded", "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c", 6),
    _undir("undir-plain-edge", "MATCH (a)-[]-(b) RETURN count(*) AS c", 18),
    _undir("undir-min1-max2", "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c", 212),
    _undir("undir-min1-max3", "MATCH (a)-[*1..3]-(b) RETURN count(*) AS c", 948),
    _undir("dir-degenerate-window", "MATCH (a)-[*1..1]->(b) RETURN count(*) AS c", 9),
    _seeded("dir-seeded-min1", "MATCH (a {kind:'a'})-[*1..2]->(b) RETURN count(*) AS c", 19),
    _seeded("dir-non-first-segment", "MATCH (a)-[]->(b)-[*2..3]->(c) RETURN count(*) AS c", 80),
]


# --- engine plumbing -----------------------------------------------------------------------

_PROBE_NODES = pd.DataFrame({"id": [0, 1, 2]})
_PROBE_EDGES = pd.DataFrame([(0, 1), (1, 2)], columns=["s", "d"])


def _graph(engine: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> Plottable:
    """Build the same logical graph in the engine's native frame type."""
    if engine in POLARS_API_ENGINES:
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(nodes), "id").edges(pl.from_pandas(edges), "s", "d")
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return graphistry.nodes(cudf.from_pandas(nodes), "id").edges(
            cudf.from_pandas(edges), "s", "d"
        )
    return graphistry.nodes(nodes, "id").edges(edges, "s", "d")


@lru_cache(maxsize=None)
def _engine_runnable(engine: str) -> bool:
    """Probe by RUNNING the smallest version of what these tests do.

    Cheaper probes do not discriminate on a box with cudf/cudf_polars importable but no
    working CUDA runtime — frame construction and simple ops all succeed there and the suite
    then dies inside the first real kernel. So the probe is an actual traversal.
    """
    try:
        g = _graph(engine, _PROBE_NODES, _PROBE_EDGES)
        g.gfql("MATCH (a)-[]->(b) RETURN count(*) AS c", engine=engine)
        return True
    except Exception:  # noqa: BLE001 — any failure here means "cannot run", never "test fails"
        return False


def _require(engine: str) -> None:
    if not _engine_runnable(engine):
        pytest.skip(
            f"engine {engine!r} is not runnable in this environment — NOT evidence that it "
            "passes; see the COVERAGE BOUNDARY note in this module's docstring"
        )


def _count(g: Plottable, query: str, engine: str) -> int:
    """Scalar ``count(*)`` out of any engine's result frame.

    Dispatched on the engine rather than on ``hasattr(col, "to_pandas")``: polars Series
    ALSO expose ``to_pandas()``, but it needs pyarrow, which the polars test lane does not
    install — so an attribute probe would silently route polars down a path that can fail
    on a dependency this file has no reason to require.
    """
    col = g.gfql(query, engine=engine)._nodes["c"]
    return int(col.to_pandas().iloc[0]) if engine == "cudf" else int(col.to_list()[0])


def _answer(engine: str, shape: Shape) -> int:
    return _count(_graph(engine, shape.nodes, shape.edges), shape.query, engine)


_DECLINED_IDS = [s.id for s in DECLINED_BY_POLARS]
_SERVED_IDS = [s.id for s in SERVED_EVERYWHERE]


# --- the contract, as tests ----------------------------------------------------------------


@pytest.mark.parametrize("engine", POLARS_API_ENGINES)
@pytest.mark.parametrize("shape", DECLINED_BY_POLARS, ids=_DECLINED_IDS)
def test_unreconstructible_shape_declines_on_the_polars_engines(shape: Shape, engine: str) -> None:
    """A shape whose multiplicity is not reconstructible must RAISE, never answer differently."""
    _require(engine)
    with pytest.raises(NotImplementedError):
        _answer(engine, shape)


@pytest.mark.parametrize("engine", PANDAS_API_ENGINES)
@pytest.mark.parametrize("shape", DECLINED_BY_POLARS, ids=_DECLINED_IDS)
def test_unreconstructible_shape_is_still_answered_on_the_pandas_api_engines(
    shape: Shape, engine: str
) -> None:
    """Declining is an ENGINE limit, not a query rejection: pandas and cuDF still answer."""
    _require(engine)
    assert _answer(engine, shape) == shape.oracle


@pytest.mark.parametrize("engine", ALL_ENGINES)
@pytest.mark.parametrize("shape", SERVED_EVERYWHERE, ids=_SERVED_IDS)
def test_neighbouring_shape_is_answered_identically_on_every_engine(
    shape: Shape, engine: str
) -> None:
    """The gate is a scalpel: every neighbour of a declined shape still runs, and agrees."""
    _require(engine)
    assert _answer(engine, shape) == shape.oracle


# --- the boundary claims the gate is built on ----------------------------------------------
# Each of these was a prose comment in row_pipeline.py. A comment asserting behaviour is
# unverified and rots; these do not.


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_degenerate_window_is_not_the_same_query_as_a_plain_edge(engine: str) -> None:
    """WHY the gate keys on an EXPLICIT window instead of ``sem.is_multihop``.

    ``-[*1..1]-`` resolves to min == max == 1 and is therefore NOT multihop, yet pandas still
    routes it through the variable-length hop and gets a different answer from ``-[]-`` on the
    very same graph (36 vs 18 — the undirected doubling). Gating on ``is_multihop`` would have
    let the degenerate window straight through, which is exactly why it went unnoticed.
    """
    _require(engine)
    plain = _count(_graph(engine, UNDIR_NODES, UNDIR_EDGES),
                   "MATCH (a)-[]-(b) RETURN count(*) AS c", engine)
    assert plain == 18, "the plain undirected edge is served by every engine, unchanged"
    if engine in POLARS_API_ENGINES:
        with pytest.raises(NotImplementedError):
            _count(_graph(engine, UNDIR_NODES, UNDIR_EDGES),
                   "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c", engine)
    else:
        degenerate = _count(_graph(engine, UNDIR_NODES, UNDIR_EDGES),
                            "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c", engine)
        assert degenerate == 36 != plain


@pytest.mark.parametrize("engine", PANDAS_API_ENGINES)
def test_directed_min_hops_3_collapses_to_empty_on_the_oracle(engine: str) -> None:
    """The observable consequence of ``max_reached_hop`` being a BFS eccentricity.

    ``compute/hop.py`` prunes against a dedup-by-node eccentricity, not a longest-walk length,
    so on this graph the oracle returns NOTHING at ``min_hops >= 3`` while still returning rows
    at ``min_hops == 2``. A raw-edge rebuild expands a different edge multiset and cannot land
    on empty, which is why ``min_hops >= 3`` is unreconstructible rather than merely narrower.
    """
    _require(engine)
    g = _graph(engine, BOUNDED_NODES, BOUNDED_EDGES)
    assert _count(g, "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c", engine) == 6
    assert _count(g, "MATCH (a)-[*3..3]->(b) RETURN count(*) AS c", engine) == 0
    assert _count(g, "MATCH (a)-[*3..4]->(b) RETURN count(*) AS c", engine) == 0


@pytest.mark.parametrize("engine", POLARS_API_ENGINES)
def test_directed_min_hops_2_declines_only_when_the_segment_is_seeded(engine: str) -> None:
    """``min_hops == 2`` is fuzz-clean off the FULL node set and only diverges under a seed.

    The seeded hop runs a per-seed BFS, which is what changes the multiplicity. Pinning both
    sides keeps the gate from being widened to all of ``min_hops >= 2`` on a hunch — the
    unseeded form is the graph-bench ``-[*1..k]->`` shape and must stay native.
    """
    _require(engine)
    g = _graph(engine, SEEDED_NODES, SEEDED_EDGES)
    assert _count(g, "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c", engine) == 50
    with pytest.raises(NotImplementedError):
        _count(g, "MATCH (a {kind:'a'})-[*2..3]->(b) RETURN count(*) AS c", engine)


@pytest.mark.parametrize("engine", POLARS_API_ENGINES)
def test_undirected_seed_declines_while_its_directed_twin_is_served(engine: str) -> None:
    """The over-count is specific to the undirected DOUBLING, not to seeding.

    Same seed, same window, same graph: the directed spelling agrees with pandas and stays
    native, so the gate is correctly restricted to ``direction == "undirected"``.
    """
    _require(engine)
    g = _graph(engine, SEEDED_NODES, SEEDED_EDGES)
    assert _count(g, "MATCH (a {kind:'a'})-[*1..2]->(b) RETURN count(*) AS c", engine) == 19
    with pytest.raises(NotImplementedError):
        _count(g, "MATCH (a {kind:'a'})-[*1..2]-(b) RETURN count(*) AS c", engine)


@pytest.mark.parametrize("engine", POLARS_API_ENGINES)
def test_undirected_non_first_segment_declines_while_its_directed_twin_is_served(
    engine: str,
) -> None:
    """A non-first segment starts from less than the full node set, exactly like a seed."""
    _require(engine)
    g = _graph(engine, SEEDED_NODES, SEEDED_EDGES)
    assert _count(g, "MATCH (a)-[]->(b)-[*1..2]->(c) RETURN count(*) AS c", engine) == 50
    with pytest.raises(NotImplementedError):
        _count(g, "MATCH (a)-[]->(b)-[*1..2]-(c) RETURN count(*) AS c", engine)


@pytest.mark.parametrize("engine", POLARS_API_ENGINES)
def test_degenerate_window_decline_is_undirected_only(engine: str) -> None:
    """``-[*1..1]->`` (DIRECTED) has no doubling to get wrong and must stay native."""
    _require(engine)
    g = _graph(engine, UNDIR_NODES, UNDIR_EDGES)
    assert _count(g, "MATCH (a)-[*1..1]->(b) RETURN count(*) AS c", engine) == 9


# --- the fuzz that found all three ----------------------------------------------------------

FUZZ_SHAPES: Tuple[str, ...] = (
    "MATCH (a)-[*1..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*2..2]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*2..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*3..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..1]-(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..2]-(b) RETURN count(*) AS c",
    "MATCH (a)-[*1..3]-(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*1..2]-(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*1..2]->(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'})-[*2..3]->(b) RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[*1..2]-(c) RETURN count(*) AS c",
    "MATCH (a)-[]->(b)-[*2..3]->(c) RETURN count(*) AS c",
)


def _random_graph(seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    import random

    rnd = random.Random(seed)
    n_nodes = rnd.randint(3, 7)
    nodes = pd.DataFrame({
        "id": list(range(n_nodes)),
        "kind": [rnd.choice("ab") for _ in range(n_nodes)],
    })
    edges = pd.DataFrame(
        [(rnd.randrange(n_nodes), rnd.randrange(n_nodes)) for _ in range(rnd.randint(3, 10))],
        columns=["s", "d"],
    )
    return nodes, edges


@pytest.mark.parametrize("engine", [e for e in ALL_ENGINES if e != "pandas"])
def test_bounded_varlen_differential_fuzz_against_pandas_oracle(engine: str) -> None:
    """Random small graphs (cyclic, parallel edges, self-loops): match pandas, or raise.

    Bounded windows only — undirected UNBOUNDED shapes through the pandas oracle can blow the
    box up (75 GiB RSS observed) and are already declined elsewhere. Seeded, so a failure is
    reproducible. The shape list spans both sides of every gate boundary, so a gate that
    declines too much shows up as the served-count assertion at the end going to zero.
    """
    _require(engine)
    served = 0
    for seed in range(25):
        nodes, edges = _random_graph(1787 + seed)
        g_pd = _graph("pandas", nodes, edges)
        g_engine = _graph(engine, nodes, edges)
        for query in FUZZ_SHAPES:
            expected = _count(g_pd, query, "pandas")
            try:
                got = _count(g_engine, query, engine)
            except NotImplementedError:
                continue  # an honest decline is allowed; a different answer is not
            served += 1
            assert got == expected, (
                f"{engine} diverged from the pandas oracle on {query!r}: "
                f"{got} != {expected}\nnodes={nodes.to_dict('list')}"
                f"\nedges={edges.values.tolist()}"
            )
    assert served > 100, f"{engine} declines too much to be a useful parity check ({served})"


# --- a divergence this cross-engine parametrization FOUND ------------------------------------

# Minimal repro reduced from the fuzz: `{kind:'a'}` selects nodes 1 and 2.
CUDF_DIVERGENCE_NODES = pd.DataFrame({"id": [0, 1, 2, 3], "kind": ["b", "a", "a", "b"]})
CUDF_DIVERGENCE_EDGES = pd.DataFrame(
    [(2, 2), (0, 3), (2, 2), (3, 1), (0, 3), (2, 2), (1, 1)], columns=["s", "d"]
)


# The cuDF parameter carries the xfail at COLLECTION time (a marker added inside the test body
# is applied too late to be reliable), and STRICT so the fix cannot land unnoticed.
_DEGENERATE_SEED_ENGINES = [
    pytest.param(
        engine,
        marks=pytest.mark.xfail(
            strict=True,
            reason="#1798: cuDF answers 1 where the pandas oracle answers 9",
        ),
    )
    if engine == "cudf"
    else engine
    for engine in ALL_ENGINES
]


@pytest.mark.parametrize("engine", _DEGENERATE_SEED_ENGINES)
def test_seeded_undirected_degenerate_window_agrees_with_the_oracle(engine: str) -> None:
    """``(a {..})-[*1..1]-(b)``: a SEPARATE cuDF defect, in the pandas-API family (#1798).

    Not the #1787 gap and not fixed by this PR — the polars engines decline this shape (it is
    the ``undir-degenerate-window`` case above with a seed), so the polars fix cannot mask it.
    cuDF answers 1 where pandas answers 9, silently: 23 of 40 random graphs diverge on this
    shape alone, while the UNSEEDED ``-[*1..1]-``, the seeded ``-[]-``, the seeded
    ``-[*1..2]-`` and the directed ``-[*1..1]->`` are all clean (0/40 each).

    Recorded as a STRICT xfail rather than left out: when #1798 lands, this test fails loudly
    and must be un-xfailed, whereas a comment or a skip would rot silently.
    """
    _require(engine)
    g = _graph(engine, CUDF_DIVERGENCE_NODES, CUDF_DIVERGENCE_EDGES)
    query = "MATCH (a {kind:'a'})-[*1..1]-(b) RETURN count(*) AS c"
    if engine in POLARS_API_ENGINES:
        with pytest.raises(NotImplementedError):
            _count(g, query, engine)
        return
    assert _count(g, query, engine) == 9
