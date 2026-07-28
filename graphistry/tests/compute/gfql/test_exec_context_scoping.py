"""Ownership and scoping of the row execution context — why ``clear`` NULLs (#1793 review).

THE QUESTION THIS FILE DECIDES. ``clear_row_exec_context`` sets ``_gfql_start_nodes`` and
``_gfql_rows_base_graph`` to ``None`` on the way out. ``attach_row_exec_context`` documents the
opposite asymmetry on the way in — a ``None`` argument means "keep whatever ``g`` already
carries". So an OUTER scope's value can reach an inner execution, and the inner execution then
drops it. Should ``clear`` instead RESTORE the value the graph carried on entry?

**No. NULL is correct, and restore would reintroduce the very bug #1793 fixed.** Three
independent reasons, each pinned by a test below rather than argued:

1. ``clear`` IS PURE. It returns ``g.bind()``; it never writes through to the object it was
   handed. So an outer scope that set the field still has it afterwards — there is no caller
   state to save and nothing to restore FOR. This is what makes it a different bug class from
   #1786, which was an in-place ``dispatch_graph._gfql_start_nodes = ...`` on the caller's own
   object. ``test_clear_never_touches_the_object_it_was_handed``.

2. THE ONLY CHANNEL RESTORE WOULD CHANGE IS THE RETURN VALUE — and putting the seed back on the
   result is exactly the second half of #1786 ("the RESULT of a WITH query carries the seed, and
   a follow-up query on that result — a different graph entirely — is answered against it").
   Measured, not asserted: hand-restoring the seed onto a result changes the answer of the next
   query run on it. ``test_a_restored_seed_would_poison_a_follow_up_query_on_the_result``.

3. NO EXECUTION FRAME EVER INHERITS A CONTEXT IT DID NOT SET. The inherit branch is reachable
   from the internal API (tests 1 and 2 use it) but no production path takes it: instrumenting
   ``attach_row_exec_context`` over the whole ``tests/compute`` tree recorded 3907 attaches and
   ZERO where the entering graph already carried a value the frame did not itself set. (53 DID
   enter on a graph that already carried a seed -- nested boundary frames -- but every one was
   handed the IDENTICAL ``start_nodes`` parameter, so the frame still owns the value it sets.)
   The cross-segment WITH seed is threaded as the explicit ``start_nodes`` PARAMETER
   (``chain_impl(..., start_nodes=)``; ``_compiled_query_reentry_state`` -> ``start_nodes=``),
   never through the graph field, so the field's lifetime is exactly ONE boundary-call run.
   ``test_no_execution_frame_inherits_a_context_it_did_not_set`` re-runs that measurement as an
   assertion over a corpus, so a future path that starts relying on inheritance fails here and
   the design question is reopened deliberately instead of silently.

COVERAGE BOUNDARY, stated rather than hidden behind a skip: the context is engine-independent
plumbing, but the shapes below are exercised on all four engines because the two attach/clear
pairs are duplicated across the generic chain and the native polars chain. No CI lane runs cuDF
or polars-gpu (``ci-gpu.yml`` is hard-disabled and does not install the ``polars`` extra), so
those two parameters report SKIPPED on CI — visibly, via a runtime probe, never as a silent
pass. They are exercised out of band on the dgx GPU box against
``graphistry/test-rapids-official:26.02-gfql-polars`` (``docker run --gpus all`` — omitting
``--gpus all`` FABRICATES failures rather than skipping).
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, List, Tuple

import pandas as pd
import pytest

import graphistry
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, e_forward, n, rows, serialize_binding_ops
from graphistry.compute.gfql.exec_context import (
    attach_row_exec_context,
    clear_row_exec_context,
)

# Fixed list, not a probe-built one: an engine that cannot run here must show up as a SKIPPED
# parameter, not vanish from the report as though it had never been in scope (the
# ``available_nonpandas_engines()`` trap).
PANDAS_API_ENGINES: Tuple[str, ...] = ("pandas", "cudf")
POLARS_API_ENGINES: Tuple[str, ...] = ("polars", "polars-gpu")
ALL_ENGINES: Tuple[str, ...] = PANDAS_API_ENGINES + POLARS_API_ENGINES

NODES = pd.DataFrame({"id": list(range(7)), "kind": ["a", "b"] * 3 + ["a"]})
EDGES = pd.DataFrame(
    [(0, 3), (1, 2), (4, 6), (3, 5), (5, 6), (0, 6), (3, 4)], columns=["s", "d"]
)

#: A named middle + bare ``rows()``: the boundary-call shape whose bindings builder READS
#: ``_gfql_start_nodes``. Both chain surfaces (generic and native polars) route it.
BOUNDARY_OPS: List[ASTObject] = [n(name="a"), e_forward(name="r"), n(name="b"), rows()]

#: Pinned literals, so a regression that makes every engine equally wrong still fails.
UNSEEDED_ROWS = 7
SEEDED_ROWS = 2  # only node 0's outgoing edges


def _frame(engine: str, df: pd.DataFrame) -> Any:
    if engine in POLARS_API_ENGINES:
        pl = pytest.importorskip("polars")
        return pl.from_pandas(df)
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(df)
    return df


def _graph(engine: str) -> Plottable:
    return graphistry.nodes(_frame(engine, NODES), "id").edges(
        _frame(engine, EDGES), "s", "d"
    )


def _seed(engine: str) -> Any:
    """The outer scope's re-entry seed: node 0 only."""
    return _frame(engine, pd.DataFrame({"id": [0]}))


@lru_cache(maxsize=None)
def _engine_runnable(engine: str) -> bool:
    """Probe by RUNNING the smallest version of what these tests do.

    Cheaper probes do not discriminate on a box with cudf/cudf_polars importable but no working
    CUDA runtime — frame construction and simple ops all succeed there and the suite then dies
    inside the first real kernel. So the probe is an actual boundary-call run.
    """
    try:
        _graph(engine).gfql(list(BOUNDARY_OPS), engine=engine)
        return True
    except Exception:  # noqa: BLE001 — any failure means "cannot run", never "test fails"
        return False


def _require(engine: str) -> None:
    if not _engine_runnable(engine):
        pytest.skip(
            f"engine {engine!r} is not runnable in this environment — NOT evidence that it "
            "passes; see the COVERAGE BOUNDARY note in this module's docstring"
        )


def _rows(g: Plottable, engine: str, ops: List[ASTObject]) -> int:
    out = g.gfql(list(ops), engine=engine)._nodes
    assert out is not None
    return len(out)


# --- 1. purity: there is nothing to restore FOR ---------------------------------------------


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_clear_never_touches_the_object_it_was_handed(engine: str) -> None:
    """``clear`` returns a COPY. The outer scope keeps its own context regardless.

    This is the whole reason save/restore is unnecessary: the motive for restoring is to
    protect a caller whose object was mutated, and no object is ever mutated here. #1786 WAS
    that mutation (``dispatch_graph._gfql_start_nodes = ...`` on the caller's graph); the
    attach/clear pair is its replacement, not another instance of it.
    """
    _require(engine)
    outer = attach_row_exec_context(_graph(engine), start_nodes=_seed(engine))
    seed_before = outer._gfql_start_nodes

    cleared = clear_row_exec_context(outer)

    assert cleared is not outer
    assert cleared._gfql_start_nodes is None
    assert outer._gfql_start_nodes is seed_before, \
        f"[{engine}] clear_row_exec_context wrote through to the graph it was handed"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_a_full_execution_leaves_the_outer_scope_context_intact(engine: str) -> None:
    """End to end, not just the helper: running a query does not strip the caller's context."""
    _require(engine)
    outer = attach_row_exec_context(_graph(engine), start_nodes=_seed(engine))
    seed_before = outer._gfql_start_nodes

    outer.gfql(list(BOUNDARY_OPS), engine=engine)

    assert outer._gfql_start_nodes is seed_before, \
        f"[{engine}] the execution consumed the caller's context instead of copying it"


# --- 2. inherit on the way IN, drop on the way OUT ------------------------------------------


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_an_outer_context_IS_inherited_on_the_way_in(engine: str) -> None:
    """``attach``'s "None means keep what ``g`` carries" is live and changes the ANSWER.

    Without this the two tests that follow would be vacuous — dropping a value on exit proves
    nothing if the value never reached the execution in the first place.
    """
    _require(engine)
    plain = _rows(_graph(engine), engine, BOUNDARY_OPS)
    seeded = _rows(
        attach_row_exec_context(_graph(engine), start_nodes=_seed(engine)),
        engine,
        BOUNDARY_OPS,
    )
    assert plain == UNSEEDED_ROWS, f"[{engine}] unseeded bindings changed: {plain}"
    assert seeded == SEEDED_ROWS, f"[{engine}] the outer seed did not reach the builder: {seeded}"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_an_outer_context_is_DROPPED_on_the_way_out(engine: str) -> None:
    """THE DECIDER. Under save/restore the result would carry the outer seed; it must not.

    Distinguishing, not merely descriptive: restore returns ``_seed(engine)`` here and NULL
    returns ``None``, so exactly one of the two implementations passes.
    """
    _require(engine)
    outer = attach_row_exec_context(_graph(engine), start_nodes=_seed(engine))

    result = outer.gfql(list(BOUNDARY_OPS), engine=engine)

    assert result._gfql_start_nodes is None, \
        f"[{engine}] the outer scope's seed rode out on the result (#1786, one hop removed)"
    assert result._gfql_rows_base_graph is None, \
        f"[{engine}] the outer scope's base graph rode out on the result"


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_dropping_the_context_does_not_change_the_answer(engine: str) -> None:
    """Do not "fix" the leak by breaking the seed: the seeded answer itself must survive."""
    _require(engine)
    outer = attach_row_exec_context(_graph(engine), start_nodes=_seed(engine))
    assert _rows(outer, engine, BOUNDARY_OPS) == SEEDED_ROWS
    assert SEEDED_ROWS != UNSEEDED_ROWS  # else the test above proves nothing


# --- 3. restore is not merely unnecessary, it is harmful ------------------------------------


def test_a_restored_seed_would_poison_a_follow_up_query_on_the_result() -> None:
    """The harm, measured: a seed left on the result changes the NEXT query's answer.

    ``restored`` is what a save/restore ``clear`` would hand back — the result graph with the
    outer scope's seed put back on it. The follow-up ``rows(binding_ops=...)`` reads
    ``_gfql_start_nodes`` off whatever graph it is run on, so the two disagree. Nothing about
    the follow-up query mentions the previous one; that is the #1786 defect.

    pandas-only ON PURPOSE, and not a coverage gap: the native polars row op DECLINES
    ``rows()`` over a row-table graph (honest ``NotImplementedError``), so the follow-up shape
    this test needs does not exist there. The engine-parametrized half of the contract is the
    four tests above; this one pins WHY those assert ``None``.
    """
    g = graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")
    seed = pd.DataFrame({"id": [0]})
    result = attach_row_exec_context(g, start_nodes=seed).gfql(list(BOUNDARY_OPS), engine="pandas")
    restored = attach_row_exec_context(result, start_nodes=seed)

    follow_up: List[ASTObject] = [rows(binding_ops=serialize_binding_ops([n(name="x")]))]
    cleared_answer = _rows(result, "pandas", follow_up)
    restored_answer = _rows(restored, "pandas", follow_up)

    assert cleared_answer == 2
    assert restored_answer != cleared_answer, (
        "a seed restored onto the result did NOT change the follow-up answer — if this "
        "becomes true the harm argument for NULL over restore has to be re-established, "
        "not quietly dropped"
    )


# --- 4. the ownership measurement, re-run as an assertion -----------------------------------


#: Every production shape that reaches an attach site: the generic chain's traversal->suffix
#: boundary, the polars twin, the all-calls (``let()`` body) run, and the Cypher WITH re-entry
#: that ``gfql_unified`` seeds.
_OWNERSHIP_CORPUS: List[Any] = [
    BOUNDARY_OPS,
    [n(name="a"), e_forward(name="r"), n(name="b"), rows(table="edges")],
    [rows(table="nodes")],
    "MATCH (a {kind:'a'}) WITH a MATCH (a)-[*]->(b) RETURN count(*) AS c",
    "MATCH (a {kind:'a'}) WITH a MATCH (a)-[]->(b) WITH b MATCH (b)-[]->(c) RETURN count(*) AS c",
]


@pytest.mark.parametrize("engine", ALL_ENGINES)
def test_no_execution_frame_inherits_a_context_it_did_not_set(
    engine: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every context value is OWNED by the frame that sets it — the scoping claim, measured.

    ``attach`` is wrapped to record any call whose incoming graph already carries a value the
    call itself does not supply. That is the ONE situation in which "restore the previous
    value" and "null it" differ for a production path, so a non-empty record means the design
    question of #1793 has to be reopened. Asserted rather than trusted, because the answer is
    a property of every execution path, not of the three lines in ``exec_context.py``.
    """
    _require(engine)
    import graphistry.compute.gfql.exec_context as exec_context
    import graphistry.compute.gfql_unified as gfql_unified

    inherited: List[str] = []
    real_attach = exec_context.attach_row_exec_context

    def recording_attach(
        g: Plottable, *, start_nodes: Any = None, rows_base_graph: Any = None
    ) -> Plottable:
        if start_nodes is None and g._gfql_start_nodes is not None:
            inherited.append("start_nodes")
        if rows_base_graph is None and g._gfql_rows_base_graph is not None:
            inherited.append("rows_base_graph")
        return real_attach(g, start_nodes=start_nodes, rows_base_graph=rows_base_graph)

    # Both bindings: the two chain surfaces import it inside the function (module attribute),
    # gfql_unified imports it by name at module scope.
    monkeypatch.setattr(exec_context, "attach_row_exec_context", recording_attach)
    monkeypatch.setattr(gfql_unified, "attach_row_exec_context", recording_attach)

    for query in _OWNERSHIP_CORPUS:
        try:
            _graph(engine).gfql(list(query) if isinstance(query, list) else query, engine=engine)
        except NotImplementedError:
            continue  # an honest per-engine decline is not an ownership signal

    assert inherited == [], (
        f"[{engine}] an execution frame inherited a row context it did not set "
        f"({sorted(set(inherited))}) — clear-to-None now DESTROYS an outer scope's value on "
        "the result, so #1793's choice must be re-decided (see this module's docstring)"
    )
