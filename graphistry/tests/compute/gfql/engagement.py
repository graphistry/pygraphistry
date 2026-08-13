"""Assert that an optimization actually FIRED, not merely that the answer is right.

WHY THIS EXISTS. GFQL's fast paths are contracted "same answer, faster": every
one of them falls back to a scan when it declines, so a completely dead
optimization still returns the correct result. Value tests are therefore
structurally blind to it -- measured in ``test_lowering.py``, 665 assertions
check values and 42 check engagement, and three real bugs slipped through that
gap (a boolean label form that never engaged, a fingerprint change that made
every fact look stale, and an opt-in flag that silently did nothing). None could
have failed a value test.

WHY NOT MONKEYPATCH. The usual way to write these -- patch a private callee on
its defining module and count calls -- is unreliable in a way that FAILS OPEN. If
another module did ``from .gfql_fast_paths import _execute_two_hop_count_fast_path``,
that module holds its own reference and patching the definition site never
intercepts; the pin then cannot fail, and the silence reads like a dead code
path. That false negative has already been produced once here. These helpers
read the public ``gfql_explain`` trace instead, so they observe the decision
where it is made and survive refactors of the private names.
"""
from typing import Any, Dict, List, Optional, cast

from graphistry.Engine import EngineAbstractType
from graphistry.Plottable import Plottable
from graphistry.compute.gfql.index.api import index_trace
from graphistry.compute.gfql.index.types import ColStatsOutcomeName, FastPathName


def col_stats_decisions(
    g: Plottable, query: str, *, engine: EngineAbstractType = "pandas"
) -> List[Dict[str, Any]]:  # hygiene-ok: explicit-any -- a trace step is a heterogeneous TypedDict by contract
    """Every column-stat fact decision made while running ``query`` on ``g``."""
    with index_trace() as steps:
        g.gfql(query, engine=engine)
    return [dict(s) for s in steps if s.get("op") == "col_stats"]


def assert_col_stats(
    g: Plottable,
    query: str,
    *,
    engine: EngineAbstractType = "pandas",
    served: Optional[bool] = None,
    outcomes: Optional[Dict[str, ColStatsOutcomeName]] = None,
) -> List[Dict[str, Any]]:  # hygiene-ok: explicit-any -- a trace step is a heterogeneous TypedDict by contract
    """Assert how the column-stat facts were USED, and return the decisions.

    ``served=True`` requires every decision to have skipped a scan; ``False``
    requires that none did. ``outcomes`` maps ``"<role>.<column>"`` to an expected
    outcome (``served`` / ``absent`` / ``stale`` / ``insufficient``) for finer
    checks. Failures name what actually happened, since "the optimization did not
    fire" is otherwise indistinguishable from "the test asserted nothing".
    """
    decisions = col_stats_decisions(g, query, engine=engine)
    assert decisions, (
        "no col_stats decisions were recorded -- the consult was never reached, "
        "so this assertion would have passed vacuously")

    if served is not None:
        actual = {d["decision_code"] for d in decisions}
        if served:
            assert all(d["served"] for d in decisions), (
                f"expected every fact consult to be served, got {sorted(actual)}")
        else:
            assert not any(d["served"] for d in decisions), (
                f"expected no fact consult to be served, got {sorted(actual)}")

    for key, expected in (outcomes or {}).items():
        role, _, column = key.partition(".")
        matching = [d for d in decisions if d["role"] == role and d["column"] == column]
        assert matching, f"no col_stats decision for {key!r}; saw " + str(
            sorted({f"{d['role']}.{d['column']}" for d in decisions}))
        got = {d["decision_code"] for d in matching}
        assert got == {f"col_stats_{expected}"}, (
            f"{key}: expected col_stats_{expected}, got {sorted(got)}")
    return decisions


def fast_path_decisions(
    g: Plottable, query: str, *, engine: EngineAbstractType = "pandas"
) -> Dict[FastPathName, bool]:
    """``{fast path name: served}`` for one query. A path absent from the map was
    never consulted (an earlier path short-circuited), which is different from
    consulted-and-declined -- so absence is not silently read as False."""
    with index_trace() as steps:
        g.gfql(query, engine=engine)
    return {cast(FastPathName, s["seam"]): bool(s["served"])
            for s in steps if s.get("op") == "fast_path"}


def assert_fast_path(
    g: Plottable, query: str, path: FastPathName, *, served: bool,
    engine: EngineAbstractType = "pandas",
) -> None:
    """Assert a named fast path served (or declined) for ``query``.

    This is the assertion a VALUE test structurally cannot make: every fast path
    falls back, so a dead one still returns the right answer.
    """
    seen = fast_path_decisions(g, query, engine=engine)
    assert path in seen, (
        f"{path!r} was never consulted for this query; consulted: {sorted(seen)}. "
        "An earlier fast path may have short-circuited.")
    assert seen[path] is served, (
        f"{path!r}: expected served={served}, got {seen[path]} (all: {seen})")
