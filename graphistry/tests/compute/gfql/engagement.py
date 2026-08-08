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
from typing import Any, Dict, List, Optional

from graphistry.compute.gfql.index.api import index_trace


def col_stats_decisions(g: Any, query: str, *, engine: str = "pandas") -> List[Dict[str, Any]]:
    """Every column-stat fact decision made while running ``query`` on ``g``."""
    with index_trace() as steps:
        g.gfql(query, engine=engine)
    return [dict(s) for s in steps if s.get("op") == "col_stats"]


def assert_col_stats(
    g: Any,
    query: str,
    *,
    engine: str = "pandas",
    served: Optional[bool] = None,
    outcomes: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
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
