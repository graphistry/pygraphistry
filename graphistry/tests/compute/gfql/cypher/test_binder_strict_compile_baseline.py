"""Regression-pin baselines for the future strict_name_resolution rollout (#1357).

The post-normalize binder pass at ``lowering.py:8400`` runs in **loose**
mode today. The eventual goal of #1357 is to flip it to
``strict_name_resolution=True`` so alias-scope enforcement is centralized
at the binder layer (validator parity). A discovery flip exposed 75 test
failures across multiple binder-coverage gaps:

- namespaced function calls — ``duration.inSeconds(...)``, ``time()``,
  ``localtime()``, ``date()``, ``datetime()`` parsed as
  ``alias.property``.
- quantifier predicates — ``all(x IN list WHERE …)``, ``any``, ``none``,
  ``single``: comprehension-scoped ``x`` not modeled.
- list comprehensions — same scope-modeling gap.
- CALL/YIELD scope — YIELD aliases must survive the
  prepass→normalize→bind cycle.
- post-WITH UNWIND traversal — ``_bind_graph_sequence`` iterates
  ``ast.unwinds`` *before* WITH stages, so ``UNWIND`` references against
  WITH-projected aliases fail strict mode.

This module pins the **current loose-mode behavior** at the
``compile_cypher_query`` boundary for representative shapes from each
gap. When a follow-up PR closes one of those gaps and is ready to flip
strict mode, the corresponding test here should be updated to assert
*rejection* instead of admit. The test names + docstrings act as
regression detectors so the flip-readiness state is observable.

See ``plans/1357-binder-strict-name-resolution/research/discovery-flip-blast-radius.md``.
"""

from __future__ import annotations

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import PlanContext


# ---------------------------------------------------------------------------
# Loose-mode admits (will flip to strict-rejection in a follow-up PR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        # Namespaced builtins — the binder's _PROPERTY_RE matches
        # ``duration.inSeconds`` and routes through the alias-scope check;
        # in strict mode this would raise on unresolved alias 'duration'.
        "RETURN duration.inSeconds(localtime(), localtime()) AS duration",
        "RETURN time() AS t",
        "RETURN date() AS d",
        "RETURN datetime() AS dt",
        "RETURN localtime() AS lt",
    ],
)
def test_loose_mode_admits_namespaced_builtin_calls(query: str) -> None:
    """Loose binder admits namespaced builtins; strict would reject as
    unresolved 'duration'/'time'/etc. Future fix: teach the binder to
    recognize known builtin namespaces."""
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert bound.semantic_table.variables


@pytest.mark.parametrize(
    "query",
    [
        # Quantifier predicates — binder doesn't model x as a
        # comprehension-local; in strict mode it raises on unresolved 'x'.
        "MATCH (n) WHERE all(x IN n.labels WHERE x = 'A') RETURN n",
        "MATCH (n) WHERE any(x IN n.labels WHERE x = 'B') RETURN n",
        "MATCH (n) WHERE none(x IN n.labels WHERE x = 'C') RETURN n",
        "MATCH (n) WHERE single(x IN n.labels WHERE x = 'D') RETURN n",
    ],
)
def test_loose_mode_admits_quantifier_predicates(query: str) -> None:
    """Loose binder admits quantifier predicates; strict rejects 'x'.
    Future fix: bind comprehension-scoped locals before evaluating the
    predicate body."""
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert "n" in bound.semantic_table.variables


def test_loose_mode_admits_post_with_unwind_against_carried_alias() -> None:
    """``WITH collect(b1) AS bees UNWIND bees AS b2`` — loose binder
    iterates ast.unwinds before WITH stages so ``bees`` looks unresolved
    at the UNWIND. Strict rejects. Future fix: interleave UNWIND/WITH/MATCH
    by AST text position in ``_bind_graph_sequence``."""
    query = (
        "MATCH (root:S)-[:X]->(b1:B) "
        "WITH collect(b1) AS bees "
        "UNWIND bees AS b2 "
        "MATCH (b2)-[:Y]->(c:C) "
        "RETURN c.id AS id"
    )
    # Loose admit — no exception; downstream lowering may still reject for
    # other reasons but the binder does not.
    FrontendBinder().bind(parse_cypher(query), PlanContext())


def test_loose_mode_admits_call_yield_then_return_yield_alias() -> None:
    """``CALL graphistry.degree() YIELD nodeId RETURN nodeId`` — loose
    binder admits; strict rejects because the prepass→normalize cycle does
    not propagate YIELD aliases into the post-normalize bind scope. Future
    fix: ensure YIELD aliases survive the cycle."""
    query = "CALL graphistry.degree() YIELD nodeId RETURN nodeId"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert "nodeId" in bound.semantic_table.variables


# ---------------------------------------------------------------------------
# Loose-mode admits caught by downstream guards (already rejected, but not
# by the binder). These are the "what strict would catch but downstream
# already does" baselines — strict flip would *move* the rejection site.
# ---------------------------------------------------------------------------


def test_compile_rejects_unresolved_return_alias_via_projection_planner() -> None:
    """``MATCH (a) RETURN ghost`` — currently rejected by
    projection_planning E108 ("Unknown Cypher alias 'ghost' in RETURN
    clause"). Strict binder would reject earlier with E204. The intent
    (reject unresolved alias in RETURN) is preserved either way."""
    query = "MATCH (a) RETURN ghost"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    # Pin the alias appears in the error context (stable across either
    # rejection site). Drop loose substring matching so a future strict
    # flip moving the rejection from E108→E204 stays observable.
    assert exc_info.value.context.get("value") == "ghost"


def test_compile_rejects_unresolved_where_alias_via_where_evaluator() -> None:
    """``MATCH (a) WHERE ghost.foo = 1 RETURN a`` — rejected by the WHERE
    evaluator E108 when reaching the literal-where path. Strict binder
    would reject earlier with E204."""
    query = "MATCH (a) WHERE ghost.foo = 1 RETURN a"
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(query)
    # Either field is acceptable depending on rejection site.
    err_str = str(exc_info.value)
    assert "ghost" in err_str
