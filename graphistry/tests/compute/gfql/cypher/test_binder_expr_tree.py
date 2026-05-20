"""Tests for binder consumption of ``WhereClause.expr_tree`` (#1200)."""

from __future__ import annotations

import pytest

from graphistry.compute.gfql.cypher.ast import CypherQuery
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.compilation import PlanContext


def _bind_predicates(query: str) -> list:
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    parts = [p for p in bound.query_parts if p.predicates]
    if not parts:
        return []
    return [p.expression for p in parts[0].predicates]


@pytest.mark.parametrize(
    ("query", "needle"),
    [
        ("MATCH (n) WHERE (n.x > 1) OR (n.y < 2) RETURN n", "OR"),
        ("MATCH (n) WHERE NOT n.x > 1 RETURN n", "NOT"),
    ],
)
def test_e2e_compound_where_produces_single_bound_predicate(query: str, needle: str) -> None:
    exprs = _bind_predicates(query)
    assert len(exprs) == 1
    assert needle in exprs[0].upper()


def test_e2e_pure_and_of_comparables_unchanged() -> None:
    parsed = parse_cypher("MATCH (n) WHERE n.x > 1 AND n.y < 2 RETURN n")
    assert isinstance(parsed, CypherQuery)
    assert parsed.where is not None
    assert parsed.where.expr_tree is None

    bound = FrontendBinder().bind(parsed, PlanContext())
    parts = [p for p in bound.query_parts if p.predicates]
    assert parts and len(parts[0].predicates) == 2


def test_e2e_top_level_and_with_or_subtree_splits_into_two_conjuncts() -> None:
    exprs = _bind_predicates(
        "MATCH (n) WHERE ((n.a > 1) OR (n.b > 2)) AND n.c > 3 RETURN n"
    )
    assert len(exprs) == 2
    assert len([expr for expr in exprs if "OR" in expr.upper()]) == 1
    assert len([expr for expr in exprs if "OR" not in expr.upper()]) == 1
    assert "n.c > 3" in exprs
