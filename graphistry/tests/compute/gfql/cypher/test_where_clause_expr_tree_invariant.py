"""Parser invariant: ``(WhereClause.expr is None) == (WhereClause.expr_tree is None)``.

Locked under #1213 sub-PR A so subsequent slices can read text/span from
``expr_tree`` without checking both fields.  Covers every parser construction
site that previously populated ``expr`` without populating ``expr_tree``:

  - ``where_clause`` non-structured (single-atom) branch
  - ``generic_where_clause`` single-atom fallback (no boolean operators)
  - ``generic_where_clause`` boolean-tree path (already invariant-holding)
  - ``_mixed_where_clause`` (WHERE pattern AND expr) — both directions

Structured paths (``where_predicates`` rule, label-narrowing lift) keep
``expr is None`` and ``expr_tree is None``; those also satisfy the invariant.
"""
from __future__ import annotations

from typing import cast

import pytest

from graphistry.compute.gfql.cypher.ast import CypherQuery, WhereClause
from graphistry.compute.gfql.cypher.parser import parse_cypher


def _where(query: str) -> WhereClause:
    parsed = parse_cypher(query)
    parsed = cast(CypherQuery, parsed)
    assert parsed.where is not None
    return parsed.where


@pytest.mark.parametrize("query", [
    # Structured: AND of comparable predicates
    "MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n",
    # Structured (label-narrowing): AND of bare label predicates
    "MATCH (n) WHERE n:Admin AND n:User RETURN n",
    # Structured: single comparable predicate
    "MATCH (n) WHERE n.x = 1 RETURN n",
    # Generic boolean tree: parenthesized OR (bare top-level OR is rejected)
    "MATCH (n) WHERE (n.x = 1 OR n.y = 2) RETURN n",
    # Generic boolean tree: XOR
    "MATCH (n) WHERE n:Admin XOR n:Active RETURN n",
    # Generic boolean tree: NOT
    "MATCH (n) WHERE NOT n.x = 1 AND n.y = 2 RETURN n",
    # Generic boolean tree: parenthesized OR before AND
    "MATCH (n) WHERE (n.x = 1 OR n.y = 2) AND n.z = 3 RETURN n",
    # Mixed pattern + expr (pattern first)
    "MATCH (n) WHERE (n)-[]->(:Admin) AND n.active = true RETURN n",
])
def test_expr_and_expr_tree_invariant_holds(query: str) -> None:
    where = _where(query)
    assert (where.expr is None) == (where.expr_tree is None), (
        f"Invariant violated for {query!r}: "
        f"expr is {where.expr!r}, expr_tree is {where.expr_tree!r}"
    )


def test_mixed_pattern_and_expr_synthesizes_atom_tree() -> None:
    where = _where("MATCH (n) WHERE (n)-[]->(:Admin) AND n.active = true RETURN n")
    assert where.expr is not None
    assert where.expr_tree is not None
    assert where.expr_tree.op == "atom"
    assert where.expr_tree.atom_text == where.expr.text


def test_boolean_tree_preserves_structural_op() -> None:
    where = _where("MATCH (n) WHERE (n.x = 1 OR n.y = 2) RETURN n")
    assert where.expr_tree is not None
    # Generic path's atom synthesis kicks in only when items[0] is non-BooleanExpr;
    # an OR-bearing tree comes through as op="or".
    assert where.expr_tree.op == "or"


def test_structured_predicates_have_no_expr_or_tree() -> None:
    where = _where("MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n")
    assert where.expr is None
    assert where.expr_tree is None
    assert len(where.predicates) == 2
