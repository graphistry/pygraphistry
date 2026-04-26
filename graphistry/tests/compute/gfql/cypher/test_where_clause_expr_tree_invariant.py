"""Parser shape contract for ``WhereClause`` (post-#1213).

After the ``WhereClause.expr`` field is removed, the surviving routing
contract has three shapes:

  - **Structured path**: ``predicates`` populated, ``expr_tree is None``.
    Pure AND of comparable predicates, or label-narrowed AND.
  - **Tree path**: ``predicates == ()``, ``expr_tree is not None``.
    Generic boolean expression (OR / XOR / NOT / paren).
  - **Mixed path**: BOTH ``predicates`` (a single ``WherePatternPredicate``)
    AND ``expr_tree`` populated.  Fires for ``WHERE pattern AND expr``.

Every WhereClause must populate at least one of the two fields with a
non-empty value (an empty WHERE is a parse error).  Locked here so future
grammar work can't silently regress the routing.

(Pre-#1213 this file asserted ``(expr is None) == (expr_tree is None)``;
that invariant became degenerate when the ``expr`` field was removed.)
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


@pytest.mark.parametrize("query, expected_shape", [
    # Structured (where_predicates rule): AND of comparable predicates
    ("MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n", "structured"),
    # Structured (label-narrowing lift in generic_where_clause)
    ("MATCH (n) WHERE n:Admin AND n:User RETURN n", "structured"),
    # Structured: single comparable predicate
    ("MATCH (n) WHERE n.x = 1 RETURN n", "structured"),
    # Tree path: parenthesized OR (bare top-level OR is rejected)
    ("MATCH (n) WHERE (n.x = 1 OR n.y = 2) RETURN n", "tree"),
    # Tree path: XOR
    ("MATCH (n) WHERE n:Admin XOR n:Active RETURN n", "tree"),
    # Tree path: NOT
    ("MATCH (n) WHERE NOT n.x = 1 AND n.y = 2 RETURN n", "tree"),
    # Tree path: parenthesized OR before AND
    ("MATCH (n) WHERE (n.x = 1 OR n.y = 2) AND n.z = 3 RETURN n", "tree"),
    # Mixed path: WHERE pattern AND expr (predicates AND expr_tree both populated)
    ("MATCH (n) WHERE (n)-[]->(:Admin) AND n.active = true RETURN n", "mixed"),
])
def test_where_clause_routing_shape(query: str, expected_shape: str) -> None:
    where = _where(query)
    has_preds = len(where.predicates) >= 1
    has_tree = where.expr_tree is not None
    actual = (
        "structured" if (has_preds and not has_tree)
        else "tree" if (not has_preds and has_tree)
        else "mixed" if (has_preds and has_tree)
        else "empty"
    )
    assert actual == expected_shape, (
        f"Routing violated for {query!r}: predicates={where.predicates!r}, "
        f"expr_tree={where.expr_tree!r}; actual={actual!r}, expected={expected_shape!r}"
    )
    assert actual != "empty", "WHERE clause must populate predicates or expr_tree"


def test_mixed_pattern_and_expr_synthesizes_atom_tree() -> None:
    """``_mixed_where_clause`` synthesizes a single-atom tree carrying the expr text."""
    where = _where("MATCH (n) WHERE (n)-[]->(:Admin) AND n.active = true RETURN n")
    assert where.expr_tree is not None
    assert where.expr_tree.op == "atom"
    assert where.expr_tree.atom_text == "n.active = true"


def test_boolean_tree_preserves_structural_op() -> None:
    """Lark's parsed tree (and_op / or_op / xor_op / not_op) is preserved in expr_tree."""
    where = _where("MATCH (n) WHERE (n.x = 1 OR n.y = 2) RETURN n")
    assert where.expr_tree is not None
    assert where.expr_tree.op == "or"


def test_structured_predicates_have_no_tree() -> None:
    """Structured-path WhereClauses produce no expr_tree at all."""
    where = _where("MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n")
    assert where.expr_tree is None
    assert len(where.predicates) == 2
