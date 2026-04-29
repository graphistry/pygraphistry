"""Surface-text and AND-flattening utilities for ``BooleanExpr`` trees.

Lifted from ``frontends/cypher/binder.py`` (see #1207) to a shared location
so consumers outside the binder (lowering, ast_normalizer) can reconstruct
WHERE-body text from ``WhereClause.expr_tree`` instead of reading the
soon-to-be-removed ``WhereClause.expr`` field.  Tracked in #1213.
"""
from __future__ import annotations

from typing import Dict, List

from graphistry.compute.gfql.cypher.ast import BooleanExpr

__all__ = ("boolean_expr_to_text", "flatten_top_level_ands")


_BOOLEAN_OP_KEYWORD: Dict[str, str] = {
    "and": "AND",
    "or": "OR",
    "xor": "XOR",
}


def flatten_top_level_ands(expr: BooleanExpr) -> List[BooleanExpr]:
    """Flatten left-associated AND chains into a flat list of conjuncts.

    ``and(and(a, b), c)`` → ``[a, b, c]``.  Non-AND nodes stop the
    recursion: ``or(a, b)`` returns ``[or(a, b)]`` (one conjunct),
    ``not(a)`` returns ``[not(a)]``, atoms return ``[atom]``.

    Used by binder ``_where_predicates`` to emit one ``BoundPredicate``
    per top-level AND conjunct so downstream passes (predicate pushdown)
    don't have to re-parse compound expression text.
    """
    if expr.op != "and":
        return [expr]
    conjuncts: List[BooleanExpr] = []
    if expr.left is not None:
        conjuncts.extend(flatten_top_level_ands(expr.left))
    if expr.right is not None:
        conjuncts.extend(flatten_top_level_ands(expr.right))
    return conjuncts


def boolean_expr_to_text(expr: BooleanExpr) -> str:
    """Reconstruct surface text for a boolean-expression subtree.

    Atoms emit ``atom_text`` directly.  Branches stringify recursively
    with parentheses around any branch operand to keep operator
    precedence unambiguous when the result is later parsed back as a
    single conjunct.  ``NOT`` prefixes its operand; binary ops produce
    ``"L OP R"``.

    Inherits the slice-1 known limitation for primitive literal atoms
    (``str(True) == "True"`` rather than Cypher ``"true"``); that is a
    follow-up under #1200 to be addressed when literal transformers
    gain span-carrying wrappers.
    """
    if expr.op == "atom":
        return expr.atom_text or ""
    if expr.op == "pattern":
        # Unreachable today: top-level AND leaves are lifted out by
        # ``_split_top_level_and_pattern_leaves`` before the binder walks
        # the tree, and patterns nested under NOT/OR/XOR are rejected
        # earlier with E108 errors.  Contract for the defensive branch:
        # emit raw pattern source for round-trippability.
        return expr.atom_text or ""
    if expr.op == "not":
        operand = boolean_expr_to_text(expr.left) if expr.left is not None else ""
        if expr.left is not None and expr.left.op != "atom":
            operand = f"({operand})"
        return f"NOT {operand}"
    keyword = _BOOLEAN_OP_KEYWORD.get(expr.op)
    if keyword is None or expr.left is None or expr.right is None:
        return expr.atom_text or ""
    left = boolean_expr_to_text(expr.left)
    right = boolean_expr_to_text(expr.right)
    if expr.left.op != "atom":
        left = f"({left})"
    if expr.right.op != "atom":
        right = f"({right})"
    return f"{left} {keyword} {right}"
