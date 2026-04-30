"""Tests for ``BooleanExpr`` structural tree in parsed WHERE clauses.

The parser now exposes Lark's already-computed ``and_op`` / ``or_op`` /
``xor_op`` / ``not_op`` tree structurally as ``WhereClause.expr_tree``.
These tests lock the tree shape so downstream consumers (binder,
predicate pushdown) can safely migrate off text re-parsing.

Routing note: pure AND conjunctions of comparable predicates route
through the structured ``where_predicates`` grammar rule and do **not**
populate ``expr_tree`` — those use ``WhereClause.predicates`` instead.
``expr_tree`` is populated only when the WHERE body forces Lark into
the generic ``expr`` path: OR / XOR / NOT operators, or parenthesized
subexpressions that mix operator flavors.

Pre-existing grammar quirk: the LALR parser cannot disambiguate bare
``OR`` between two raw comparable predicates (``WHERE a = 1 OR b = 2``
fails); parenthesized forms (``WHERE (a = 1) OR (b = 2)``) work.  Tests
use the parenthesized form to probe the expected boolean-tree output.
"""
from __future__ import annotations

from typing import cast

from graphistry.compute.gfql.cypher import (
    BooleanExpr,
    CypherQuery,
    parse_cypher,
)


def _parsed_where(query: str) -> CypherQuery:
    return cast(CypherQuery, parse_cypher(query))


def _expr_tree(query: str) -> BooleanExpr:
    parsed = _parsed_where(query)
    assert parsed.where is not None, "parsed query has no WHERE clause"
    assert parsed.where.expr_tree is not None, (
        f"expected expr_tree for boolean-operator WHERE, got None.  "
        f"Predicates={parsed.where.predicates}"
    )
    return parsed.where.expr_tree


# ---------------------------------------------------------------------------
# No expr_tree for structured paths
# ---------------------------------------------------------------------------


def test_single_predicate_where_clause_has_no_expr_tree() -> None:
    parsed = _parsed_where("MATCH (n) WHERE n.age > 5 RETURN n")
    assert parsed.where is not None
    assert parsed.where.expr_tree is None


def test_pure_and_of_comparables_routes_through_where_predicates() -> None:
    parsed = _parsed_where("MATCH (n) WHERE n.x > 1 AND n.y < 2 RETURN n")
    assert parsed.where is not None
    assert len(parsed.where.predicates) == 2
    assert parsed.where.expr_tree is None


def test_structured_label_narrowing_does_not_populate_expr_tree() -> None:
    parsed = _parsed_where("MATCH (n) WHERE n:Admin AND n:Active RETURN n")
    assert parsed.where is not None
    assert len(parsed.where.predicates) == 2
    assert parsed.where.expr_tree is None


# ---------------------------------------------------------------------------
# Single-operator shapes that force the expr path
# ---------------------------------------------------------------------------


def test_or_op_produces_or_tree() -> None:
    # Outer parens are transparent (see ``grouped_expr`` passthrough);
    # atoms carry the inner predicate text without surrounding parens.
    tree = _expr_tree("MATCH (n) WHERE (n.x > 1) OR (n.y < 2) RETURN n")
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.right is not None and tree.right.op == "atom"
    assert tree.left.atom_text == "n.x > 1"
    assert tree.right.atom_text == "n.y < 2"


def test_xor_op_produces_xor_tree() -> None:
    tree = _expr_tree("MATCH (n) WHERE (n.x > 1) XOR (n.y < 2) RETURN n")
    assert tree.op == "xor"
    assert tree.left is not None and tree.left.atom_text == "n.x > 1"
    assert tree.right is not None and tree.right.atom_text == "n.y < 2"


def test_not_op_produces_not_tree() -> None:
    tree = _expr_tree("MATCH (n) WHERE NOT n.x > 1 RETURN n")
    assert tree.op == "not"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.left.atom_text == "n.x > 1"
    assert tree.right is None


# ---------------------------------------------------------------------------
# Multi-operator precedence
# ---------------------------------------------------------------------------


def test_and_binds_tighter_than_or_when_mixed() -> None:
    # (a) OR (b AND c) — expr path because top-level OR.
    # Outer: or_op.  Parens are transparent, so atoms/subtrees appear
    # unwrapped.  The right side ``(b AND c)`` is a Lark and_op Tree —
    # after my and_op transformer fires, it becomes a BooleanExpr and
    # passes through grouped_expr unchanged.
    tree = _expr_tree("MATCH (n) WHERE (n.a > 1) OR (n.b > 2 AND n.c > 3) RETURN n")
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.left.atom_text == "n.a > 1"
    assert tree.right is not None and tree.right.op == "and"
    assert tree.right.left is not None and tree.right.left.atom_text == "n.b > 2"
    assert tree.right.right is not None and tree.right.right.atom_text == "n.c > 3"


def test_not_binds_tighter_than_or() -> None:
    # NOT (a) OR (b) → parses as ((NOT a) OR b)
    tree = _expr_tree("MATCH (n) WHERE NOT (n.a > 1) OR (n.b > 2) RETURN n")
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "not"
    assert tree.right is not None and tree.right.op == "atom"


def test_parenthesized_or_inside_and_routes_through_expr() -> None:
    # (a OR b) AND c — top-level AND combined with OR subexpression forces expr.
    # Outer: and_op (because left is a complex expression, not plain comparable).
    tree = _expr_tree("MATCH (n) WHERE ((n.a > 1) OR (n.b > 2)) AND n.c > 3 RETURN n")
    assert tree.op == "and"
    assert tree.left is not None and tree.left.op == "or"
    assert tree.right is not None and tree.right.op == "atom"
    assert tree.right.atom_text == "n.c > 3"


# ---------------------------------------------------------------------------
# Span coverage
# ---------------------------------------------------------------------------


def test_atom_span_matches_source_slice() -> None:
    # Atom spans must quote exactly the inner predicate text from source.
    # grouped_expr is transparent, so the atom span covers "n.x > 1"
    # (without surrounding parens).
    query = "MATCH (n) WHERE (n.x > 1) OR (n.y < 2) RETURN n"
    tree = _expr_tree(query)
    assert tree.left is not None and tree.left.atom_span is not None
    lhs_span = tree.left.atom_span
    assert query[lhs_span.start_pos:lhs_span.end_pos] == tree.left.atom_text


def test_branch_span_covers_full_subexpression() -> None:
    # Strict span bounds — no .strip() — so any future drift is caught.
    query = "MATCH (n) WHERE (n.x > 1) OR (n.y < 2) RETURN n"
    tree = _expr_tree(query)
    assert query[tree.span.start_pos:tree.span.end_pos] == "(n.x > 1) OR (n.y < 2)"


# ---------------------------------------------------------------------------
# Deeper nesting and additional precedence combinations
# ---------------------------------------------------------------------------


def test_deeply_nested_or_grouping_preserves_structure() -> None:
    # ((a OR b) OR (c OR d)) — nested OR-of-ORs; grouped_expr must pass
    # through cleanly at every level.
    tree = _expr_tree(
        "MATCH (n) WHERE ((n.a > 1) OR (n.b > 2)) OR ((n.c > 3) OR (n.d > 4)) RETURN n"
    )
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "or"
    assert tree.right is not None and tree.right.op == "or"
    # Leaves
    assert tree.left.left is not None and tree.left.left.atom_text == "n.a > 1"
    assert tree.left.right is not None and tree.left.right.atom_text == "n.b > 2"
    assert tree.right.left is not None and tree.right.left.atom_text == "n.c > 3"
    assert tree.right.right is not None and tree.right.right.atom_text == "n.d > 4"


def test_nested_not_not_produces_nested_not_nodes() -> None:
    # NOT NOT a — two unary NOT applications.
    tree = _expr_tree("MATCH (n) WHERE NOT NOT n.a > 1 RETURN n")
    assert tree.op == "not"
    assert tree.left is not None and tree.left.op == "not"
    assert tree.left.left is not None and tree.left.left.atom_text == "n.a > 1"


def test_and_xor_mixed_precedence_via_parens() -> None:
    # AND binds tighter than XOR: (a) XOR (b AND c) → xor(a, and(b, c)).
    # The bare form ``a XOR b AND c`` hits the same LALR ambiguity as bare
    # ``a OR b``; parenthesizing the XOR operands lets it parse cleanly.
    tree = _expr_tree("MATCH (n) WHERE (n.a > 1) XOR (n.b > 2 AND n.c > 3) RETURN n")
    assert tree.op == "xor"
    assert tree.left is not None and tree.left.atom_text == "n.a > 1"
    assert tree.right is not None and tree.right.op == "and"


# ---------------------------------------------------------------------------
# Primitive literal fallback atoms
# ---------------------------------------------------------------------------


def test_literal_boolean_atom_text_uses_cypher_keywords() -> None:
    parsed = _parsed_where("MATCH (n) WHERE true XOR n.x > 1 RETURN n")
    assert parsed.where is not None and parsed.where.expr_tree is not None
    tree = parsed.where.expr_tree
    assert tree.op == "xor"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.left.atom_text == "true"
    # Right operand is a comparable with a Lark Tree span — accurate slice.
    assert tree.right is not None and tree.right.atom_text == "n.x > 1"


def test_literal_false_atom_text_uses_cypher_keyword() -> None:
    parsed = _parsed_where("MATCH (n) WHERE false OR n.x > 1 RETURN n")
    assert parsed.where is not None and parsed.where.expr_tree is not None
    tree = parsed.where.expr_tree
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.left.atom_text == "false"


def test_literal_null_atom_text_uses_cypher_keyword() -> None:
    parsed = _parsed_where("MATCH (n) WHERE null OR n.x > 1 RETURN n")
    assert parsed.where is not None and parsed.where.expr_tree is not None
    tree = parsed.where.expr_tree
    assert tree.op == "or"
    assert tree.left is not None and tree.left.op == "atom"
    assert tree.left.atom_text == "null"


# ---------------------------------------------------------------------------
# boolean_expr_to_text contract for the op == "pattern" branch
# ---------------------------------------------------------------------------


def test_boolean_expr_to_text_emits_atom_text_for_pattern_op() -> None:
    # Pattern leaves are normally lifted out of expr_tree by
    # _split_top_level_and_pattern_leaves before the binder walks the
    # tree, so this branch is unreachable in production.  The unit test
    # locks the contract explicitly so a future code path that DOES
    # reach boolean_expr_to_text with a pattern leaf gets the raw
    # pattern source rather than the empty-string fallthrough.
    from graphistry.compute.gfql.cypher.ast import SourceSpan
    from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text

    span = SourceSpan(line=1, column=1, end_line=1, end_column=10, start_pos=0, end_pos=10)
    pattern_leaf = BooleanExpr(
        op="pattern",
        span=span,
        atom_text="(a)-->(b)",
        atom_span=span,
    )
    assert boolean_expr_to_text(pattern_leaf) == "(a)-->(b)"
