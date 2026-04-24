"""Tests for binder's ``expr_tree`` walking (#1200 slice 2).

When the parser populates ``WhereClause.expr_tree`` with a structural
boolean tree, the binder's ``_where_predicates`` flattens top-level AND
conjuncts and emits one ``BoundPredicate`` per conjunct — replacing the
previous behavior of one giant ``BoundPredicate`` carrying the full
expression text.

Covers:
- Hand-built ``BooleanExpr`` trees through the binder helpers in
  isolation (no parser dependency).
- End-to-end queries that trigger ``generic_where_clause`` so
  ``expr_tree`` is actually populated.
- Backward-compat: paths that leave ``expr_tree=None`` produce the same
  BoundPredicate list as before.
"""
from __future__ import annotations

from graphistry.compute.gfql.cypher.ast import (
    BooleanExpr,
    ExpressionText,
    SourceSpan,
    WhereClause,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import (
    FrontendBinder,
    _boolean_expr_to_text,
    _flatten_top_level_ands,
    _where_predicates,
)
from graphistry.compute.gfql.ir.compilation import PlanContext


# ---------------------------------------------------------------------------
# Hand-built helpers (no parser)
# ---------------------------------------------------------------------------


def _span(start: int = 0, end: int = 0) -> SourceSpan:
    return SourceSpan(line=1, column=1, end_line=1, end_column=1, start_pos=start, end_pos=end)


def _atom(text: str) -> BooleanExpr:
    return BooleanExpr(op="atom", span=_span(), atom_text=text, atom_span=_span())


def test_flatten_single_atom_returns_one_conjunct() -> None:
    atom = _atom("a > 1")
    assert _flatten_top_level_ands(atom) == [atom]


def test_flatten_left_associative_and_chain() -> None:
    # and(and(a, b), c) → [a, b, c]
    a, b, c = _atom("a"), _atom("b"), _atom("c")
    inner = BooleanExpr(op="and", span=_span(), left=a, right=b)
    outer = BooleanExpr(op="and", span=_span(), left=inner, right=c)
    assert _flatten_top_level_ands(outer) == [a, b, c]


def test_flatten_right_associative_and_chain() -> None:
    a, b, c = _atom("a"), _atom("b"), _atom("c")
    inner = BooleanExpr(op="and", span=_span(), left=b, right=c)
    outer = BooleanExpr(op="and", span=_span(), left=a, right=inner)
    assert _flatten_top_level_ands(outer) == [a, b, c]


def test_flatten_does_not_descend_through_or() -> None:
    a, b = _atom("a"), _atom("b")
    or_expr = BooleanExpr(op="or", span=_span(), left=a, right=b)
    # OR stops the recursion — the whole OR is one conjunct.
    assert _flatten_top_level_ands(or_expr) == [or_expr]


def test_flatten_and_with_or_subtree_keeps_or_intact() -> None:
    # and(a, or(b, c)) → [a, or(b, c)]
    a, b, c = _atom("a"), _atom("b"), _atom("c")
    or_expr = BooleanExpr(op="or", span=_span(), left=b, right=c)
    and_expr = BooleanExpr(op="and", span=_span(), left=a, right=or_expr)
    result = _flatten_top_level_ands(and_expr)
    assert result == [a, or_expr]


def test_flatten_does_not_descend_through_not() -> None:
    a = _atom("a")
    not_expr = BooleanExpr(op="not", span=_span(), left=a)
    assert _flatten_top_level_ands(not_expr) == [not_expr]


def test_text_atom_emits_atom_text() -> None:
    assert _boolean_expr_to_text(_atom("n.x > 1")) == "n.x > 1"


def test_text_or_two_atoms_glues_with_or_keyword() -> None:
    a, b = _atom("a"), _atom("b")
    or_expr = BooleanExpr(op="or", span=_span(), left=a, right=b)
    assert _boolean_expr_to_text(or_expr) == "a OR b"


def test_text_xor_two_atoms_glues_with_xor_keyword() -> None:
    a, b = _atom("a"), _atom("b")
    xor_expr = BooleanExpr(op="xor", span=_span(), left=a, right=b)
    assert _boolean_expr_to_text(xor_expr) == "a XOR b"


def test_text_not_atom_prefixes_keyword() -> None:
    a = _atom("a")
    not_expr = BooleanExpr(op="not", span=_span(), left=a)
    assert _boolean_expr_to_text(not_expr) == "NOT a"


def test_text_not_branch_wraps_operand_in_parens() -> None:
    # NOT (a OR b) — branch operand requires parens.
    a, b = _atom("a"), _atom("b")
    or_expr = BooleanExpr(op="or", span=_span(), left=a, right=b)
    not_expr = BooleanExpr(op="not", span=_span(), left=or_expr)
    assert _boolean_expr_to_text(not_expr) == "NOT (a OR b)"


def test_text_or_with_branch_operand_wraps_in_parens() -> None:
    # a OR (b AND c) — right side parenthesized to disambiguate.
    a, b, c = _atom("a"), _atom("b"), _atom("c")
    and_expr = BooleanExpr(op="and", span=_span(), left=b, right=c)
    or_expr = BooleanExpr(op="or", span=_span(), left=a, right=and_expr)
    assert _boolean_expr_to_text(or_expr) == "a OR (b AND c)"


# ---------------------------------------------------------------------------
# _where_predicates with hand-built WhereClause
# ---------------------------------------------------------------------------


def test_where_predicates_with_expr_tree_emits_one_per_conjunct() -> None:
    # and(and(a, b), c) → 3 BoundPredicates, one per conjunct.
    a, b, c = _atom("a > 1"), _atom("b > 2"), _atom("c > 3")
    inner = BooleanExpr(op="and", span=_span(), left=a, right=b)
    outer = BooleanExpr(op="and", span=_span(), left=inner, right=c)
    where = WhereClause(
        predicates=(),
        span=_span(),
        expr=ExpressionText(text="a > 1 AND b > 2 AND c > 3", span=_span()),
        expr_tree=outer,
    )
    bps = _where_predicates(where)
    assert [bp.expression for bp in bps] == ["a > 1", "b > 2", "c > 3"]


def test_where_predicates_with_or_root_emits_single_compound() -> None:
    # or(a, b) is one top-level conjunct (OR can't be split).
    a, b = _atom("a"), _atom("b")
    or_expr = BooleanExpr(op="or", span=_span(), left=a, right=b)
    where = WhereClause(
        predicates=(),
        span=_span(),
        expr=ExpressionText(text="a OR b", span=_span()),
        expr_tree=or_expr,
    )
    bps = _where_predicates(where)
    assert [bp.expression for bp in bps] == ["a OR b"]


def test_where_predicates_with_expr_tree_none_falls_back_to_expr_text() -> None:
    # Backward compat — expr_tree absent → same one-BoundPredicate path.
    where = WhereClause(
        predicates=(),
        span=_span(),
        expr=ExpressionText(text="raw expression text", span=_span()),
        expr_tree=None,
    )
    bps = _where_predicates(where)
    assert [bp.expression for bp in bps] == ["raw expression text"]


def test_where_predicates_with_no_expr_at_all_returns_empty() -> None:
    where = WhereClause(predicates=(), span=_span(), expr=None, expr_tree=None)
    assert _where_predicates(where) == []


# ---------------------------------------------------------------------------
# End-to-end through the parser + binder
# ---------------------------------------------------------------------------


def _bind_predicates(query: str) -> list:
    parsed = parse_cypher(query)
    bound = FrontendBinder().bind(parsed, PlanContext())
    # Predicates live on the BoundQueryPart for the matching MATCH/WHERE.
    parts = [p for p in bound.query_parts if p.predicates]
    if not parts:
        return []
    return [p.expression for p in parts[0].predicates]


def test_e2e_or_where_produces_single_compound_bound_predicate() -> None:
    # WHERE (a) OR (b) routes through generic_where_clause → expr_tree
    # populated → root op == 'or' → one conjunct → one BoundPredicate.
    exprs = _bind_predicates("MATCH (n) WHERE (n.x > 1) OR (n.y < 2) RETURN n")
    assert len(exprs) == 1
    assert "OR" in exprs[0].upper()


def test_e2e_pure_and_of_comparables_unchanged() -> None:
    # Routes through where_predicates (structured); expr_tree=None.
    # Two structured WherePredicates → two BoundPredicates as before.
    # Verify the routing invariant directly so a future parser change
    # that accidentally populates expr_tree on this path is caught,
    # rather than masked by the count happening to match.
    from graphistry.compute.gfql.cypher.ast import CypherQuery
    parsed = parse_cypher("MATCH (n) WHERE n.x > 1 AND n.y < 2 RETURN n")
    assert isinstance(parsed, CypherQuery)
    assert parsed.where is not None
    assert parsed.where.expr_tree is None, (
        "pure AND of comparables must route through where_predicates, "
        "not generic_where_clause"
    )
    bound = FrontendBinder().bind(parsed, PlanContext())
    parts = [p for p in bound.query_parts if p.predicates]
    assert parts and len(parts[0].predicates) == 2


def test_e2e_not_where_produces_single_not_bound_predicate() -> None:
    exprs = _bind_predicates("MATCH (n) WHERE NOT n.x > 1 RETURN n")
    assert len(exprs) == 1
    assert "NOT" in exprs[0].upper()


def test_e2e_top_level_and_with_or_subtree_splits_into_two_conjuncts() -> None:
    # ((a) OR (b)) AND c — top-level AND, but left side has OR.
    # Lark routes the whole thing through `expr` because of the OR,
    # producing and_op(or_op(...), c).  Binder flattens AND → 2 conjuncts.
    exprs = _bind_predicates(
        "MATCH (n) WHERE ((n.a > 1) OR (n.b > 2)) AND n.c > 3 RETURN n"
    )
    assert len(exprs) == 2
    # First conjunct is the OR sub-tree; second is the bare comparison.
    or_exprs = [e for e in exprs if "OR" in e.upper()]
    other_exprs = [e for e in exprs if "OR" not in e.upper()]
    assert len(or_exprs) == 1
    assert len(other_exprs) == 1
    assert other_exprs[0] == "n.c > 3"
