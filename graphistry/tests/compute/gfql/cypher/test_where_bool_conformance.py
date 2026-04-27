"""WHERE boolean-shape conformance matrix (#1201).

Three-section suite locking shape contracts at each pipeline phase:

  1. Parser  — WhereClause.predicates (structured) vs .expr (raw text)
  2. Binder  — BoundQueryPart.predicates count and routing per shape
  3. Pushdown — split_top_level_and + is_null_rejecting for each shape's
                expression form (IR-level push/keep already in
                test_predicate_pushdown_pass.py; this file adds the
                expression-shape angle)

Shapes covered per issue #1201:
  A — A AND B
  B — A AND (B OR C)
  C — (A OR B) AND C
  D — NOT A AND B
  E — A XOR B
  F — label predicate mixed with property predicate
  G — quoted / backtick / bracket / brace containing AND text

Design: assertions target stable shape contracts, not implementation
strings, so tests survive the #1194 grammar-disambiguation and #1200
IR boolean-tree migrations with minimal churn.
"""
from __future__ import annotations

import pytest

from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.cypher._boolean_expr_text import boolean_expr_to_text
from graphistry.compute.gfql.cypher.ast import (
    CypherQuery,
    WhereClause,
)
from graphistry.compute.exceptions import GFQLSyntaxError
from graphistry.compute.gfql.expr_split import split_top_level_and
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.pushdown_safety import is_null_rejecting
from graphistry.compute.gfql.ir.types import BoundPredicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_where(cypher: str) -> WhereClause:
    parsed = parse_cypher(cypher)
    assert isinstance(parsed, CypherQuery)
    assert parsed.where is not None
    return parsed.where


def _bind(cypher: str) -> BoundIR:
    return FrontendBinder().bind(parse_cypher(cypher), PlanContext())


def _match_parts(bound: BoundIR) -> list[BoundQueryPart]:
    return [p for p in bound.query_parts if p.clause == "MATCH"]


def _pred(expr: str, refs: frozenset = frozenset()) -> BoundPredicate:
    return BoundPredicate(expression=expr, references=refs)


# ---------------------------------------------------------------------------
# Section 1 — Parser shape
# ---------------------------------------------------------------------------

class TestParserShape:
    """WhereClause routing: .predicates (structured) vs .expr (raw text)."""

    # Shape A: A AND B — pure AND of two property comparisons
    def test_a_and_b_lands_in_structured_predicates(self) -> None:
        w = _parse_where("MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n")
        assert w.expr_tree is None
        assert len(w.predicates) == 2

    # Shape A extended: three AND conjuncts
    def test_triple_and_lands_in_structured_predicates(self) -> None:
        w = _parse_where("MATCH (n) WHERE n.x = 1 AND n.y = 2 AND n.z = 3 RETURN n")
        assert w.expr_tree is None
        assert len(w.predicates) == 3

    # Shape B: A AND (B OR C) — parenthesised OR inside AND
    def test_and_with_paren_or_lands_in_and_tree(self) -> None:
        w = _parse_where(
            "MATCH (n) WHERE n.x = 1 AND (n.y = 2 OR n.z = 3) RETURN n"
        )
        assert w.expr_tree is not None
        assert w.expr_tree.op == "and"
        assert w.expr_tree.right is not None
        assert w.expr_tree.right.op == "or"

    # Shape C: (A OR B) AND C — OR before AND
    def test_paren_or_and_c_lands_in_tree(self) -> None:
        w = _parse_where(
            "MATCH (n) WHERE (n.x = 1 OR n.y = 2) AND n.z = 3 RETURN n"
        )
        assert w.expr_tree is not None

    # Shape D: NOT A AND B — NOT prefix
    def test_not_a_and_b_lands_in_tree(self) -> None:
        w = _parse_where("MATCH (n) WHERE NOT n.x = 1 AND n.y = 2 RETURN n")
        # NOT is not a structured WherePredicate form; expression falls to expr_tree
        assert w.expr_tree is not None

    # Shape E: A XOR B — XOR operator
    def test_xor_lands_in_tree(self) -> None:
        w = _parse_where("MATCH (n) WHERE n:Admin XOR n:Active RETURN n")
        assert w.expr_tree is not None
        assert "XOR" in boolean_expr_to_text(w.expr_tree).upper()
        assert w.predicates == ()

    # Shape F: label predicate AND property predicate → structured route
    def test_label_and_property_routes_to_structured_predicates(self) -> None:
        w = _parse_where("MATCH (n) WHERE n:Admin AND n.active = true RETURN n")
        assert w.expr_tree is None
        assert len(w.predicates) == 2

    # Shape G: quoted AND must not be counted as a predicate boundary
    def test_quoted_and_does_not_produce_extra_predicate(self) -> None:
        w = _parse_where(
            "MATCH (n) WHERE n.name = 'has AND text' AND n.x = 1 RETURN n"
        )
        # Parser grammar handles string literals; quoted AND is not a split point
        assert w.expr_tree is None
        assert len(w.predicates) == 2


# ---------------------------------------------------------------------------
# Section 2 — Binder shape
# ---------------------------------------------------------------------------

class TestBinderShape:
    """BoundQueryPart.predicates count and expression routing per shape."""

    # Shape A: A AND B → two separate BoundPredicates
    def test_a_and_b_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(_bind("MATCH (n) WHERE n.x = 1 AND n.y = 2 RETURN n"))
        assert len(parts) == 1
        assert len(parts[0].predicates) == 2

    # Shape A extended: three conjuncts → three BoundPredicates
    def test_triple_and_produces_three_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE n.x = 1 AND n.y = 2 AND n.z = 3 RETURN n")
        )
        assert len(parts) == 1
        assert len(parts[0].predicates) == 3

    # Shape B: A AND (B OR C) → 2 BoundPredicates (top-level AND splits)
    def test_and_with_paren_or_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE n.x = 1 AND (n.y = 2 OR n.z = 3) RETURN n")
        )
        assert len(parts) == 1
        exprs = [p.expression for p in parts[0].predicates]
        assert len(exprs) == 2
        assert any(e == "n.x = 1" for e in exprs)
        assert any("OR" in e.upper() for e in exprs)

    # Shape C: (A OR B) AND C → 2 BoundPredicates (OR-compound + bare C)
    def test_paren_or_and_c_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE (n.x = 1 OR n.y = 2) AND n.z = 3 RETURN n")
        )
        assert len(parts) == 1
        exprs = [p.expression for p in parts[0].predicates]
        assert len(exprs) == 2
        # One conjunct is the OR-compound, the other is the bare C.
        assert any("OR" in e.upper() for e in exprs)
        assert any("n.z = 3" in e for e in exprs)

    # Shape D: NOT A AND B → 2 BoundPredicates
    def test_not_and_b_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE NOT n.x = 1 AND n.y = 2 RETURN n")
        )
        assert len(parts) == 1
        exprs = [p.expression for p in parts[0].predicates]
        assert len(exprs) == 2
        assert any(e.upper().startswith("NOT") for e in exprs)
        assert any("n.y = 2" in e for e in exprs)

    # Shape E: A XOR B → single BoundPredicate containing XOR
    def test_xor_produces_one_bound_predicate_with_xor(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE n:Admin XOR n:Active RETURN n")
        )
        assert len(parts) == 1
        assert len(parts[0].predicates) == 1
        assert "XOR" in parts[0].predicates[0].expression.upper()

    # Shape F: label AND property → 2 BoundPredicates (structured route).
    # `expression` carries dataclass repr per binder's pre-existing
    # literal-atom fidelity caveat (tracked in #1200); substring-match.
    def test_label_and_property_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE n:Admin AND n.active = true RETURN n")
        )
        assert len(parts) == 1
        exprs = [p.expression for p in parts[0].predicates]
        assert len(exprs) == 2
        assert any("LabelRef" in e and "Admin" in e for e in exprs)
        assert any("PropertyRef" in e and "active" in e for e in exprs)

    # Shape G: quoted AND → two BoundPredicates (no extra split)
    def test_quoted_and_produces_two_bound_predicates(self) -> None:
        parts = _match_parts(
            _bind("MATCH (n) WHERE n.name = 'has AND text' AND n.x = 1 RETURN n")
        )
        assert len(parts) == 1
        assert len(parts[0].predicates) == 2


# ---------------------------------------------------------------------------
# Section 3 — Pushdown expression shapes
# ---------------------------------------------------------------------------
# Tests the split_top_level_and + is_null_rejecting behaviour for the
# expression forms each boolean shape produces.  IR-level push/keep
# decisions are already covered in test_predicate_pushdown_pass.py.

class TestPushdownExprShape:
    """Expression-level split and null-rejection per boolean shape."""

    # Shape A: plain AND splits into independent conjuncts
    def test_a_and_b_splits_into_two_conjuncts(self) -> None:
        assert split_top_level_and("n.x = 1 AND n.y = 2") == (
            "n.x = 1",
            "n.y = 2",
        )

    # Shape B: AND with parenthesized OR — outer AND splits; inner OR is opaque.
    # The splitter is a standalone utility; this shape raises GFQLSyntaxError in the
    # parser but is valid input here (splitter operates independently of the parser).
    def test_and_with_paren_or_splits_at_outer_and_only(self) -> None:
        parts = split_top_level_and("n.x = 1 AND (n.y = 2 OR n.z = 3)")
        assert len(parts) == 2
        assert parts[0] == "n.x = 1"
        assert "OR" in parts[1].upper()

    # Shape C: OR-first compound — outer AND splits; OR group is opaque
    def test_paren_or_and_c_splits_at_outer_and_only(self) -> None:
        parts = split_top_level_and("(n.x = 1 OR n.y = 2) AND n.z = 3")
        assert len(parts) == 2
        assert "OR" in parts[0].upper()
        assert parts[1] == "n.z = 3"

    # Shape D: NOT prefix — AND still splits correctly
    def test_not_a_and_b_splits_at_and(self) -> None:
        parts = split_top_level_and("NOT n.x = 1 AND n.y = 2")
        assert len(parts) == 2
        assert parts[0] == "NOT n.x = 1"
        assert parts[1] == "n.y = 2"

    # Shape E: XOR — not an AND so no split
    def test_xor_does_not_split(self) -> None:
        assert split_top_level_and("n:Admin XOR n:Active") == ("n:Admin XOR n:Active",)

    # Shape F: label predicate AND property — splits normally
    def test_label_and_property_splits(self) -> None:
        assert split_top_level_and("n:Admin AND n.active = true") == (
            "n:Admin",
            "n.active = true",
        )

    # Shape G: quoted AND is opaque — single-quoted
    def test_single_quoted_and_does_not_split(self) -> None:
        assert split_top_level_and("n.name = 'has AND text' AND n.x = 1") == (
            "n.name = 'has AND text'",
            "n.x = 1",
        )

    # Shape G: backtick-quoted identifier with AND does not split
    def test_backtick_and_does_not_split(self) -> None:
        assert split_top_level_and("n.`weird AND name` = 1 AND n.x = 2") == (
            "n.`weird AND name` = 1",
            "n.x = 2",
        )

    # Shape G: square-bracket AND is opaque
    def test_square_bracket_and_does_not_split(self) -> None:
        assert split_top_level_and("a[0 AND 1] AND b = 2") == (
            "a[0 AND 1]",
            "b = 2",
        )

    # Shape G: curly-brace AND is opaque
    def test_curly_brace_and_does_not_split(self) -> None:
        assert split_top_level_and("{k: v AND w} AND z = 1") == (
            "{k: v AND w}",
            "z = 1",
        )

    # Null-rejection: plain comparison on optional alias is null-rejecting
    def test_plain_comparison_is_null_rejecting(self) -> None:
        pred = _pred("n.x = 1", frozenset({"n"}))
        assert is_null_rejecting(pred, frozenset({"n"}))

    # Null-rejection: OR expression on optional alias — conservative: no null-safe form
    # present, so is_null_rejecting returns True (the function does not analyze OR chains).
    def test_or_on_optional_alias_is_rejecting(self) -> None:
        pred = _pred("n.y = 2 OR n.z = 3", frozenset({"n"}))
        assert is_null_rejecting(pred, frozenset({"n"}))

    # Null-rejection: IS NULL on optional alias is null-safe (not rejecting)
    def test_is_null_on_optional_alias_is_not_rejecting(self) -> None:
        pred = _pred("n.x IS NULL", frozenset({"n"}))
        assert not is_null_rejecting(pred, frozenset({"n"}))

    # Null-rejection: compound AND containing IS NULL is still rejecting
    # (conservative: True AND NULL = NULL, so the AND makes it rejecting)
    def test_and_with_is_null_is_still_rejecting(self) -> None:
        pred = _pred("n.x IS NULL AND n.y = 1", frozenset({"n"}))
        assert is_null_rejecting(pred, frozenset({"n"}))

    # Null-rejection: XOR on optional alias — treated as null-rejecting (unknown form)
    def test_xor_on_optional_alias_is_rejecting(self) -> None:
        pred = _pred("n:Admin XOR n:Active", frozenset({"n"}))
        assert is_null_rejecting(pred, frozenset({"n"}))
