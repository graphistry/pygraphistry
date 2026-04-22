"""Tests for predicate-pushdown safety primitives (M4-PR2 / issue #1181).

Covers:
  - is_null_rejecting: null-rejecting classification on optional-arm aliases
  - is_null_safe: null-safe classification (positive cases + IS NULL / IS NOT NULL / COALESCE)
  - with_barrier_blocks_pushdown: WITH boundary blocks backward predicate movement
"""
from __future__ import annotations

from graphistry.compute.gfql.ir.bound_ir import ScopeFrame
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import BoundPredicate
from graphistry.compute.gfql.ir.pushdown_safety import (
    is_null_rejecting,
    is_null_safe,
    with_barrier_blocks_pushdown,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pred(expr: str, refs: frozenset) -> BoundPredicate:
    return BoundPredicate(expression=expr, references=refs)


def _frame(origin: str, visible: frozenset) -> ScopeFrame:
    return ScopeFrame(visible_vars=visible, schema=RowSchema(), origin_clause=origin)


# ---------------------------------------------------------------------------
# is_null_rejecting
# ---------------------------------------------------------------------------

class TestIsNullRejecting:
    def test_comparison_on_optional_alias_is_rejecting(self):
        assert is_null_rejecting(_pred("n.age > 5", frozenset({"n"})), frozenset({"n"}))

    def test_equality_on_optional_alias_is_rejecting(self):
        assert is_null_rejecting(_pred("n.name = 'Alice'", frozenset({"n"})), frozenset({"n"}))

    def test_contains_on_optional_alias_is_rejecting(self):
        assert is_null_rejecting(_pred("n.tags CONTAINS 'x'", frozenset({"n"})), frozenset({"n"}))

    def test_is_null_expr_is_not_rejecting(self):
        assert not is_null_rejecting(_pred("n IS NULL", frozenset({"n"})), frozenset({"n"}))

    def test_is_not_null_expr_is_not_rejecting(self):
        assert not is_null_rejecting(_pred("n IS NOT NULL", frozenset({"n"})), frozenset({"n"}))

    def test_coalesce_expr_is_not_rejecting(self):
        assert not is_null_rejecting(_pred("coalesce(n.age, 0) > 5", frozenset({"n"})), frozenset({"n"}))

    def test_nullif_expr_is_not_rejecting(self):
        assert not is_null_rejecting(_pred("nullif(n.val, '') IS NOT NULL", frozenset({"n"})), frozenset({"n"}))

    def test_predicate_not_referencing_optional_alias_is_not_rejecting(self):
        # predicate touches "m"; "n" is the null-extended alias — no overlap
        assert not is_null_rejecting(_pred("m.age > 5", frozenset({"m"})), frozenset({"n"}))

    def test_empty_optional_aliases_is_not_rejecting(self):
        assert not is_null_rejecting(_pred("n.age > 5", frozenset({"n"})), frozenset())

    def test_empty_expression_is_rejecting_conservative(self):
        # Empty expression with a reference to optional alias → conservative
        assert is_null_rejecting(_pred("", frozenset({"n"})), frozenset({"n"}))

    def test_case_insensitive_is_null_form(self):
        assert not is_null_rejecting(_pred("N IS NULL", frozenset({"n"})), frozenset({"n"}))

    def test_multiple_refs_one_optional_is_rejecting(self):
        # n is optional, m is not; predicate touches both
        assert is_null_rejecting(_pred("n.age = m.age", frozenset({"n", "m"})), frozenset({"n"}))


# ---------------------------------------------------------------------------
# is_null_safe
# ---------------------------------------------------------------------------

class TestIsNullSafe:
    def test_null_safe_is_inverse_for_rejecting(self):
        pred = _pred("n.age > 5", frozenset({"n"}))
        assert not is_null_safe(pred, frozenset({"n"}))

    def test_null_safe_is_inverse_for_safe(self):
        pred = _pred("n IS NULL", frozenset({"n"}))
        assert is_null_safe(pred, frozenset({"n"}))

    def test_non_optional_alias_is_safe(self):
        pred = _pred("m.age > 5", frozenset({"m"}))
        assert is_null_safe(pred, frozenset({"n"}))


# ---------------------------------------------------------------------------
# with_barrier_blocks_pushdown
# ---------------------------------------------------------------------------

class TestWithBarrier:
    def test_with_barrier_blocks_ref_not_visible_before_boundary(self):
        stack = [_frame("WITH", frozenset({"n"}))]
        # "m" is not visible before the WITH — blocked
        assert with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_with_barrier_allows_ref_visible_before_boundary(self):
        stack = [_frame("WITH", frozenset({"n", "m"}))]
        # "n" IS visible before the WITH — not blocked
        assert not with_barrier_blocks_pushdown(stack, frozenset({"n"}))

    def test_match_frame_never_blocks(self):
        stack = [_frame("MATCH", frozenset({"n"}))]
        assert not with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_unwind_frame_never_blocks(self):
        stack = [_frame("UNWIND", frozenset({"n"}))]
        assert not with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_empty_stack_never_blocks(self):
        assert not with_barrier_blocks_pushdown([], frozenset({"n"}))

    def test_empty_refs_never_blocked(self):
        stack = [_frame("WITH", frozenset({"n"}))]
        assert not with_barrier_blocks_pushdown(stack, frozenset())

    def test_origin_clause_case_insensitive(self):
        stack = [_frame("with", frozenset({"n"}))]
        assert with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_mixed_stack_with_barrier_wins(self):
        stack = [
            _frame("MATCH", frozenset({"n", "m"})),
            _frame("WITH", frozenset({"n"})),    # collapses visible to just n
            _frame("MATCH", frozenset({"n", "r"})),
        ]
        # "m" not visible in the WITH frame — blocked
        assert with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_multiple_with_frames_all_must_allow(self):
        stack = [
            _frame("WITH", frozenset({"n", "m"})),   # both visible
            _frame("WITH", frozenset({"n"})),          # drops m
        ]
        # Blocked by the second WITH frame
        assert with_barrier_blocks_pushdown(stack, frozenset({"m"}))

    def test_refs_visible_in_all_with_frames_not_blocked(self):
        stack = [
            _frame("WITH", frozenset({"n", "m"})),
            _frame("WITH", frozenset({"n", "m"})),
        ]
        assert not with_barrier_blocks_pushdown(stack, frozenset({"n"}))
