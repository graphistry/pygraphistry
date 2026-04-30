"""Tests for predicate-pushdown safety primitives (issue #1181).

Covers:
  - is_null_rejecting: null-rejecting classification on optional-arm aliases
  - is_null_safe: null-safe classification (positive cases + IS NULL / IS NOT NULL / COALESCE)
  - compound AND: always null-rejecting regardless of null-safe subterms
  - with_barrier_blocks_pushdown: WITH boundary blocks backward predicate movement
  - ir/__init__.py re-export smoke test
"""
from __future__ import annotations

# Import via ir package (exercises the __init__.py re-export path).
from graphistry.compute.gfql.ir import (
    is_null_rejecting,
    is_null_safe,
    with_barrier_blocks_pushdown,
)
from graphistry.compute.gfql.ir.bound_ir import ScopeFrame
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import BoundPredicate


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

    # --- compound AND --- #

    def test_compound_and_is_null_plus_comparison_is_rejecting(self):
        # n.name IS NULL is null-safe in isolation, but AND n.type = 'Person'
        # makes the whole expression null-rejecting: True AND NULL = NULL.
        assert is_null_rejecting(
            _pred("n.name IS NULL AND n.type = 'Person'", frozenset({"n"})),
            frozenset({"n"}),
        )

    def test_compound_and_comparison_plus_is_null_is_rejecting(self):
        # Same as above with operand order reversed.
        assert is_null_rejecting(
            _pred("n.type = 'Person' AND n.name IS NULL", frozenset({"n"})),
            frozenset({"n"}),
        )

    def test_compound_and_both_null_safe_is_still_rejecting(self):
        # Conservative: even "n IS NULL AND n IS NOT NULL" is classified
        # null-rejecting. The expression is degenerate (always false), but
        # parsing compound semantics is out of scope for this heuristic.
        assert is_null_rejecting(
            _pred("n IS NULL AND n IS NOT NULL", frozenset({"n"})),
            frozenset({"n"}),
        )

    def test_compound_and_no_optional_alias_ref_still_not_rejecting(self):
        # Even with AND, if no optional alias is referenced the early-exit
        # short-circuits before the AND guard.
        assert not is_null_rejecting(
            _pred("m.a = 1 AND m.b = 2", frozenset({"m"})),
            frozenset({"n"}),
        )

    # --- OR expressions --- #

    def test_or_with_is_null_form_is_conservatively_rejecting(self):
        # Conservative policy: OR compounds are always treated as
        # null-rejecting when they reference optional aliases.
        assert is_null_rejecting(
            _pred("n IS NULL OR n.age > 5", frozenset({"n"})),
            frozenset({"n"}),
        )

    def test_or_without_null_safe_form_is_rejecting(self):
        # n.age > 5 OR n.name = 'x': if n=NULL → NULL OR NULL = NULL → row filtered.
        assert is_null_rejecting(
            _pred("n.age > 5 OR n.name = 'x'", frozenset({"n"})),
            frozenset({"n"}),
        )

    # --- property-level IS NULL --- #

    def test_property_is_null_is_not_rejecting(self):
        # n.name IS NULL is a common real-world pattern;
        # if n=NULL → NULL.name = NULL, NULL IS NULL = True → null-safe.
        assert not is_null_rejecting(
            _pred("n.name IS NULL", frozenset({"n"})),
            frozenset({"n"}),
        )

    # --- empty references --- #

    def test_empty_references_is_not_rejecting(self):
        # Predicate with no alias references never touches any optional alias.
        assert not is_null_rejecting(
            _pred("1 = 1", frozenset()),
            frozenset({"n"}),
        )

    # --- multiple optional aliases --- #

    def test_multiple_optional_aliases_one_referenced_is_rejecting(self):
        # n and m are both optional; predicate only references n.
        assert is_null_rejecting(
            _pred("n.age > 5", frozenset({"n"})),
            frozenset({"n", "m"}),
        )

    def test_multiple_optional_aliases_neither_referenced_is_not_rejecting(self):
        # r is optional but predicate only references n, which is not optional.
        assert not is_null_rejecting(
            _pred("n.age > 5", frozenset({"n"})),
            frozenset({"r"}),
        )

    # --- string-literal AND heuristic boundary --- #

    def test_coalesce_with_and_in_literal_is_conservatively_rejecting(self):
        # Known limitation: " and " in a string literal triggers the AND guard.
        # coalesce(n.name, 'Alice and Bob') IS NOT NULL is semantically null-safe
        # (COALESCE always returns non-NULL here), but the substring heuristic
        # cannot distinguish operator AND from the literal " and ".
        # Over-conservative: prevents valid pushdown but never allows incorrect pushdown.
        assert is_null_rejecting(
            _pred("coalesce(n.name, 'Alice and Bob') IS NOT NULL", frozenset({"n"})),
            frozenset({"n"}),
        )


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

    def test_with_empty_visible_vars_blocks_any_ref(self):
        # A WITH that projects nothing blocks all predicate references.
        stack = [_frame("WITH", frozenset())]
        assert with_barrier_blocks_pushdown(stack, frozenset({"n"}))

    def test_refs_exactly_equal_to_visible_vars_not_blocked(self):
        # Predicate refs == visible_vars → all refs survive the WITH → not blocked.
        stack = [_frame("WITH", frozenset({"n", "m"}))]
        assert not with_barrier_blocks_pushdown(stack, frozenset({"n", "m"}))

    def test_refs_superset_of_visible_vars_blocked(self):
        # Predicate refs {n, m, r}; WITH only forwards {n, m} → r missing → blocked.
        stack = [_frame("WITH", frozenset({"n", "m"}))]
        assert with_barrier_blocks_pushdown(stack, frozenset({"n", "m", "r"}))

    def test_all_match_frames_no_block(self):
        # Three MATCH frames — none is WITH — never blocks.
        stack = [
            _frame("MATCH", frozenset({"n"})),
            _frame("MATCH", frozenset({"n", "m"})),
            _frame("MATCH", frozenset({"n", "m", "r"})),
        ]
        assert not with_barrier_blocks_pushdown(stack, frozenset({"r"}))

    def test_with_between_unwind_frames_blocks(self):
        # WITH sandwiched between UNWIND frames; still blocks if ref missing.
        stack = [
            _frame("UNWIND", frozenset({"n", "m"})),
            _frame("WITH", frozenset({"n"})),
            _frame("UNWIND", frozenset({"n", "r"})),
        ]
        assert with_barrier_blocks_pushdown(stack, frozenset({"m"}))
