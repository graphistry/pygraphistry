"""Tests for CHILD_SLOTS + iter_children invariants in logical_plan.py.

These lock the shared child-slot enumeration used by the IR verifier,
physical planner, and rewrite passes (UnnestApply, PredicatePushdownPass).
"""
from __future__ import annotations

from graphistry.compute.gfql.ir.logical_plan import (
    CHILD_SLOTS,
    Filter,
    Join,
    LogicalPlan,
    NodeScan,
    PatternMatch,
    RowSchema,
    iter_children,
)
from graphistry.compute.gfql.ir.types import BoundPredicate, NodeRef
from graphistry.compute.gfql.passes import PredicatePushdownPass, UnnestApply
from graphistry.compute.gfql.ir.compilation import PlanContext


def _schema(*aliases: str) -> RowSchema:
    return RowSchema(columns={alias: NodeRef() for alias in aliases})


def test_child_slots_tuple_covers_all_LogicalPlan_child_fields() -> None:
    # Sanity: the tuple must include the four structural slot names in a
    # deterministic order.  Adding a new slot requires updating this tuple
    # first, which is the whole point of centralizing it.
    assert CHILD_SLOTS == ("input", "left", "right", "subquery")


def test_iter_children_skips_none_slots() -> None:
    leaf = NodeScan(op_id=1, label="Person", output_schema=_schema("n"))
    # NodeScan has no child slots populated — iter_children yields nothing.
    assert list(iter_children(leaf)) == []


def test_iter_children_yields_populated_input_slot() -> None:
    leaf = NodeScan(op_id=1, label="Person", output_schema=_schema("n"))
    parent = Filter(
        op_id=2,
        input=leaf,
        predicate=BoundPredicate(expression="n.age > 0", references=frozenset({"n"})),
        output_schema=_schema("n"),
    )
    children = list(iter_children(parent))
    assert len(children) == 1
    slot, child = children[0]
    assert slot == "input"
    assert child is leaf


def test_iter_children_yields_left_and_right_for_join() -> None:
    left = NodeScan(op_id=1, label="Person", output_schema=_schema("a"))
    right = NodeScan(op_id=2, label="Company", output_schema=_schema("b"))
    join = Join(
        op_id=3,
        left=left,
        right=right,
        condition=None,
        join_type="cross",
        output_schema=_schema("a", "b"),
    )
    slots = {slot: child for slot, child in iter_children(join)}
    assert slots == {"left": left, "right": right}


def test_unnest_apply_preserves_parent_identity_when_no_descendant_rewrites() -> None:
    # A tree where no Apply exists: UnnestApply should return the root
    # unchanged (same Python object), relying on the ``rewritten_child is
    # not child`` identity guard at every level.
    leaf = NodeScan(op_id=1, label="Person", output_schema=_schema("n"))
    mid = PatternMatch(
        op_id=2,
        input=leaf,
        pattern={"aliases": ("n",)},
        output_schema=_schema("n"),
    )
    root = Filter(
        op_id=3,
        input=mid,
        predicate=BoundPredicate(expression="n.age > 0", references=frozenset({"n"})),
        output_schema=_schema("n"),
    )

    result = UnnestApply().run(root, PlanContext()).plan
    assert result is root, "UnnestApply must return the identical root when no descendant is rewritten"


def test_predicate_pushdown_preserves_parent_identity_when_no_pushdown() -> None:
    # Same contract for PredicatePushdownPass: if no pushdown happens anywhere
    # in the tree, the root must come back by identity.
    leaf = NodeScan(op_id=1, label="Person", output_schema=_schema("n"))
    root = Filter(
        op_id=2,
        input=leaf,
        predicate=BoundPredicate(expression="n.age > 0", references=frozenset({"n"})),
        output_schema=_schema("n"),
    )
    # Filter-above-NodeScan is not a pushdown candidate (only Filter-above-
    # PatternMatch is), so the pass leaves the tree untouched.
    result = PredicatePushdownPass().run(root, PlanContext()).plan
    assert result is root
    assert isinstance(result, LogicalPlan)
