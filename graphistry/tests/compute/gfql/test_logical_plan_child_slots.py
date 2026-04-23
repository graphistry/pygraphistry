"""Tests for CHILD_SLOTS + iter_children invariants in logical_plan.py.

These lock the shared child-slot enumeration used by the IR verifier,
physical planner, and rewrite passes (UnnestApply, PredicatePushdownPass).
"""
from __future__ import annotations

from typing import get_type_hints

from graphistry.compute.gfql.ir.logical_plan import (
    Apply,
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


def test_unnest_apply_preserves_sibling_identity_when_one_branch_rewrites() -> None:
    # Locks the asymmetric case: in a Join whose left branch contains a
    # rewritable Apply and whose right branch does not, the rewrite must
    # produce a new Join (because left changed) but the right child must
    # come through by identity — not a gratuitous ``dataclasses.replace``
    # on the untouched subtree.
    right_leaf = NodeScan(op_id=1, label="B", output_schema=_schema("b"))
    apply_left = Apply(
        op_id=2,
        input=NodeScan(op_id=3, label="A", output_schema=_schema("a")),
        subquery=NodeScan(op_id=4, label="S", output_schema=_schema("s")),
        correlation_vars=frozenset(),
        output_schema=_schema("a", "s"),
    )
    join = Join(
        op_id=5,
        left=apply_left,
        right=right_leaf,
        condition=None,
        join_type="cross",
        output_schema=_schema("a", "s", "b"),
    )

    result = UnnestApply().run(join, PlanContext()).plan
    assert isinstance(result, Join)
    assert result is not join  # left changed, so a new Join was built
    assert result.right is right_leaf  # right subtree preserved by identity


def test_child_slots_covers_every_logical_plan_field_of_optional_logical_plan_type() -> None:
    # Structural enforcement: any LogicalPlan subclass field that has
    # type ``Optional[LogicalPlan]`` (directly or via Union with None)
    # must be listed in CHILD_SLOTS.  Guards future subclasses adding
    # a new child slot without updating the tuple.
    for subclass in LogicalPlan.__subclasses__() + [
        cls
        for parent in LogicalPlan.__subclasses__()
        for cls in parent.__subclasses__()
    ]:
        try:
            hints = get_type_hints(subclass)
        except Exception:  # pragma: no cover - forward refs etc.
            continue
        for fname, ftype in hints.items():
            origin = getattr(ftype, "__origin__", None)
            args = getattr(ftype, "__args__", ())
            if origin is None:
                continue
            # Union[LogicalPlan, None] or Optional[LogicalPlan]
            if any(
                isinstance(arg, type) and issubclass(arg, LogicalPlan)
                for arg in args
            ):
                assert fname in CHILD_SLOTS, (
                    f"{subclass.__name__}.{fname} has LogicalPlan-typed field "
                    f"but is not in CHILD_SLOTS — update CHILD_SLOTS in logical_plan.py"
                )


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
