"""Tests for CHILD_SLOTS + iter_children invariants in logical_plan.py.

These lock the shared child-slot enumeration used by the IR verifier,
physical planner, and rewrite passes (UnnestApply, PredicatePushdownPass).
"""
from __future__ import annotations

from typing import Union, get_type_hints

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


def _all_logical_plan_subclasses() -> list[type]:
    """Return every concrete subclass of ``LogicalPlan`` reachable transitively."""
    seen: list[type] = []
    stack = list(LogicalPlan.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.append(cls)
        stack.extend(cls.__subclasses__())
    return seen


def test_child_slots_covers_every_logical_plan_field_of_optional_logical_plan_type() -> None:
    # Structural enforcement: any ``LogicalPlan`` subclass field whose
    # resolved type is ``Optional[LogicalPlan]`` (Union with None) must
    # appear in ``CHILD_SLOTS``.  Guards future subclasses adding a
    # new child slot without updating the tuple.
    #
    # Also asserts every subclass was actually inspected — otherwise a
    # ``get_type_hints`` failure (e.g. Python 3.8 hitting PEP 585 bare
    # ``frozenset[str]`` subscripts) would silently skip exactly the
    # classes (``Apply``, ``SemiApply``, ``AntiSemiApply``) that motivate
    # this test, turning it into a vacuous pass.
    subclasses = _all_logical_plan_subclasses()
    assert len(subclasses) >= 10, (
        f"expected >=10 LogicalPlan subclasses, found {len(subclasses)}"
    )

    # Only swallow resolution failures for a specific, expected case
    # (PEP 585 subscripts on older Pythons).  Any other failure is a
    # real defect that should surface.
    checked = 0
    skipped_for_type_subscript: list[str] = []
    for subclass in subclasses:
        try:
            hints = get_type_hints(subclass)
        except TypeError as exc:
            if "subscript" in str(exc) or "not subscriptable" in str(exc):
                skipped_for_type_subscript.append(subclass.__name__)
                continue
            raise
        checked += 1
        for fname, ftype in hints.items():
            # Only treat ``Optional[LogicalPlan]`` / ``Union[LogicalPlan, None]``
            # as a structural child slot.  ``List[LogicalPlan]`` and similar
            # container-typed fields are out of scope for the CHILD_SLOTS
            # single-slot model and should not trigger this assertion.
            if getattr(ftype, "__origin__", None) is not Union:
                continue
            args = getattr(ftype, "__args__", ())
            if any(
                isinstance(arg, type) and issubclass(arg, LogicalPlan)
                for arg in args
            ):
                assert fname in CHILD_SLOTS, (
                    f"{subclass.__name__}.{fname} has Optional[LogicalPlan] "
                    f"type but is not in CHILD_SLOTS — update CHILD_SLOTS "
                    f"in logical_plan.py"
                )

    # If PEP 585 subscripts caused skips, require that at least the three
    # child-carrying Apply-family classes were inspected somewhere in the
    # run — they are the reason this test exists.  On Python 3.9+ this is
    # a no-op; on 3.8 it fails loudly rather than silently when the target
    # classes get skipped.
    if skipped_for_type_subscript:
        inspected = {c.__name__ for c in subclasses} - set(skipped_for_type_subscript)
        for required in ("Apply", "SemiApply", "AntiSemiApply"):
            assert required in inspected, (
                f"{required} was skipped due to PEP 585 subscript error; "
                f"reflective CHILD_SLOTS coverage is incomplete on this runtime. "
                f"Skipped classes: {skipped_for_type_subscript}"
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
