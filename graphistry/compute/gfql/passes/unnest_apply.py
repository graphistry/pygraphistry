"""UnnestApply: Tier 1 structural pass eliminating non-correlated Apply operators.

A non-correlated Apply (``correlation_vars == frozenset()``) is semantically
equivalent to a cross join because the subquery does not reference any variable
from the outer input.  Rewriting to ``Join(join_type="cross")`` exposes the
shape to downstream join-ordering and predicate-pushdown passes.

Correlated Apply operators are left untouched.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Tuple, cast

from graphistry.compute.gfql.ir.logical_plan import Apply, CHILD_SLOTS, Join, LogicalPlan
from graphistry.compute.gfql.passes.manager import LogicalPass, PassResult


class UnnestApply:
    """Rewrite non-correlated Apply nodes to cross Join nodes."""

    name = "unnest_apply"

    def run(self, plan: LogicalPlan, ctx: Any) -> PassResult:  # noqa: ANN401
        _ = ctx
        rewritten, count = _unnest_tree(plan)
        return PassResult(
            plan=rewritten,
            metadata={"unnested": count},
            changed=count > 0,
        )


def _unnest_tree(plan: LogicalPlan) -> Tuple[LogicalPlan, int]:
    count = 0
    children_updates = {}
    for slot in CHILD_SLOTS:
        child = getattr(plan, slot, None)
        if isinstance(child, LogicalPlan):
            rewritten_child, child_count = _unnest_tree(child)
            count += child_count
            if rewritten_child is not child:
                children_updates[slot] = rewritten_child

    current: LogicalPlan = (
        cast(LogicalPlan, replace(cast(Any, plan), **children_updates))
        if children_updates
        else plan
    )

    if isinstance(current, Apply) and not current.correlation_vars:
        joined = Join(
            op_id=current.op_id,
            output_schema=current.output_schema,
            left=current.input,
            right=current.subquery,
            condition=None,
            join_type="cross",
        )
        return joined, count + 1

    return current, count
