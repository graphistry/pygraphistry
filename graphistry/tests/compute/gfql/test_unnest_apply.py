"""Tests for UnnestApply Tier 1 structural pass."""
from __future__ import annotations

from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Apply, AntiSemiApply, Join, NodeScan, SemiApply
from graphistry.compute.gfql.passes import UnnestApply


def _ctx() -> PlanContext:
    return PlanContext()


class TestUnnestApply:
    def test_non_correlated_apply_rewrites_to_cross_join(self):
        left = NodeScan(op_id=1)
        right = NodeScan(op_id=2)
        plan = Apply(op_id=3, input=left, subquery=right, correlation_vars=frozenset())

        result = UnnestApply().run(plan, _ctx())

        assert isinstance(result.plan, Join)
        assert result.plan.join_type == "cross"  # type: ignore[union-attr]
        assert result.plan.left is left  # type: ignore[union-attr]
        assert result.plan.right is right  # type: ignore[union-attr]
        assert result.plan.op_id == 3
        assert result.changed is True

    def test_correlated_apply_is_left_untouched(self):
        left = NodeScan(op_id=1)
        right = NodeScan(op_id=2)
        plan = Apply(op_id=3, input=left, subquery=right, correlation_vars=frozenset({"n"}))

        result = UnnestApply().run(plan, _ctx())

        assert isinstance(result.plan, Apply)
        assert result.plan is plan
        assert result.changed is False

    def test_no_apply_nodes_is_identity(self):
        plan = NodeScan(op_id=1)
        result = UnnestApply().run(plan, _ctx())
        assert result.plan is plan
        assert result.changed is False

    def test_nested_non_correlated_apply_is_rewritten(self):
        inner_left = NodeScan(op_id=1)
        inner_right = NodeScan(op_id=2)
        inner_apply = Apply(op_id=3, input=inner_left, subquery=inner_right, correlation_vars=frozenset())
        outer = Apply(op_id=4, input=inner_apply, subquery=NodeScan(op_id=5), correlation_vars=frozenset())

        result = UnnestApply().run(outer, _ctx())

        # Outer Apply → Join, inner Apply → Join
        assert isinstance(result.plan, Join)
        assert isinstance(result.plan.left, Join)  # type: ignore[union-attr]
        assert result.metadata["unnested"] == 2
        assert result.changed is True

    def test_semi_apply_non_correlated_is_not_rewritten(self):
        # UnnestApply only targets Apply, not SemiApply or AntiSemiApply
        plan = SemiApply(op_id=1, input=NodeScan(op_id=2), subquery=NodeScan(op_id=3), correlation_vars=frozenset())
        result = UnnestApply().run(plan, _ctx())
        assert isinstance(result.plan, SemiApply)
        assert result.changed is False

    def test_anti_semi_apply_non_correlated_is_not_rewritten(self):
        plan = AntiSemiApply(op_id=1, input=NodeScan(op_id=2), subquery=NodeScan(op_id=3), correlation_vars=frozenset())
        result = UnnestApply().run(plan, _ctx())
        assert isinstance(result.plan, AntiSemiApply)
        assert result.changed is False

    def test_metadata_reports_unnested_count(self):
        plan = Apply(op_id=1, input=NodeScan(op_id=2), subquery=NodeScan(op_id=3), correlation_vars=frozenset())
        result = UnnestApply().run(plan, _ctx())
        assert result.metadata["unnested"] == 1

    def test_unchanged_plan_reports_zero_unnested(self):
        plan = NodeScan(op_id=1)
        result = UnnestApply().run(plan, _ctx())
        assert result.metadata["unnested"] == 0
