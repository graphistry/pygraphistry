"""Tests for UnnestApply Tier 1 structural pass."""
from __future__ import annotations

from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Apply, AntiSemiApply, Filter, Join, NodeScan, PatternMatch, RowSchema, SemiApply
from graphistry.compute.gfql.ir.types import BoundPredicate, ScalarType
from graphistry.compute.gfql.passes import DEFAULT_LOGICAL_PASSES, DEFAULT_TIER2_PASSES, PassManager, UnnestApply


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

    def test_output_schema_is_preserved_after_rewrite(self):
        schema = RowSchema(columns={"n": ScalarType(kind="string")})
        plan = Apply(op_id=3, output_schema=schema, input=NodeScan(op_id=1), subquery=NodeScan(op_id=2), correlation_vars=frozenset())
        result = UnnestApply().run(plan, _ctx())
        assert result.plan.output_schema == schema

    def test_default_pass_manager_unnest_plus_pushdown_integration(self):
        # UnnestApply (T1) rewrites Apply → Join; then PredicatePushdownPass (T2)
        # pushes the filter predicate into the PatternMatch.
        schema = RowSchema(columns={"n": ScalarType(kind="string")})
        pattern = PatternMatch(op_id=1, output_schema=schema, predicates=[])
        pred = BoundPredicate(expression="n.name = 'x'", references=frozenset({"n"}))
        filt = Filter(op_id=2, output_schema=schema, input=pattern, predicate=pred)
        apply_node = Apply(op_id=3, input=filt, subquery=NodeScan(op_id=4), correlation_vars=frozenset())

        result = PassManager(DEFAULT_LOGICAL_PASSES, DEFAULT_TIER2_PASSES).run(apply_node, _ctx())

        # T1 UnnestApply: Apply → Join
        assert isinstance(result.plan, Join)
        # T2 PredicatePushdown: Filter around PatternMatch is eliminated — predicate moved in
        join = result.plan
        assert isinstance(join.left, PatternMatch)  # type: ignore[union-attr]
        assert len(join.left.predicates) == 1  # type: ignore[union-attr]
