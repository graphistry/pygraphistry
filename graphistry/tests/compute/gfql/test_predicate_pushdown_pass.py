from __future__ import annotations

from graphistry.compute.gfql.ir.bound_ir import ScopeFrame
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, NodeScan, PatternMatch, Project, RowSchema
from graphistry.compute.gfql.ir.types import BoundPredicate, NodeRef
from graphistry.compute.gfql.passes import PredicatePushdownPass


def _schema(*aliases: str) -> RowSchema:
    return RowSchema(columns={alias: NodeRef() for alias in aliases})


def _pred(expr: str, refs: frozenset[str]) -> BoundPredicate:
    return BoundPredicate(expression=expr, references=refs)


def test_predicate_pushdown_full_push_rewrites_filter_into_pattern() -> None:
    plan = Filter(
        op_id=3,
        input=PatternMatch(op_id=2, pattern={"aliases": ("n",)}, output_schema=_schema("n")),
        predicate=_pred("n.age > 5", frozenset({"n"})),
        output_schema=_schema("n"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, PatternMatch)
    assert [pred.expression for pred in result.predicates] == ["n.age > 5"]


def test_predicate_pushdown_partial_push_keeps_residual_for_optional_arm() -> None:
    input_scan = NodeScan(op_id=1, label="Person", output_schema=_schema("m"))
    optional_match = PatternMatch(
        op_id=2,
        input=input_scan,
        pattern={"aliases": ("n",)},
        optional=True,
        arm_id="arm1",
        output_schema=_schema("m", "n"),
    )
    plan = Filter(
        op_id=3,
        input=optional_match,
        predicate=_pred("n IS NULL AND n.age > 5", frozenset({"n"})),
        output_schema=_schema("m", "n"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, Filter)
    assert result.predicate.expression == "n.age > 5"
    assert isinstance(result.input, PatternMatch)
    assert [pred.expression for pred in result.input.predicates] == ["n IS NULL"]


def test_predicate_pushdown_no_push_for_optional_null_rejecting_predicate() -> None:
    input_scan = NodeScan(op_id=1, label="Person", output_schema=_schema("m"))
    optional_match = PatternMatch(
        op_id=2,
        input=input_scan,
        pattern={"aliases": ("n",)},
        optional=True,
        arm_id="arm1",
        output_schema=_schema("m", "n"),
    )
    plan = Filter(
        op_id=3,
        input=optional_match,
        predicate=_pred("n.age > 5", frozenset({"n"})),
        output_schema=_schema("m", "n"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, Filter)
    assert isinstance(result.input, PatternMatch)
    assert result.input.predicates == []


def test_predicate_pushdown_no_push_for_optional_null_rejecting_bare_alias_predicate() -> None:
    input_scan = NodeScan(op_id=1, label="Person", output_schema=_schema("m"))
    optional_match = PatternMatch(
        op_id=2,
        input=input_scan,
        pattern={"aliases": ("score",)},
        optional=True,
        arm_id="arm1",
        output_schema=_schema("m", "score"),
    )
    plan = Filter(
        op_id=3,
        input=optional_match,
        predicate=_pred("score > 5", frozenset()),
        output_schema=_schema("m", "score"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, Filter)
    assert isinstance(result.input, PatternMatch)
    assert result.input.predicates == []


def test_predicate_pushdown_partial_push_with_bare_alias_and_empty_refs() -> None:
    input_scan = NodeScan(op_id=1, label="Person", output_schema=_schema("m"))
    optional_match = PatternMatch(
        op_id=2,
        input=input_scan,
        pattern={"aliases": ("score",)},
        optional=True,
        arm_id="arm1",
        output_schema=_schema("m", "score"),
    )
    plan = Filter(
        op_id=3,
        input=optional_match,
        predicate=_pred("score IS NULL AND score > 5", frozenset()),
        output_schema=_schema("m", "score"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, Filter)
    assert result.predicate.expression == "score > 5"
    assert isinstance(result.input, PatternMatch)
    assert [pred.expression for pred in result.input.predicates] == ["score IS NULL"]


def test_predicate_pushdown_does_not_cross_with_like_projection_barrier() -> None:
    pattern = PatternMatch(
        op_id=2,
        pattern={"aliases": ("n",)},
        output_schema=_schema("n"),
    )
    with_like_project = Project(
        op_id=3,
        input=pattern,
        expressions=["n"],
        output_schema=_schema("n"),
    )
    plan = Filter(
        op_id=4,
        input=with_like_project,
        predicate=_pred("n.age > 5", frozenset({"n"})),
        output_schema=_schema("n"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    assert isinstance(result, Filter)
    assert isinstance(result.input, Project)
    assert isinstance(result.input.input, PatternMatch)
    assert result.input.input.predicates == []


def test_predicate_pushdown_blocks_when_scope_metadata_reports_with_barrier() -> None:
    plan = Filter(
        op_id=3,
        input=PatternMatch(op_id=2, pattern={"aliases": ("n",)}, output_schema=_schema("n")),
        predicate=_pred("n.age > 5", frozenset({"n"})),
        output_schema=_schema("n"),
    )
    ctx = PlanContext(
        scope_stack=(
            ScopeFrame(visible_vars=frozenset({"n"}), schema=_schema("n"), origin_clause="MATCH"),
            ScopeFrame(visible_vars=frozenset({"m"}), schema=_schema("m"), origin_clause="WITH"),
        ),
    )

    result = PredicatePushdownPass().run(plan, ctx).plan

    assert isinstance(result, Filter)
    assert isinstance(result.input, PatternMatch)
    assert result.input.predicates == []


def test_predicate_pushdown_allows_when_scope_metadata_preserves_alias_across_with() -> None:
    plan = Filter(
        op_id=3,
        input=PatternMatch(op_id=2, pattern={"aliases": ("n",)}, output_schema=_schema("n")),
        predicate=_pred("n.age > 5", frozenset({"n"})),
        output_schema=_schema("n"),
    )
    ctx = PlanContext(
        scope_stack=(
            ScopeFrame(visible_vars=frozenset({"n"}), schema=_schema("n"), origin_clause="MATCH"),
            ScopeFrame(visible_vars=frozenset({"n"}), schema=_schema("n"), origin_clause="WITH"),
        ),
    )

    result = PredicatePushdownPass().run(plan, ctx).plan

    assert isinstance(result, PatternMatch)
    assert [pred.expression for pred in result.predicates] == ["n.age > 5"]


def test_predicate_pushdown_splits_multi_alias_conjunct_and_narrows_references() -> None:
    # Locks the fix for issue #1195: _refs_for_segment previously used
    # ``rf"\\b..."`` (literal backslash-b) inside an rf-string, so per-segment
    # alias detection never matched and fell back to ``original_refs`` —
    # silently widening the reference set.  With the fix, each split conjunct
    # carries only the aliases it actually mentions.
    input_scan = NodeScan(op_id=1, label="Person", output_schema=_schema("m"))
    optional_match = PatternMatch(
        op_id=2,
        input=input_scan,
        pattern={"aliases": ("n",)},
        optional=True,
        arm_id="arm1",
        output_schema=_schema("m", "n"),
    )
    # "m.x > 1 AND n.y > 2" — m is an input alias (pushable past the optional
    # arm); n is an optional-arm alias (null-rejecting, must not be pushed).
    # If ``_refs_for_segment`` falsely returns original_refs={m,n} for the
    # m-only conjunct, is_null_rejecting treats it as referencing n and the
    # whole conjunct is held back — we verify m.x > 1 is independently pushed.
    plan = Filter(
        op_id=3,
        input=optional_match,
        predicate=_pred("m.x > 1 AND n.y > 2", frozenset({"m", "n"})),
        output_schema=_schema("m", "n"),
    )

    result = PredicatePushdownPass().run(plan, PlanContext()).plan

    # Filter-above-PatternMatch shape: partial push leaves a residual Filter.
    assert isinstance(result, Filter)
    pushed = result.input
    assert isinstance(pushed, PatternMatch)
    pushed_exprs = [pred.expression for pred in pushed.predicates]
    residual_exprs = [result.predicate.expression]
    # m-only conjunct is safely pushed; n-referencing conjunct stays as residual.
    assert "m.x > 1" in pushed_exprs
    assert any("n.y > 2" in expr for expr in residual_exprs)
