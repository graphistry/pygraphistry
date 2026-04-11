"""Tests for LogicalPlan structural verifier (issue #1127 / M2-PR3).

Five invariants:
  1. op_id uniqueness — all operators in the tree have distinct op_ids (0 is exempt: unassigned)
  2. Dangling refs — input/left/right/subquery children must be LogicalPlan instances
  3. Predicate scope — BoundPredicate.expression must be non-empty on all predicate-bearing ops
  4. Output schema consistency — RowSchema columns values must be LogicalType instances
  5. Optional-arm nullability — PatternMatch with optional=True must have a non-empty arm_id
"""
import dataclasses

import pytest

from graphistry.compute.gfql.ir import (
    Aggregate,
    Apply,
    AntiSemiApply,
    CompilerError,
    Distinct,
    EdgeScan,
    Filter,
    GraphToRows,
    IndexScan,
    Join,
    Limit,
    LogicalPlan,
    NodeScan,
    OrderBy,
    PathProjection,
    PatternMatch,
    Project,
    RowSchema,
    RowsToGraph,
    SemiApply,
    Skip,
    Union,
    Unwind,
)
from graphistry.compute.gfql.ir.types import BoundPredicate, EdgeRef, ListType, NodeRef, PathType, ScalarType
from graphistry.compute.gfql.ir.verifier import verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns(op_id: int = 1, label: str = "Person") -> NodeScan:
    return NodeScan(op_id=op_id, label=label)


def _errors(plan: LogicalPlan) -> list[str]:
    """Return error messages for convenience."""
    return [e.message for e in verify(plan)]


def _inject(op: LogicalPlan, field: str, value: object) -> LogicalPlan:
    """Return a copy of *op* with *field* force-set to *value* (bypasses type constraints).

    dataclasses.replace() accepts wrong-typed values without raising — it only raises
    TypeError for unknown field names. object.__setattr__ is needed to bypass the
    frozen=True guard on the dataclass.
    """
    copy = dataclasses.replace(op)
    object.__setattr__(copy, field, value)
    return copy  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Positive cases — valid plans must produce zero errors
# ---------------------------------------------------------------------------

class TestVerifyPositive:
    def test_single_leaf_no_errors(self) -> None:
        assert verify(_ns()) == []

    def test_edge_scan_no_errors(self) -> None:
        assert verify(EdgeScan(op_id=1, edge_type="KNOWS")) == []

    def test_linear_chain_no_errors(self) -> None:
        scan = _ns(op_id=1)
        filt = Filter(op_id=2, input=scan, predicate=BoundPredicate(expression="n.age > 0"))
        assert verify(filt) == []

    def test_union_no_errors(self) -> None:
        plan = Union(op_id=3, left=_ns(op_id=1), right=_ns(op_id=2, label="Company"))
        assert verify(plan) == []

    def test_apply_no_errors(self) -> None:
        plan = Apply(op_id=2, input=_ns(op_id=1), subquery=_ns(op_id=3, label="X"))
        assert verify(plan) == []

    def test_semi_apply_no_errors(self) -> None:
        plan = SemiApply(op_id=2, input=_ns(op_id=1), subquery=_ns(op_id=3, label="X"))
        assert verify(plan) == []

    def test_anti_semi_apply_no_errors(self) -> None:
        plan = AntiSemiApply(op_id=2, input=_ns(op_id=1), subquery=_ns(op_id=3, label="X"))
        assert verify(plan) == []

    def test_unwind_no_errors(self) -> None:
        plan = Unwind(op_id=2, input=_ns(op_id=1), variable="x")
        assert verify(plan) == []

    def test_index_scan_no_errors(self) -> None:
        plan = IndexScan(
            op_id=1,
            predicate=BoundPredicate(expression="n.id = 1"),
            residual_predicates=[BoundPredicate(expression="n.age > 0")],
        )
        assert verify(plan) == []

    def test_optional_pattern_match_with_arm_id_no_errors(self) -> None:
        plan = PatternMatch(
            op_id=1,
            optional=True,
            arm_id="arm_0",
            predicates=[BoundPredicate(expression="n.x > 0")],
        )
        assert verify(plan) == []

    def test_non_optional_pattern_match_no_arm_id_no_errors(self) -> None:
        plan = PatternMatch(op_id=1, optional=False, arm_id=None)
        assert verify(plan) == []

    def test_schema_with_valid_logical_types_no_errors(self) -> None:
        schema = RowSchema(columns={
            "n": NodeRef(),
            "age": ScalarType(kind="int64", nullable=False),
            "rel": EdgeRef(),
            "path": PathType(min_hops=1, max_hops=5),
        })
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    def test_default_op_id_zero_exempt_from_uniqueness(self) -> None:
        # op_id=0 means "unassigned" — multiple zeros are allowed
        a = NodeScan(op_id=0)
        b = NodeScan(op_id=0)
        plan = Union(op_id=0, left=a, right=b)
        assert verify(plan) == []

    def test_project_no_errors(self) -> None:
        assert verify(Project(op_id=2, input=_ns(op_id=1))) == []

    def test_aggregate_no_errors(self) -> None:
        assert verify(Aggregate(op_id=2, input=_ns(op_id=1))) == []

    def test_distinct_no_errors(self) -> None:
        assert verify(Distinct(op_id=2, input=_ns(op_id=1))) == []

    def test_order_by_no_errors(self) -> None:
        assert verify(OrderBy(op_id=2, input=_ns(op_id=1))) == []

    def test_limit_no_errors(self) -> None:
        assert verify(Limit(op_id=2, input=_ns(op_id=1), count=10)) == []

    def test_skip_no_errors(self) -> None:
        assert verify(Skip(op_id=2, input=_ns(op_id=1), count=5)) == []

    def test_join_no_errors(self) -> None:
        plan = Join(op_id=3, left=_ns(op_id=1), right=_ns(op_id=2, label="Co"))
        assert verify(plan) == []

    def test_graph_to_rows_no_errors(self) -> None:
        assert verify(GraphToRows(op_id=2, input=_ns(op_id=1), variant="nodes")) == []

    def test_rows_to_graph_no_errors(self) -> None:
        plan = RowsToGraph(op_id=2, input=_ns(op_id=1), node_id_col="id", src_col="src", dst_col="dst")
        assert verify(plan) == []

    def test_path_projection_no_errors(self) -> None:
        plan = PathProjection(op_id=2, input=_ns(op_id=1), path_var="p", hop_count_col="hops")
        assert verify(plan) == []


# ---------------------------------------------------------------------------
# Invariant 1: op_id uniqueness
# ---------------------------------------------------------------------------

class TestOpIdUniqueness:
    def test_duplicate_nonzero_op_ids_caught(self) -> None:
        scan = _ns(op_id=1)
        filt = Filter(op_id=1, input=scan)  # same op_id as scan
        errs = _errors(filt)
        assert any("op_id" in e and "1" in e for e in errs)

    def test_duplicate_in_deep_tree_caught(self) -> None:
        a = _ns(op_id=1)
        b = _ns(op_id=2)
        c = Union(op_id=1, left=a, right=b)  # duplicates a's op_id
        errs = _errors(c)
        assert any("op_id" in e and "1" in e for e in errs)

    def test_all_unique_no_error(self) -> None:
        a = _ns(op_id=1)
        b = _ns(op_id=2)
        c = Union(op_id=3, left=a, right=b)
        assert verify(c) == []


# ---------------------------------------------------------------------------
# Invariant 2: Dangling references
# ---------------------------------------------------------------------------

class TestDanglingRefs:
    def test_non_logicalplan_input_caught(self) -> None:
        bad = _inject(Filter(op_id=2, input=_ns(op_id=1)), "input", "not_a_plan")
        errs = _errors(bad)
        assert any("dangling" in e.lower() or "input" in e.lower() for e in errs)

    def test_non_logicalplan_left_caught(self) -> None:
        bad = _inject(Union(op_id=1, left=_ns(op_id=2), right=_ns(op_id=3)), "left", 42)
        errs = _errors(bad)
        assert any("dangling" in e.lower() for e in errs)

    def test_non_logicalplan_right_caught(self) -> None:
        bad = _inject(Union(op_id=1, left=_ns(op_id=2), right=_ns(op_id=3)), "right", {"bad": True})
        errs = _errors(bad)
        assert any("dangling" in e.lower() for e in errs)

    def test_non_logicalplan_subquery_caught(self) -> None:
        bad = _inject(Apply(op_id=2, input=_ns(op_id=1), subquery=_ns(op_id=3)), "subquery", [])
        errs = _errors(bad)
        assert any("dangling" in e.lower() for e in errs)

    def test_none_input_is_valid(self) -> None:
        # None means "leaf" — not a dangling ref; give it a valid predicate
        filt = Filter(op_id=1, input=None, predicate=BoundPredicate(expression="n.x > 0"))
        assert verify(filt) == []

    def test_none_subquery_is_valid(self) -> None:
        plan = Apply(op_id=1, input=None, subquery=None)
        assert verify(plan) == []


# ---------------------------------------------------------------------------
# Invariant 3: Predicate scope
# ---------------------------------------------------------------------------

class TestPredicateScope:
    # --- PatternMatch ---
    def test_empty_expression_in_pattern_match_caught(self) -> None:
        plan = PatternMatch(op_id=1, predicates=[BoundPredicate(expression="")])
        errs = _errors(plan)
        assert any("predicate" in e.lower() or "expression" in e.lower() for e in errs)

    def test_pattern_match_multiple_predicates_one_empty_caught(self) -> None:
        plan = PatternMatch(op_id=1, predicates=[
            BoundPredicate(expression="n.age > 0"),
            BoundPredicate(expression=""),           # empty — should flag predicate[1]
            BoundPredicate(expression="x.id = 5"),
        ])
        errs = _errors(plan)
        assert any("predicate[1]" in e for e in errs)

    def test_nonempty_expression_ok(self) -> None:
        plan = PatternMatch(op_id=1, predicates=[BoundPredicate(expression="n.age > 0")])
        assert verify(plan) == []

    def test_no_predicates_ok(self) -> None:
        plan = PatternMatch(op_id=1, predicates=[])
        assert verify(plan) == []

    # --- Filter ---
    def test_empty_filter_predicate_caught(self) -> None:
        plan = Filter(op_id=1, input=None, predicate=BoundPredicate(expression=""))
        errs = _errors(plan)
        assert any("predicate" in e.lower() or "expression" in e.lower() for e in errs)

    def test_nonempty_filter_predicate_ok(self) -> None:
        plan = Filter(op_id=1, input=None, predicate=BoundPredicate(expression="n.x > 0"))
        assert verify(plan) == []

    # --- IndexScan ---
    def test_empty_index_scan_predicate_caught(self) -> None:
        plan = IndexScan(op_id=1, predicate=BoundPredicate(expression=""))
        errs = _errors(plan)
        assert any("predicate" in e.lower() or "expression" in e.lower() for e in errs)

    def test_empty_index_scan_residual_predicate_caught(self) -> None:
        plan = IndexScan(
            op_id=1,
            predicate=BoundPredicate(expression="n.id = 1"),
            residual_predicates=[BoundPredicate(expression="")],
        )
        errs = _errors(plan)
        assert any("residual" in e.lower() for e in errs)


# ---------------------------------------------------------------------------
# Invariant 4: Output schema consistency
# ---------------------------------------------------------------------------

class TestOutputSchemaConsistency:
    def test_non_logicaltype_column_value_caught(self) -> None:
        bad_schema = RowSchema(columns={"x": "not_a_type"})  # type: ignore[arg-type]
        plan = NodeScan(op_id=1, output_schema=bad_schema)
        errs = _errors(plan)
        assert any("schema" in e.lower() or "column" in e.lower() for e in errs)

    def test_integer_column_value_caught(self) -> None:
        bad_schema = RowSchema(columns={"count": 42})  # type: ignore[arg-type]
        plan = NodeScan(op_id=1, output_schema=bad_schema)
        errs = _errors(plan)
        assert any("column" in e.lower() for e in errs)

    def test_valid_schema_no_error(self) -> None:
        schema = RowSchema(columns={"n": NodeRef()})
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    def test_empty_schema_no_error(self) -> None:
        plan = NodeScan(op_id=1, output_schema=RowSchema())
        assert verify(plan) == []

    def test_pathtype_in_schema_no_error(self) -> None:
        schema = RowSchema(columns={"p": PathType(min_hops=1, max_hops=5)})
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    def test_listtype_in_schema_no_error(self) -> None:
        schema = RowSchema(columns={"tags": ListType(element_type=ScalarType(kind="string"))})
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    @pytest.mark.xfail(
        strict=True,
        reason="Known gap: verifier does not yet recurse into ListType.element_type. "
               "Flip to xpass (remove decorator) when element_type recursion is added.",
    )
    def test_listtype_element_type_recursion_not_yet_implemented(self) -> None:
        # When element_type recursion is added, invalid element_type should be caught.
        bad_list = ListType(element_type="not_a_type")  # type: ignore[arg-type]
        schema = RowSchema(columns={"items": bad_list})
        plan = NodeScan(op_id=1, output_schema=schema)
        errs = _errors(plan)
        assert len(errs) >= 1  # expected to fail until recursion is implemented


# ---------------------------------------------------------------------------
# Invariant 5: Optional-arm nullability contract
# ---------------------------------------------------------------------------

class TestOptionalArmNullability:
    def test_optional_without_arm_id_caught(self) -> None:
        plan = PatternMatch(op_id=1, optional=True, arm_id=None)
        errs = _errors(plan)
        assert any("arm_id" in e.lower() or "optional" in e.lower() for e in errs)

    def test_optional_with_empty_arm_id_caught(self) -> None:
        plan = PatternMatch(op_id=1, optional=True, arm_id="")
        errs = _errors(plan)
        assert any("arm_id" in e.lower() or "optional" in e.lower() for e in errs)

    def test_optional_with_arm_id_ok(self) -> None:
        plan = PatternMatch(op_id=1, optional=True, arm_id="arm_0")
        assert verify(plan) == []

    def test_non_optional_without_arm_id_ok(self) -> None:
        plan = PatternMatch(op_id=1, optional=False, arm_id=None)
        assert verify(plan) == []


# ---------------------------------------------------------------------------
# Multi-error accumulation
# ---------------------------------------------------------------------------

class TestErrorAccumulation:
    def test_multiple_violations_all_reported(self) -> None:
        # op_id=1 on both NodeScan and PatternMatch (duplicate) + optional without arm_id
        scan = NodeScan(op_id=1)
        bad_match = PatternMatch(op_id=1, optional=True, arm_id=None)  # dup op_id + missing arm_id
        plan = Union(op_id=2, left=scan, right=bad_match)
        errs = _errors(plan)
        assert any("op_id" in e and "1" in e for e in errs), "expected duplicate op_id error"
        assert any("arm_id" in e.lower() or "optional" in e.lower() for e in errs), "expected arm_id error"
        assert len(errs) >= 2

    def test_three_violations_in_one_node(self) -> None:
        # bad schema + empty predicate + optional without arm_id, all on one PatternMatch
        bad_schema = RowSchema(columns={"x": "bad"})  # type: ignore[arg-type]
        plan = PatternMatch(
            op_id=1,
            optional=True,
            arm_id=None,
            predicates=[BoundPredicate(expression="")],
            output_schema=bad_schema,
        )
        errs = _errors(plan)
        assert any("predicate" in e.lower() or "expression" in e.lower() for e in errs)
        assert any("arm_id" in e.lower() or "optional" in e.lower() for e in errs)
        assert any("schema" in e.lower() or "column" in e.lower() for e in errs)
        assert len(errs) == 3  # exactly these three, no extras
