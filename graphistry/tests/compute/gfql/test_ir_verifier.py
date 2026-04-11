"""Tests for LogicalPlan structural verifier (issue #1127 / M2-PR3).

Five invariants:
  1. op_id uniqueness — all operators in the tree have distinct op_ids (0 is exempt: unassigned)
  2. Dangling refs — input/left/right/subquery children must be LogicalPlan instances
  3. Predicate scope — PatternMatch predicates must have non-empty expression strings
  4. Output schema consistency — RowSchema columns values must be LogicalType instances
  5. Optional-arm nullability — PatternMatch with optional=True must have a non-empty arm_id
"""
from graphistry.compute.gfql.ir import (
    CompilerError,
    Filter,
    LogicalPlan,
    NodeScan,
    PatternMatch,
    RowSchema,
    Union,
)
from graphistry.compute.gfql.ir.types import BoundPredicate, NodeRef, ScalarType
from graphistry.compute.gfql.ir.verifier import verify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ns(op_id: int = 1, label: str = "Person") -> NodeScan:
    return NodeScan(op_id=op_id, label=label)


def _errors(plan: LogicalPlan) -> list[str]:
    """Return error messages for convenience."""
    return [e.message for e in verify(plan)]


# ---------------------------------------------------------------------------
# Positive cases — valid plans must produce zero errors
# ---------------------------------------------------------------------------

class TestVerifyPositive:
    def test_single_leaf_no_errors(self) -> None:
        assert verify(_ns()) == []

    def test_linear_chain_no_errors(self) -> None:
        scan = _ns(op_id=1)
        filt = Filter(op_id=2, input=scan, predicate=BoundPredicate(expression="n.age > 0"))
        assert verify(filt) == []

    def test_union_no_errors(self) -> None:
        plan = Union(op_id=3, left=_ns(op_id=1), right=_ns(op_id=2, label="Company"))
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
        schema = RowSchema(columns={"n": NodeRef(), "age": ScalarType(kind="int64", nullable=False)})
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    def test_default_op_id_zero_exempt_from_uniqueness(self) -> None:
        # op_id=0 means "unassigned" — multiple zeros are allowed
        a = NodeScan(op_id=0)
        b = NodeScan(op_id=0)
        plan = Union(op_id=0, left=a, right=b)
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
        # Manually construct a Filter with a bad input type (bypassing frozen via object())
        # We use object.__setattr__ since the dataclass is frozen
        scan = _ns(op_id=1)
        filt = Filter(op_id=2, input=scan)
        # Replace input with a non-plan sentinel via a plain dict trick:
        # Build a bad plan fixture using a subclass override
        import dataclasses
        bad = dataclasses.replace(filt)
        # We need to inject a bad value — use object.__setattr__ on the frozen instance
        object.__setattr__(bad, "input", "not_a_plan")
        errs = _errors(bad)
        assert any("dangling" in e.lower() or "input" in e.lower() for e in errs)

    def test_none_input_is_valid(self) -> None:
        # None means "leaf" — not a dangling ref
        filt = Filter(op_id=1, input=None)
        assert verify(filt) == []


# ---------------------------------------------------------------------------
# Invariant 3: Predicate scope (PatternMatch)
# ---------------------------------------------------------------------------

class TestPredicateScope:
    def test_empty_expression_in_pattern_match_caught(self) -> None:
        plan = PatternMatch(
            op_id=1,
            predicates=[BoundPredicate(expression="")],  # empty — invalid
        )
        errs = _errors(plan)
        assert any("predicate" in e.lower() or "expression" in e.lower() for e in errs)

    def test_nonempty_expression_ok(self) -> None:
        plan = PatternMatch(
            op_id=1,
            predicates=[BoundPredicate(expression="n.age > 0")],
        )
        assert verify(plan) == []

    def test_no_predicates_ok(self) -> None:
        plan = PatternMatch(op_id=1, predicates=[])
        assert verify(plan) == []


# ---------------------------------------------------------------------------
# Invariant 4: Output schema consistency
# ---------------------------------------------------------------------------

class TestOutputSchemaConsistency:
    def test_non_logicaltype_column_value_caught(self) -> None:
        bad_schema = RowSchema(columns={"x": "not_a_type"})  # type: ignore[arg-type]
        plan = NodeScan(op_id=1, output_schema=bad_schema)
        errs = _errors(plan)
        assert any("schema" in e.lower() or "column" in e.lower() for e in errs)

    def test_valid_schema_no_error(self) -> None:
        schema = RowSchema(columns={"n": NodeRef()})
        plan = NodeScan(op_id=1, output_schema=schema)
        assert verify(plan) == []

    def test_empty_schema_no_error(self) -> None:
        plan = NodeScan(op_id=1, output_schema=RowSchema())
        assert verify(plan) == []


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
        # op_id duplicate + optional without arm_id
        scan = NodeScan(op_id=1)
        bad_match = PatternMatch(op_id=1, optional=True, arm_id=None)  # dup op_id + missing arm_id
        plan = Union(op_id=2, left=scan, right=bad_match)
        errs = _errors(plan)
        assert len(errs) >= 2
