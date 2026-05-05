"""Tests for the T3 type/nullability metadata contract (#1300).

Two surfaces under test:

1. ``graphistry.compute.gfql.ir.metadata`` — pure helpers (is_nullable,
   with_nullable, widen_to_nullable, column_logical_type, column_is_nullable,
   merge_types, bound_variable_type).
2. The verifier's invariant 6 (type propagation continuity across a unary op's
   ``input`` slot).

These tests are intentionally narrow: they pin the contract callers in T2
strict-validation, T4 arrow bridge, and downstream rewrite passes will rely
on. Behavior beyond the documented contract is not asserted.
"""
from __future__ import annotations

import pytest

from graphistry.compute.gfql.ir.bound_ir import BoundVariable
from graphistry.compute.gfql.ir.logical_plan import (
    Distinct,
    Filter,
    NodeScan,
    OrderBy,
    PatternMatch,
    Project,
    RowSchema,
)
from graphistry.compute.gfql.ir.metadata import (
    bound_variable_is_nullable,
    bound_variable_type,
    column_is_nullable,
    column_logical_type,
    is_nullable,
    merge_types,
    widen_to_nullable,
    with_nullable,
)
from graphistry.compute.gfql.ir.types import (
    BoundPredicate,
    EdgeRef,
    ListType,
    NodeRef,
    PathType,
    ScalarType,
)
from graphistry.compute.gfql.ir.verifier import verify


# ---------------------------------------------------------------------------
# is_nullable / with_nullable / widen_to_nullable
# ---------------------------------------------------------------------------


class TestNullableHelpers:
    def test_is_nullable_scalar_default_is_true(self) -> None:
        assert is_nullable(ScalarType()) is True

    def test_is_nullable_scalar_explicit_false(self) -> None:
        assert is_nullable(ScalarType("int64", nullable=False)) is False

    @pytest.mark.parametrize(
        "structural",
        [NodeRef(), EdgeRef(), PathType(), ListType()],
    )
    def test_is_nullable_structural_types_are_not_nullable(self, structural: object) -> None:
        assert is_nullable(structural) is False  # type: ignore[arg-type]

    def test_with_nullable_returns_same_instance_when_unchanged(self) -> None:
        s = ScalarType("int64", nullable=True)
        assert with_nullable(s, True) is s

    def test_with_nullable_flips_flag(self) -> None:
        s = ScalarType("int64", nullable=False)
        flipped = with_nullable(s, True)
        assert isinstance(flipped, ScalarType)
        assert flipped.kind == "int64"
        assert flipped.nullable is True
        # Source stays frozen-immutable.
        assert s.nullable is False

    @pytest.mark.parametrize(
        "structural",
        [NodeRef(frozenset({"A"})), EdgeRef(type="REL"), PathType(1, 3), ListType()],
    )
    def test_with_nullable_passes_structural_types_through(self, structural: object) -> None:
        assert with_nullable(structural, True) is structural  # type: ignore[arg-type]
        assert with_nullable(structural, False) is structural  # type: ignore[arg-type]

    def test_widen_to_nullable_forces_true_on_scalar(self) -> None:
        s = ScalarType("int64", nullable=False)
        widened = widen_to_nullable(s)
        assert isinstance(widened, ScalarType)
        assert widened.nullable is True

    def test_widen_to_nullable_passes_structural_through(self) -> None:
        n = NodeRef(frozenset({"X"}))
        assert widen_to_nullable(n) is n


# ---------------------------------------------------------------------------
# column_logical_type / column_is_nullable
# ---------------------------------------------------------------------------


class TestSchemaAccessors:
    def _schema(self) -> RowSchema:
        return RowSchema(
            columns={
                "a": ScalarType("int64", nullable=True),
                "b": ScalarType("string", nullable=False),
                "n": NodeRef(frozenset({"Person"})),
            }
        )

    def test_column_logical_type_present(self) -> None:
        s = self._schema()
        assert column_logical_type(s, "a") == ScalarType("int64", nullable=True)

    def test_column_logical_type_absent_returns_none(self) -> None:
        assert column_logical_type(self._schema(), "missing") is None

    def test_column_is_nullable_scalar(self) -> None:
        s = self._schema()
        assert column_is_nullable(s, "a") is True
        assert column_is_nullable(s, "b") is False

    def test_column_is_nullable_structural_returns_none(self) -> None:
        # Structural types do not carry a nullable bit; helper returns None
        # so callers can distinguish "not nullable" from "kind has no answer".
        assert column_is_nullable(self._schema(), "n") is None

    def test_column_is_nullable_missing_returns_none(self) -> None:
        assert column_is_nullable(self._schema(), "missing") is None


# ---------------------------------------------------------------------------
# merge_types
# ---------------------------------------------------------------------------


class TestMergeTypes:
    def test_merge_scalar_same_kind_ors_nullability(self) -> None:
        merged = merge_types(
            ScalarType("int64", nullable=False),
            ScalarType("int64", nullable=True),
        )
        assert merged == ScalarType("int64", nullable=True)

    def test_merge_scalar_diff_kind_returns_none(self) -> None:
        assert merge_types(ScalarType("int64"), ScalarType("string")) is None

    def test_merge_scalar_unknown_takes_concrete_kind(self) -> None:
        merged = merge_types(
            ScalarType("unknown", nullable=False),
            ScalarType("int64", nullable=False),
        )
        assert merged == ScalarType("int64", nullable=False)

    def test_merge_scalar_unknown_both_sides(self) -> None:
        merged = merge_types(ScalarType("unknown"), ScalarType("unknown"))
        assert merged == ScalarType("unknown", nullable=True)

    def test_merge_node_unions_labels(self) -> None:
        merged = merge_types(
            NodeRef(frozenset({"Person"})),
            NodeRef(frozenset({"Admin"})),
        )
        assert merged == NodeRef(frozenset({"Person", "Admin"}))

    def test_merge_edge_widens_diverging_fields_to_none(self) -> None:
        merged = merge_types(
            EdgeRef(type="KNOWS", src_label="Person", dst_label="Person"),
            EdgeRef(type="WORKS_AT", src_label="Person", dst_label="Org"),
        )
        assert merged == EdgeRef(type=None, src_label="Person", dst_label=None)

    def test_merge_edge_keeps_equal_fields(self) -> None:
        merged = merge_types(
            EdgeRef(type="KNOWS", src_label="Person", dst_label="Person"),
            EdgeRef(type="KNOWS", src_label="Person", dst_label="Person"),
        )
        assert merged == EdgeRef(type="KNOWS", src_label="Person", dst_label="Person")

    def test_merge_list_recurses_into_element(self) -> None:
        merged = merge_types(
            ListType(ScalarType("int64", nullable=False)),
            ListType(ScalarType("int64", nullable=True)),
        )
        assert merged == ListType(ScalarType("int64", nullable=True))

    def test_merge_list_incompatible_elements_returns_none(self) -> None:
        merged = merge_types(
            ListType(ScalarType("int64")),
            ListType(ScalarType("string")),
        )
        assert merged is None

    def test_merge_path_takes_loosest_bounds(self) -> None:
        merged = merge_types(PathType(min_hops=1, max_hops=3), PathType(min_hops=2, max_hops=5))
        assert merged == PathType(min_hops=1, max_hops=5)

    @pytest.mark.parametrize(
        "left,right",
        [
            (ScalarType(), NodeRef()),
            (NodeRef(), EdgeRef()),
            (PathType(), ListType()),
            (ListType(), ScalarType()),
        ],
    )
    def test_merge_cross_kind_returns_none(self, left: object, right: object) -> None:
        assert merge_types(left, right) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# bound_variable_type
# ---------------------------------------------------------------------------


class TestBoundVariableType:
    def test_scalar_nullability_reconciled_from_bound_variable(self) -> None:
        bv = BoundVariable(
            name="age",
            logical_type=ScalarType("int64", nullable=False),
            nullable=True,
            null_extended_from=frozenset({"opt_arm"}),
            entity_kind="scalar",
        )
        assert bound_variable_type(bv) == ScalarType("int64", nullable=True)

    def test_scalar_nullability_kept_when_already_aligned(self) -> None:
        bv = BoundVariable(
            name="age",
            logical_type=ScalarType("int64", nullable=False),
            nullable=False,
            null_extended_from=frozenset(),
            entity_kind="scalar",
        )
        assert bound_variable_type(bv) == ScalarType("int64", nullable=False)

    @pytest.mark.parametrize(
        "logical_type,entity_kind",
        [
            (NodeRef(frozenset({"Person"})), "node"),
            (EdgeRef(type="KNOWS"), "edge"),
            (PathType(min_hops=1, max_hops=3), "scalar"),
            (ListType(ScalarType("int64")), "scalar"),
        ],
    )
    def test_structural_pass_through_ignores_bv_nullable(
        self, logical_type: object, entity_kind: str
    ) -> None:
        # Whole-row / structural variables — bv.nullable is recorded for
        # provenance but does not propagate onto the LogicalType today.
        bv = BoundVariable(
            name="x",
            logical_type=logical_type,  # type: ignore[arg-type]
            nullable=True,
            null_extended_from=frozenset({"opt_arm"}),
            entity_kind=entity_kind,  # type: ignore[arg-type]
        )
        assert bound_variable_type(bv) is logical_type


# ---------------------------------------------------------------------------
# bound_variable_is_nullable
# ---------------------------------------------------------------------------


class TestBoundVariableIsNullable:
    @pytest.mark.parametrize(
        "logical_type,entity_kind",
        [
            (ScalarType("int64", nullable=False), "scalar"),
            (NodeRef(frozenset({"Person"})), "node"),
            (EdgeRef(type="KNOWS"), "edge"),
            (PathType(), "scalar"),
            (ListType(), "scalar"),
        ],
    )
    def test_returns_bv_nullable_directly_across_kinds(
        self, logical_type: object, entity_kind: str
    ) -> None:
        # Source of truth is BoundVariable.nullable; the helper must NOT
        # silently drop it for structural variables (which is what
        # `is_nullable(bound_variable_type(bv))` would do).
        bv_nullable = BoundVariable(
            name="x",
            logical_type=logical_type,  # type: ignore[arg-type]
            nullable=True,
            null_extended_from=frozenset({"opt_arm"}),
            entity_kind=entity_kind,  # type: ignore[arg-type]
        )
        assert bound_variable_is_nullable(bv_nullable) is True
        bv_not_nullable = BoundVariable(
            name="x",
            logical_type=logical_type,  # type: ignore[arg-type]
            nullable=False,
            null_extended_from=frozenset(),
            entity_kind=entity_kind,  # type: ignore[arg-type]
        )
        assert bound_variable_is_nullable(bv_not_nullable) is False


# ---------------------------------------------------------------------------
# Verifier invariant 6 — propagation continuity
# ---------------------------------------------------------------------------


def _scan(name: str, columns: dict, op_id: int = 1) -> NodeScan:
    return NodeScan(label=name, op_id=op_id, output_schema=RowSchema(columns=columns))


class TestPropagationContinuity:
    def test_empty_schemas_are_skipped(self) -> None:
        # Default Project with empty output_schema — back-compat for plans
        # the lowering layer hasn't fully populated yet.
        plan = Project(input=_scan("Person", {}), op_id=2)
        assert verify(plan) == []

    def test_widening_nullability_passes(self) -> None:
        scan = _scan("Person", {"a": ScalarType("int64", nullable=False)})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=True)}),
        )
        assert verify(plan) == []

    def test_kind_mismatch_fails(self) -> None:
        scan = _scan("Person", {"x": ScalarType("int64")})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"x": NodeRef()}),
        )
        errors = verify(plan)
        assert any("changed kind across input edge" in e.message for e in errors)

    def test_narrowing_nullability_on_project_fails(self) -> None:
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        errors = verify(plan)
        assert any("narrowed nullability" in e.message for e in errors)

    def test_filter_may_narrow_nullability(self) -> None:
        # Filter is the carve-out — it can drop NULL rows, so a non-nullable
        # output is consistent with a nullable input.
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = Filter(
            input=scan,
            op_id=2,
            predicate=BoundPredicate(expression="a IS NOT NULL", references=frozenset({"a"})),
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        assert verify(plan) == []

    def test_patternmatch_non_optional_may_narrow_nullability(self) -> None:
        # Non-optional PatternMatch with a WHERE-style predicate can drop
        # rows where the pattern fails — same row-dropping semantics as
        # Filter, so it sits in the same carve-out.  Optional arms remain
        # locked nullable=True by invariant 5.
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = PatternMatch(
            input=scan,
            op_id=2,
            predicates=[BoundPredicate(expression="a IS NOT NULL", references=frozenset({"a"}))],
            optional=False,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        assert verify(plan) == []

    def test_distinct_must_preserve_nullability(self) -> None:
        # Distinct is NOT in the narrowing carve-out — preserves nullability.
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = Distinct(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        errors = verify(plan)
        assert any("narrowed nullability" in e.message for e in errors)

    def test_orderby_must_preserve_nullability(self) -> None:
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = OrderBy(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        errors = verify(plan)
        assert any("narrowed nullability" in e.message for e in errors)

    def test_new_column_has_no_propagation_constraint(self) -> None:
        # Newly-introduced columns aren't compared against the input schema.
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(
                columns={
                    "a": ScalarType("int64", nullable=True),  # carried
                    "b": ScalarType("string", nullable=False),  # newly introduced
                }
            ),
        )
        assert verify(plan) == []

    def test_no_input_slot_skipped(self) -> None:
        # Pure source ops with no `input` slot: nothing to compare against.
        scan = _scan("Person", {"a": ScalarType("int64", nullable=True)}, op_id=1)
        assert verify(scan) == []

    def test_dropping_column_is_allowed(self) -> None:
        # Project may drop columns from its input — only shared columns are
        # compared.
        scan = _scan(
            "Person",
            {
                "a": ScalarType("int64", nullable=True),
                "b": ScalarType("string", nullable=True),
            },
        )
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=True)}),
        )
        assert verify(plan) == []

    def test_node_to_node_label_change_ok(self) -> None:
        # Kind continuity is by family (NodeRef vs NodeRef), not by label set.
        scan = _scan("Person", {"n": NodeRef(frozenset({"Person"}))})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"n": NodeRef(frozenset({"Admin"}))}),
        )
        assert verify(plan) == []

    def test_listtype_kind_continuity(self) -> None:
        scan = _scan("Person", {"xs": ListType(ScalarType("int64"))})
        plan = Project(
            input=scan,
            op_id=2,
            output_schema=RowSchema(columns={"xs": ScalarType()}),
        )
        errors = verify(plan)
        assert any("changed kind across input edge" in e.message for e in errors)
