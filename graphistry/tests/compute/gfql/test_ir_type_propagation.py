"""Tests for verifier type/nullability propagation continuity (#1300).

These tests are intentionally narrow: they pin the contract callers in T2
strict-validation, T4 arrow bridge, and downstream rewrite passes will rely
on. Behavior beyond the documented contract is not asserted.
"""
from __future__ import annotations

from graphistry.compute.gfql.ir.logical_plan import (
    Distinct,
    Filter,
    NodeScan,
    OrderBy,
    PatternMatch,
    Project,
    RowSchema,
)
from graphistry.compute.gfql.ir.types import (
    BoundPredicate,
    ListType,
    NodeRef,
    ScalarType,
)
from graphistry.compute.gfql.ir.verifier import verify


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


# ---------------------------------------------------------------------------
# Seam amplification: Cypher lowering modules emit LogicalPlan-shaped output
# that the IR verifier consumes.  These tests pin that import coexistence and
# invariant-6 behavior stay stable across projection/reentry helper modules.
# ---------------------------------------------------------------------------


class TestSeamWith1303LoweringSplit:
    def test_post_1303_modules_and_ir_verifier_coimport(self) -> None:
        # Behavioral only: import the split Cypher helper modules beside IR
        # verifier without asserting private helper ownership.
        import sys

        # The import-coexistence smoke: each module must land in sys.modules
        # without raising.
        from graphistry.compute.gfql.cypher import lowering  # noqa: F401
        from graphistry.compute.gfql.cypher import projection_planning  # noqa: F401
        from graphistry.compute.gfql.cypher.reentry import compiletime  # noqa: F401
        from graphistry.compute.gfql.ir.verifier import verify as _verify

        assert "graphistry.compute.gfql.cypher.lowering" in sys.modules
        assert "graphistry.compute.gfql.cypher.projection_planning" in sys.modules
        assert "graphistry.compute.gfql.cypher.reentry.compiletime" in sys.modules
        assert callable(_verify)

    def test_realistic_match_with_filter_chain_passes_invariant_6(self) -> None:
        # Plan shape representative of what the post-#1303 lowering pipeline
        # emits for `MATCH (n:Person) WHERE n.age IS NOT NULL RETURN n.id`:
        # NodeScan → PatternMatch (non-optional) → Filter (carve-out) →
        # Project. Each edge respects type/nullability propagation continuity.
        from graphistry.compute.gfql.ir.types import NodeRef as _NodeRef

        scan = NodeScan(
            label="Person",
            op_id=1,
            output_schema=RowSchema(
                columns={
                    "n": _NodeRef(frozenset({"Person"})),
                    "age": ScalarType("int64", nullable=True),
                    "id": ScalarType("int64", nullable=False),
                }
            ),
        )
        match = PatternMatch(
            input=scan,
            op_id=2,
            optional=False,
            output_schema=RowSchema(
                columns={
                    "n": _NodeRef(frozenset({"Person"})),
                    "age": ScalarType("int64", nullable=True),
                    "id": ScalarType("int64", nullable=False),
                }
            ),
        )
        filt = Filter(
            input=match,
            op_id=3,
            predicate=BoundPredicate(
                expression="age IS NOT NULL", references=frozenset({"age"})
            ),
            output_schema=RowSchema(
                columns={
                    "n": _NodeRef(frozenset({"Person"})),
                    # Filter narrows nullability — invariant 6 carve-out.
                    "age": ScalarType("int64", nullable=False),
                    "id": ScalarType("int64", nullable=False),
                }
            ),
        )
        project = Project(
            input=filt,
            op_id=4,
            output_schema=RowSchema(
                columns={"id": ScalarType("int64", nullable=False)}
            ),
        )
        assert verify(project) == []

    def test_optional_arm_chain_produces_only_invariant_5_signal(self) -> None:
        # Plan shape representative of OPTIONAL MATCH lowered via the post-#1303
        # path: invariant 5 (PatternMatch optional=True must produce nullable
        # outputs) is the active signal; invariant 6 must NOT double-flag here.
        scan = NodeScan(
            label="Person",
            op_id=1,
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        # An optional arm with a non-nullable scalar output is the exact
        # shape invariant 5 catches; arm_id is set so the missing-arm-id
        # error path doesn't muddy this test.
        match = PatternMatch(
            input=scan,
            op_id=2,
            optional=True,
            arm_id="opt1",
            output_schema=RowSchema(columns={"a": ScalarType("int64", nullable=False)}),
        )
        errors = verify(match)
        # Exactly one error, from invariant 5 ("optional arms must produce
        # nullable outputs").  Invariant 6 stays silent because the input
        # already has nullable=False (no narrowing).
        assert len(errors) == 1
        assert "optional arms must produce nullable outputs" in errors[0].message
