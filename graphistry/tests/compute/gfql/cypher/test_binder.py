import pytest

from graphistry.compute.ast import ASTCall
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher, cypher_to_gfql
from graphistry.compute.gfql.cypher.ast import CypherGraphQuery, CypherUnionQuery
from graphistry.compute.gfql.cypher.lowering import (
    _alias_target,
    _bound_visible_aliases,
    _merge_bound_params,
    _return_references_optional_only_alias,
    lower_match_clause,
)
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundVariable, ScopeFrame, SemanticTable
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.types import EdgeRef, NodeRef, PathType, ScalarType
from typing import Any, List, Tuple, cast


def test_cypher_frontend_binder_module_imports() -> None:
    binder = FrontendBinder()
    assert isinstance(binder, FrontendBinder)


def test_cypher_frontend_binder_returns_bound_ir_placeholder() -> None:
    ast = parse_cypher("RETURN 1 AS x")
    bound = FrontendBinder().bind(ast, PlanContext())
    assert isinstance(bound, BoundIR)


def _capture_binder_calls(monkeypatch: pytest.MonkeyPatch) -> List[Tuple[object, PlanContext]]:
    calls: List[Tuple[object, PlanContext]] = []

    def _fake_bind(self: FrontendBinder, ast: object, ctx: PlanContext) -> BoundIR:
        calls.append((ast, ctx))
        return BoundIR()

    monkeypatch.setattr(FrontendBinder, "bind", _fake_bind)
    return calls


def _stub_bound_ir(monkeypatch: pytest.MonkeyPatch, bound_ir: BoundIR) -> None:
    def _fake_bind(self: FrontendBinder, ast: object, ctx: PlanContext) -> BoundIR:
        _ = (self, ast, ctx)
        return bound_ir

    monkeypatch.setattr(FrontendBinder, "bind", _fake_bind)


def test_compile_cypher_invokes_binder_prepass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("RETURN 1 AS x")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][1], PlanContext)


def test_compile_cypher_rebinds_after_normalization(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("MATCH (n) RETURN n")
    assert compiled is not None
    assert len(calls) >= 2


def test_cypher_to_gfql_invokes_binder_prepass(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    chain = cypher_to_gfql("RETURN 1 AS x")
    assert chain is not None
    assert len(calls) >= 1


def test_compile_cypher_union_invokes_binder_with_union_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("RETURN 1 AS x UNION RETURN 2 AS x")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][0], CypherUnionQuery)


def test_compile_cypher_graph_constructor_invokes_binder_with_graph_ast(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _capture_binder_calls(monkeypatch)
    compiled = compile_cypher("GRAPH { MATCH (a)-[r]->(b) }")
    assert compiled is not None
    assert len(calls) >= 1
    assert isinstance(calls[0][0], CypherGraphQuery)


def test_binder_does_not_mutate_input_ast() -> None:
    ast = parse_cypher("MATCH (n) RETURN n")
    before = repr(ast)
    bound = FrontendBinder().bind(ast, PlanContext())
    assert isinstance(bound, BoundIR)
    assert repr(ast) == before


def test_compile_cypher_uses_bound_ir_params_when_runtime_params_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_bound_ir(monkeypatch, BoundIR(params={"x": 7}))
    compiled = cast(Any, compile_cypher("RETURN $x AS x"))
    projection_call = cast(ASTCall, compiled.chain.chain[1])
    assert projection_call.params["items"] == [("x", 7)]


def test_compile_cypher_runtime_params_override_bound_ir_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_bound_ir(monkeypatch, BoundIR(params={"x": 7}))
    compiled = cast(Any, compile_cypher("RETURN $x AS x", params={"x": 3}))
    projection_call = cast(ASTCall, compiled.chain.chain[1])
    assert projection_call.params["items"] == [("x", 3)]


def test_compile_cypher_uses_bound_ir_scope_membership_for_initial_with_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = cast(Any, compile_cypher("MATCH (a) WITH a RETURN a"))
    baseline_rows_calls = [cast(ASTCall, op) for op in baseline.chain.chain if isinstance(op, ASTCall)]
    assert not any("binding_ops" in call.params for call in baseline_rows_calls)

    _stub_bound_ir(monkeypatch, BoundIR())
    no_scope = cast(Any, compile_cypher("MATCH (a) WITH a RETURN a"))
    no_scope_rows_calls = [cast(ASTCall, op) for op in no_scope.chain.chain if isinstance(op, ASTCall)]
    assert not any("binding_ops" in call.params for call in no_scope_rows_calls)

    _stub_bound_ir(
        monkeypatch,
        BoundIR(
            scope_stack=[
                ScopeFrame(
                    visible_vars=frozenset({"a"}),
                    schema=RowSchema(),
                    origin_clause="WITH",
                )
            ]
        ),
    )
    compiled = cast(Any, compile_cypher("MATCH (a) WITH a RETURN a"))
    rows_calls = [cast(ASTCall, op) for op in compiled.chain.chain if isinstance(op, ASTCall)]
    assert not any("binding_ops" in call.params for call in rows_calls)


def test_bound_visible_aliases_uses_latest_scope_frame_only() -> None:
    visible = _bound_visible_aliases(
        BoundIR(
            scope_stack=[
                ScopeFrame(visible_vars=frozenset({"a", "b"}), schema=RowSchema(), origin_clause="MATCH"),
                ScopeFrame(visible_vars=frozenset({"a"}), schema=RowSchema(), origin_clause="WITH"),
            ]
        )
    )
    assert visible == frozenset({"a"})


def test_compile_cypher_uses_bound_ir_semantic_table_entity_kind_for_alias_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _stub_bound_ir(
        monkeypatch,
        BoundIR(
            semantic_table=SemanticTable(
                variables={
                    "r": BoundVariable(
                        name="r",
                        logical_type=ScalarType(kind="edge"),
                        nullable=False,
                        null_extended_from=frozenset(),
                        entity_kind="node",
                    )
                }
            )
        ),
    )
    compiled = cast(Any, compile_cypher("MATCH (a)-[r]->(b) RETURN r"))
    row_call = next(
        cast(ASTCall, op)
        for op in compiled.chain.chain
        if isinstance(op, ASTCall) and op.params.get("source") == "r"
    )
    assert row_call.params["table"] == "nodes"


def test_merge_bound_params_filters_binder_metadata_keys() -> None:
    merged = _merge_bound_params(
        params=None,
        bound_params={
            "x": 7,
            "_binder_schema_confidence": 0.91,
            "_binder_parameter_names": ("x",),
        },
    )
    assert merged == {"x": 7}


def test_optional_only_alias_detection_accepts_nullable_alias_hints() -> None:
    parsed = cast(Any, parse_cypher("MATCH (a)-[:R]->(c) OPTIONAL MATCH (a)-[:R]->(b) RETURN a"))
    alias_targets = {}
    for clause in parsed.matches:
        alias_targets.update(_alias_target(lower_match_clause(clause)))

    assert _return_references_optional_only_alias(
        parsed,
        alias_targets=alias_targets,
        params=None,
    ) is False
    assert _return_references_optional_only_alias(
        parsed,
        alias_targets=alias_targets,
        params=None,
        bound_nullable_aliases=frozenset({"a"}),
    ) is True


def test_binder_optional_match_tracks_null_extended_from() -> None:
    query = (
        "MATCH (m:M) "
        "OPTIONAL MATCH (m)-[:T1]->(a:A) "
        "OPTIONAL MATCH (m)-[:T2]->(b:B) "
        "RETURN m.id AS mid, "
        "CASE a WHEN null THEN 'no-a' ELSE a.id END AS aid, "
        "CASE b WHEN null THEN 'no-b' ELSE b.id END AS bid"
    )
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    assert bound.semantic_table.variables["mid"].null_extended_from == frozenset()
    assert bound.semantic_table.variables["aid"].null_extended_from == frozenset({"optional_arm_1"})
    assert bound.semantic_table.variables["bid"].null_extended_from == frozenset({"optional_arm_2"})

    optional_parts = [part for part in bound.query_parts if part.clause == "OPTIONAL MATCH"]
    assert len(optional_parts) == 2
    assert optional_parts[0].metadata["arm_id"] == "optional_arm_1"
    assert optional_parts[1].metadata["arm_id"] == "optional_arm_2"


def test_binder_with_boundary_scope_lineage() -> None:
    query = (
        "MATCH (t:Tag) "
        "WITH t.tagId AS knownTagId "
        "MATCH (post:Post)-[:HAS_TAG]->(x:Tag {tagId: knownTagId}) "
        "RETURN post.id AS pid"
    )
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    assert [part.clause for part in bound.query_parts] == ["MATCH", "WITH", "MATCH", "RETURN"]
    second_match = bound.query_parts[2]
    assert second_match.inputs == frozenset({"knownTagId"})
    assert {"knownTagId", "post", "x"} <= second_match.outputs
    assert set(bound.semantic_table.variables) == {"pid"}


def test_binder_assigns_node_edge_path_and_scalar_types() -> None:
    query = "MATCH p = (a:Person)-[r:KNOWS]->(b:Person) UNWIND [1,2] AS i RETURN p, a, r, i"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    vars_ = bound.semantic_table.variables
    assert isinstance(vars_["p"].logical_type, PathType)
    assert isinstance(vars_["a"].logical_type, NodeRef)
    assert vars_["a"].logical_type.labels == frozenset({"Person"})
    assert isinstance(vars_["r"].logical_type, EdgeRef)
    assert vars_["r"].logical_type.type == "KNOWS"
    assert isinstance(vars_["i"].logical_type, ScalarType)


def test_binder_extracts_parameter_names() -> None:
    query = "MATCH (n:Person {id: $id}) RETURN n"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    assert bound.params["_binder_parameter_names"] == ("id",)


def test_binder_union_merges_branch_bindings() -> None:
    bound = FrontendBinder().bind(parse_cypher("RETURN 1 AS x UNION RETURN 2 AS x"), PlanContext())
    assert "x" in bound.semantic_table.variables
    assert any(part.clause == "UNION" for part in bound.query_parts)


def test_binder_with_scope_boundary_keeps_projected_alias_only() -> None:
    query = "MATCH (n:Person) WITH n AS m RETURN m"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    assert set(bound.semantic_table.variables) == {"m"}
    assert bound.semantic_table.variables["m"].entity_kind == "node"
    assert bound.query_parts[1].outputs == frozenset({"m"})


def test_binder_unresolved_name_failure_after_with_scope_reset() -> None:
    query = "MATCH (n:Person) WITH n AS m RETURN n"
    with pytest.raises(GFQLValidationError, match="Unresolved identifier"):
        FrontendBinder().bind(parse_cypher(query), PlanContext(), strict_name_resolution=True)


def test_binder_unresolved_name_failure_in_compound_expression() -> None:
    query = "MATCH (n:Person) RETURN ghost + 1 AS x"
    with pytest.raises(GFQLValidationError, match="Unresolved identifier"):
        FrontendBinder().bind(parse_cypher(query), PlanContext(), strict_name_resolution=True)


def test_binder_unresolved_name_failure_in_function_argument_expression() -> None:
    query = "MATCH (n:Person) RETURN coalesce(ghost, 1) AS x"
    with pytest.raises(GFQLValidationError, match="Unresolved identifier"):
        FrontendBinder().bind(parse_cypher(query), PlanContext(), strict_name_resolution=True)


def test_binder_unresolved_name_failure_in_mixed_known_unknown_expression() -> None:
    query = "MATCH (n:Person) RETURN n.id + ghost AS x"
    with pytest.raises(GFQLValidationError, match="Unresolved identifier"):
        FrontendBinder().bind(parse_cypher(query), PlanContext(), strict_name_resolution=True)


def test_binder_strict_name_resolution_allows_known_function_expression() -> None:
    query = "MATCH (n:Person) RETURN coalesce(n.id, 1) AS x"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext(), strict_name_resolution=True)
    assert "x" in bound.semantic_table.variables


def test_binder_unwind_extends_existing_scope() -> None:
    query = "MATCH (n:Person) UNWIND [1, 2] AS x RETURN n, x"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    unwind_part = next(part for part in bound.query_parts if part.clause == "UNWIND")
    assert unwind_part.inputs == frozenset({"n"})
    assert unwind_part.outputs == frozenset({"n", "x"})


def test_binder_label_narrowing_from_match_labels_and_where_conjunction() -> None:
    query = "MATCH (n:Person) WHERE n:Admin AND n:Active RETURN n"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    n_var = bound.semantic_table.variables["n"]
    assert isinstance(n_var.logical_type, NodeRef)
    assert n_var.logical_type.labels == frozenset({"Person", "Admin", "Active"})


def test_binder_label_narrowing_does_not_apply_for_or_expression() -> None:
    query = "MATCH (n) WHERE n:Admin OR n:Active RETURN n"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    n_var = bound.semantic_table.variables["n"]
    assert isinstance(n_var.logical_type, NodeRef)
    assert n_var.logical_type.labels == frozenset()


def test_binder_label_narrowing_does_not_apply_for_not_expression() -> None:
    query = "MATCH (n:Person) WHERE NOT n:Admin RETURN n"
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())

    n_var = bound.semantic_table.variables["n"]
    assert isinstance(n_var.logical_type, NodeRef)
    assert n_var.logical_type.labels == frozenset({"Person"})


def test_binder_schema_confidence_min_rule_count_and_operand_inheritance() -> None:
    query = (
        "MATCH (n:Person) "
        "WITH n.id AS pid, n AS node_alias "
        "RETURN pid AS pid_out, node_alias.id AS node_id, pid + node_alias.id AS mixed, count(node_alias) AS cnt, 1 AS one"
    )
    bound = FrontendBinder().bind(parse_cypher(query), PlanContext())
    confidence = bound.params["_binder_schema_confidence"]
    assert isinstance(confidence, dict)

    assert confidence["pid_out"] == "propagated"  # operand inheritance from pid
    assert confidence["node_id"] == "propagated"  # property-level demotion
    assert confidence["mixed"] == "propagated"  # min-rule over declared + propagated
    assert confidence["cnt"] == "declared"  # strong aggregate semantics
    assert confidence["one"] == "declared"  # strong literal semantics


def test_binder_unresolved_identifier_code_is_e204() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        FrontendBinder().bind(parse_cypher("RETURN ghost"), PlanContext(), strict_name_resolution=True)
    assert exc_info.value.code == ErrorCode.E204
