import pytest

from graphistry.compute.ast import ASTCall
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
from graphistry.compute.gfql.ir.types import ScalarType
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
    assert any("binding_ops" in call.params for call in rows_calls)


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
