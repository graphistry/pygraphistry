from dataclasses import is_dataclass
from typing import Any, get_args, get_origin, get_type_hints


def test_compilation_module_surface() -> None:
    from graphistry.compute.gfql.ir.compilation import (
        CompilationState,
        GFQLSchema,
        GraphSchemaCatalog,
        PlanContext,
        QueryLanguage,
    )

    assert set(QueryLanguage.__members__.keys()) == {"CYPHER", "CHAIN_DSL"}
    assert GFQLSchema is GraphSchemaCatalog

    assert CompilationState is not None
    assert PlanContext is not None


def test_compilation_type_contracts() -> None:
    from graphistry.compute.gfql.ir.bound_ir import BoundIR, SemanticTable
    from graphistry.compute.gfql.ir.compilation import (
        CompilationState,
        GraphSchemaCatalog,
        PhysicalPlan,
        PlanContext,
        QueryLanguage,
    )
    from graphistry.compute.gfql.ir.logical_plan import LogicalPlan
    from graphistry.compute.gfql.ir.query_graph import QueryGraph

    compilation_hints = get_type_hints(CompilationState)
    plan_hints = get_type_hints(PlanContext)

    assert compilation_hints["frontend"] is QueryLanguage
    assert set(get_args(compilation_hints["frontend_ast"])) == {Any, type(None)}
    assert set(get_args(compilation_hints["bound_ir"])) == {BoundIR, type(None)}
    assert set(get_args(compilation_hints["semantic_table"])) == {SemanticTable, type(None)}
    assert set(get_args(compilation_hints["logical_plan"])) == {LogicalPlan, type(None)}
    assert set(get_args(compilation_hints["query_graph"])) == {QueryGraph, type(None)}
    assert set(get_args(compilation_hints["physical_plan"])) == {PhysicalPlan, type(None)}

    assert plan_hints["catalog"] is GraphSchemaCatalog


def test_compilation_defaults() -> None:
    from graphistry.compute.gfql.ir.compilation import CompilationState, QueryLanguage

    state = CompilationState()
    assert state.frontend is QueryLanguage.CYPHER
    assert state.frontend_ast is None
    assert state.query_text == ""
    assert state.bound_ir is None
    assert state.semantic_table is None


def test_graph_schema_catalog_default_factory_is_not_shared() -> None:
    from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog

    left = GraphSchemaCatalog()
    right = GraphSchemaCatalog()

    left.metadata["x"] = 1

    assert left.metadata["x"] == 1
    assert "x" not in right.metadata


def test_ir_package_exports_compilation_symbols() -> None:
    from graphistry.compute.gfql.ir import (
        CompilationState,
        GFQLSchema,
        GraphSchemaCatalog,
        PlanContext,
        QueryLanguage,
    )

    assert CompilationState is not None
    assert PlanContext is not None
    assert QueryLanguage.CYPHER.name == "CYPHER"
    assert GFQLSchema is GraphSchemaCatalog


def test_compilation_state_is_mutable_dataclass() -> None:
    from graphistry.compute.gfql.ir.compilation import CompilationState

    assert is_dataclass(CompilationState)
    assert getattr(CompilationState, "__dataclass_params__").frozen is False

    state = CompilationState()
    state.query_text = "MATCH (n) RETURN n"
    assert state.query_text.startswith("MATCH")
