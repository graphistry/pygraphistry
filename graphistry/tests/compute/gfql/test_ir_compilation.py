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
    from graphistry.compute.gfql.ir.compilation import (
        CompilationState,
        GraphSchemaCatalog,
        PlanContext,
        QueryLanguage,
    )

    compilation_hints = get_type_hints(CompilationState)
    plan_hints = get_type_hints(PlanContext)

    assert compilation_hints["frontend"] is QueryLanguage
    assert get_origin(compilation_hints["frontend_ast"]) is not None
    assert set(get_args(compilation_hints["frontend_ast"])) == {Any, type(None)}
    assert plan_hints["catalog"] is GraphSchemaCatalog


def test_compilation_defaults() -> None:
    from graphistry.compute.gfql.ir.compilation import CompilationState, QueryLanguage

    state = CompilationState()
    assert state.frontend is QueryLanguage.CYPHER
    assert state.frontend_ast is None


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
