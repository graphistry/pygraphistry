from dataclasses import fields, is_dataclass
from typing import Any, Set, Type, get_args, get_origin, get_type_hints


def _field_names(cls: Type[Any]) -> Set[str]:
    return {f.name for f in fields(cls)}


def test_m0b_required_ir_modules_are_importable() -> None:
    import graphistry.compute.gfql.ir.bound_ir as bound_ir  # noqa: F401
    import graphistry.compute.gfql.ir.capabilities as capabilities  # noqa: F401
    import graphistry.compute.gfql.ir.compilation as compilation  # noqa: F401
    import graphistry.compute.gfql.ir.logical_plan as logical_plan  # noqa: F401
    import graphistry.compute.gfql.ir.query_graph as query_graph  # noqa: F401
    import graphistry.compute.gfql.ir.types as types  # noqa: F401


def test_types_layer_matches_engine_neutral_contract() -> None:
    from graphistry.compute.gfql.ir import types
    from graphistry.compute.gfql.ir.types import EdgeRef, ListType, NodeRef, PathType, ScalarType

    assert is_dataclass(NodeRef)
    assert "CypherAST" not in vars(types)

    assert _field_names(EdgeRef) == {"type", "src_label", "dst_label"}
    assert _field_names(ScalarType) == {"kind", "nullable"}
    assert _field_names(PathType) == {"min_hops", "max_hops"}
    assert _field_names(ListType) == {"element_type"}


def test_bound_ir_shapes_match_m0b_contract() -> None:
    from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundVariable, ScopeFrame

    assert _field_names(BoundIR) == {"query_parts", "semantic_table", "scope_stack", "params"}

    scope_fields = _field_names(ScopeFrame)
    assert {"visible_vars", "schema", "origin_clause"} <= scope_fields

    variable_hints = get_type_hints(BoundVariable)
    assert "scope_id" in variable_hints
    assert set(get_args(variable_hints["entity_kind"])) == {"node", "edge", "scalar"}


def test_compilation_state_has_phase_fields_and_frontend_contract() -> None:
    from graphistry.compute.gfql.ir.compilation import CompilationState, GraphSchemaCatalog, PlanContext, QueryLanguage

    assert is_dataclass(CompilationState)
    assert getattr(CompilationState, "__dataclass_params__").frozen is False

    hints = get_type_hints(CompilationState)
    assert hints["frontend"] is QueryLanguage
    assert set(get_args(hints["frontend_ast"])) == {Any, type(None)}
    assert {"query_text", "ctx", "bound_ir", "semantic_table", "logical_plan", "query_graph", "physical_plan"} <= set(hints)

    plan_hints = get_type_hints(PlanContext)
    assert plan_hints["catalog"] is GraphSchemaCatalog


def test_query_graph_and_logical_plan_contract_bits() -> None:
    from graphistry.compute.gfql.ir.logical_plan import PatternMatch
    from graphistry.compute.gfql.ir.query_graph import QueryGraph

    assert "optional_arms" in _field_names(QueryGraph)

    hints = get_type_hints(PatternMatch)
    assert get_origin(hints["predicates"]) is list
    assert hints["optional"] is bool
    assert set(get_args(hints["arm_id"])) == {str, type(None)}
    assert "output_schema" in _field_names(PatternMatch)


def test_capability_registry_surface() -> None:
    from graphistry.compute.gfql.ir.capabilities import Decomposable, Monotonicity, OpCapability

    assert is_dataclass(OpCapability)
    assert Decomposable.NONE.name == "NONE"
    assert Monotonicity.UNKNOWN.name == "UNKNOWN"
    assert "decomposable" in _field_names(OpCapability)
