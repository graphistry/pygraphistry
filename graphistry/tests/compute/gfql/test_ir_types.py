from dataclasses import FrozenInstanceError
from typing import FrozenSet, get_args, get_origin, get_type_hints

import pytest

from graphistry.compute.gfql.ir.bound_ir import (
    BoundIR,
    BoundVariable,
    ScopeFrame,
    SemanticTable,
)
from graphistry.compute.gfql.ir.types import (
    EdgeRef,
    NodeRef,
    PathType,
    QueryGraph,
    RelSpec,
    ScalarType,
)


def _assert_frozen_assignment(instance: object, field_name: str, value: object) -> None:
    with pytest.raises(FrozenInstanceError):
        setattr(instance, field_name, value)


def _mk_bound_variable(name: str = "n") -> BoundVariable:
    return BoundVariable(
        name=name,
        logical_type=NodeRef(labels=frozenset({"Person"})),
        nullable=False,
        null_extended_from=frozenset(),
        entity_kind="node",
    )


def test_ir_types_defaults_match_spec() -> None:
    assert NodeRef().labels == frozenset()
    assert EdgeRef().types == frozenset()

    rel_spec = RelSpec()
    assert rel_spec.types == frozenset()
    assert rel_spec.direction == "undirected"


def test_ir_types_are_frozen() -> None:
    _assert_frozen_assignment(NodeRef(), "labels", frozenset({"X"}))
    _assert_frozen_assignment(EdgeRef(), "types", frozenset({"R"}))
    _assert_frozen_assignment(RelSpec(), "direction", "forward")
    _assert_frozen_assignment(ScalarType(), "x", 1)
    _assert_frozen_assignment(PathType(), "x", 1)
    _assert_frozen_assignment(QueryGraph(), "x", 1)

    bound_variable = _mk_bound_variable("n")
    semantic_table = SemanticTable({"n": bound_variable})
    scope_frame = ScopeFrame(frozenset({"n"}), "match")

    _assert_frozen_assignment(bound_variable, "name", "m")
    _assert_frozen_assignment(semantic_table, "variables", {})
    _assert_frozen_assignment(scope_frame, "scope_kind", "with")


def test_bound_ir_types_are_importable_and_instantiable() -> None:
    bound_variable = BoundVariable(
        name="n",
        logical_type=NodeRef(labels=frozenset({"Person", "Admin"})),
        nullable=True,
        null_extended_from=frozenset({"opt_left", "opt_right"}),
        entity_kind="node",
    )

    semantic_table = SemanticTable(variables={"n": bound_variable})
    scope_frame = ScopeFrame(visible_vars=frozenset({"n"}), scope_kind="match")

    bound_ir = BoundIR(
        semantic_table=semantic_table,
        scope_stack=[scope_frame],
        query_graph=QueryGraph(),
    )

    assert bound_ir.semantic_table.variables["n"].null_extended_from == frozenset(
        {"opt_left", "opt_right"}
    )


def test_semantic_table_default_factory_is_not_shared() -> None:
    left = SemanticTable()
    right = SemanticTable()

    left.variables["n"] = _mk_bound_variable("n")

    assert "n" in left.variables
    assert "n" not in right.variables


def test_bound_ir_scope_stack_is_list_by_design() -> None:
    bound_ir = BoundIR(
        semantic_table=SemanticTable({"n": _mk_bound_variable("n")}),
        scope_stack=[],
        query_graph=QueryGraph(),
    )

    bound_ir.scope_stack.append(ScopeFrame(frozenset({"n"}), "match"))

    assert isinstance(bound_ir.scope_stack, list)
    assert len(bound_ir.scope_stack) == 1
    _assert_frozen_assignment(bound_ir, "scope_stack", [])


def test_bound_ir_type_hints_match_spec_contract() -> None:
    bound_variable_hints = get_type_hints(BoundVariable)
    scope_frame_hints = get_type_hints(ScopeFrame)
    bound_ir_hints = get_type_hints(BoundIR)
    rel_spec_hints = get_type_hints(RelSpec)

    assert bound_variable_hints["null_extended_from"] == FrozenSet[str]
    assert set(get_args(bound_variable_hints["entity_kind"])) == {
        "node",
        "edge",
        "scalar",
        "path",
    }
    assert set(get_args(scope_frame_hints["scope_kind"])) == {
        "match",
        "with",
        "union",
        "subquery",
    }
    assert "ast" not in bound_ir_hints
    assert get_origin(bound_ir_hints["scope_stack"]) is list
    assert get_args(bound_ir_hints["scope_stack"]) == (ScopeFrame,)
    assert set(get_args(rel_spec_hints["direction"])) == {
        "forward",
        "reverse",
        "undirected",
    }
