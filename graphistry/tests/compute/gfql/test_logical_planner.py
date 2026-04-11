from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import List, Optional, Type, TypeVar, cast

from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, LogicalPlan, NodeScan, Project, Unwind
from graphistry.compute.gfql.ir.types import BoundPredicate, NodeRef, ScalarType
from graphistry.compute.gfql.logical_planner import IdGen, LogicalPlanner

TPlan = TypeVar("TPlan", bound=LogicalPlan)


def _sample_bound_ir() -> BoundIR:
    return BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="WHERE",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n"}),
                predicates=[
                    BoundPredicate(expression="n.id > 0"),
                    BoundPredicate(expression="n.score > 10"),
                ],
            ),
            BoundQueryPart(clause="RETURN", outputs=frozenset({"n"})),
        ],
        semantic_table=SemanticTable(
            variables={
                "n": BoundVariable(
                    name="n",
                    logical_type=NodeRef(labels=frozenset({"Person"})),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                )
            }
        ),
    )


def _children(plan: LogicalPlan) -> List[LogicalPlan]:
    out: List[LogicalPlan] = []
    for field in fields(plan):
        value = getattr(plan, field.name)
        if isinstance(value, LogicalPlan):
            out.append(value)
    return out


def _walk(plan: LogicalPlan) -> List[LogicalPlan]:
    stack = [plan]
    out: List[LogicalPlan] = []
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(_children(current))
    return out


def _find_first(plan: LogicalPlan, target: Type[TPlan]) -> Optional[TPlan]:
    for node in _walk(plan):
        if isinstance(node, target):
            return cast(TPlan, node)
    return None


def test_logical_planner_importable_and_returns_logical_plan_root() -> None:
    root = LogicalPlanner().plan(_sample_bound_ir(), PlanContext())
    assert isinstance(root, LogicalPlan)
    assert isinstance(root, Project)


def test_logical_planner_assigns_unique_op_ids_per_plan() -> None:
    root = LogicalPlanner().plan(_sample_bound_ir(), PlanContext())
    op_ids = [node.op_id for node in _walk(root)]
    assert len(op_ids) >= 4
    assert all(op_id > 0 for op_id in op_ids)
    assert len(op_ids) == len(set(op_ids))


def test_logical_planner_repeated_planning_is_deterministic_and_pure() -> None:
    bound_ir = _sample_bound_ir()
    before = deepcopy(bound_ir)
    planner = LogicalPlanner()

    first = planner.plan(bound_ir, PlanContext())
    second = planner.plan(bound_ir, PlanContext())

    assert first == second
    assert bound_ir == before


def test_logical_planner_keeps_typed_nodes_in_output_schema() -> None:
    root = LogicalPlanner().plan(_sample_bound_ir(), PlanContext())
    scan = _find_first(root, NodeScan)
    filt = _find_first(root, Filter)
    assert scan is not None
    assert filt is not None
    assert isinstance(scan.output_schema.columns["n"], NodeRef)
    assert isinstance(filt.output_schema.columns["n"], NodeRef)


def test_logical_planner_empty_ir_returns_deterministic_fallback_project() -> None:
    root = LogicalPlanner().plan(BoundIR(), PlanContext())
    assert isinstance(root, Project)
    assert root.op_id == 1
    assert root.expressions == []
    assert root.output_schema.columns == {}


def test_logical_planner_unwind_sets_variable_and_expression() -> None:
    bound_ir = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="UNWIND",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n", "x"}),
                metadata={"expression": "[1, 2, 3]"},
            ),
            BoundQueryPart(clause="RETURN", outputs=frozenset({"n", "x"})),
        ],
        semantic_table=SemanticTable(
            variables={
                "n": BoundVariable(
                    name="n",
                    logical_type=NodeRef(labels=frozenset({"Person"})),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                ),
                "x": BoundVariable(
                    name="x",
                    logical_type=ScalarType(kind="int", nullable=False),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="scalar",
                ),
            }
        ),
    )
    root = LogicalPlanner().plan(bound_ir, PlanContext())
    assert isinstance(root, Project)
    unwind_node = _find_first(root, Unwind)
    assert unwind_node is not None
    assert unwind_node.list_expr == "[1, 2, 3]"
    assert unwind_node.variable == "x"


def test_id_gen_monotonic_sequence() -> None:
    id_gen = IdGen(start=7)
    assert [id_gen.next(), id_gen.next(), id_gen.next()] == [7, 8, 9]
