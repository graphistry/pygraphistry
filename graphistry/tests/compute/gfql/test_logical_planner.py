from __future__ import annotations

from copy import deepcopy
from dataclasses import fields
from typing import List, Optional, Type, TypeVar, cast

import pytest
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, LogicalPlan, NodeScan, Project, Unwind
from graphistry.compute.gfql.ir.types import BoundPredicate, NodeRef, ScalarType
from graphistry.compute.gfql.logical_planner import IdGen, LogicalPlanner

TPlan = TypeVar("TPlan", bound=LogicalPlan)


def _bind_query(query: str) -> BoundIR:
    return FrontendBinder().bind(parse_cypher(query), PlanContext())


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


def _filters(plan: LogicalPlan) -> List[Filter]:
    return [node for node in _walk(plan) if isinstance(node, Filter)]


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


def test_logical_planner_match_label_is_stable_from_sorted_node_labels() -> None:
    bound_ir = BoundIR(
        query_parts=[BoundQueryPart(clause="MATCH", outputs=frozenset({"n"}))],
        semantic_table=SemanticTable(
            variables={
                "n": BoundVariable(
                    name="n",
                    logical_type=NodeRef(labels=frozenset({"Zeta", "Alpha"})),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                )
            }
        ),
    )
    root = LogicalPlanner().plan(bound_ir, PlanContext())
    assert isinstance(root, NodeScan)
    assert root.label == "Alpha"


def test_logical_planner_match_label_falls_back_for_non_node_ref_types() -> None:
    bound_ir = BoundIR(
        query_parts=[BoundQueryPart(clause="MATCH", outputs=frozenset({"n"}))],
        semantic_table=SemanticTable(
            variables={
                "n": BoundVariable(
                    name="n",
                    logical_type=ScalarType(kind="string", nullable=False),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                )
            }
        ),
    )
    root = LogicalPlanner().plan(bound_ir, PlanContext())
    assert isinstance(root, NodeScan)
    assert root.label == ""


def test_logical_planner_unwind_uses_binder_predicate_expression() -> None:
    query = "MATCH (n:Person) UNWIND [1, 2, 3] AS x RETURN n, x"
    bound = _bind_query(query)

    root = LogicalPlanner().plan(bound, PlanContext())
    unwind_node = _find_first(root, Unwind)

    assert unwind_node is not None
    assert str(unwind_node.list_expr).replace(" ", "") == "[1,2,3]"
    assert unwind_node.variable == "x"


def test_logical_planner_rejects_optional_match_shapes() -> None:
    query = "MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m:Person) RETURN n, m"
    bound = _bind_query(query)

    with pytest.raises(GFQLValidationError, match="OPTIONAL MATCH"):
        LogicalPlanner().plan(bound, PlanContext())


def test_logical_planner_rejects_multiple_match_stages() -> None:
    bound_ir = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"a"})),
            BoundQueryPart(clause="MATCH", outputs=frozenset({"a"})),
        ],
        semantic_table=SemanticTable(
            variables={
                "a": BoundVariable(
                    name="a",
                    logical_type=NodeRef(labels=frozenset({"Person"})),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="node",
                )
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="multiple MATCH"):
        LogicalPlanner().plan(bound_ir, PlanContext())


def test_logical_planner_applies_predicates_attached_to_match_part() -> None:
    query = "MATCH (n:Person) WHERE n.id > 0 RETURN n"
    bound = _bind_query(query)
    expected_predicate = bound.query_parts[0].predicates[0].expression

    root = LogicalPlanner().plan(bound, PlanContext())
    predicates = [flt.predicate.expression for flt in _filters(root)]

    assert expected_predicate in predicates


def test_logical_planner_applies_predicates_attached_to_with_part() -> None:
    query = "MATCH (n:Person) WITH n AS m WHERE m.id > 0 RETURN m"
    bound = _bind_query(query)
    with_part = next(part for part in bound.query_parts if part.clause == "WITH")
    expected_predicate = with_part.predicates[0].expression

    root = LogicalPlanner().plan(bound, PlanContext())
    predicates = [flt.predicate.expression for flt in _filters(root)]

    assert expected_predicate in predicates


def test_logical_planner_rejects_unknown_clause_types() -> None:
    bound_ir = BoundIR(query_parts=[BoundQueryPart(clause="MERGE")])
    with pytest.raises(GFQLValidationError, match="does not support this clause type"):
        LogicalPlanner().plan(bound_ir, PlanContext())


def test_logical_planner_rejects_multi_alias_match_shapes() -> None:
    query = "MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b"
    bound = _bind_query(query)

    with pytest.raises(GFQLValidationError, match="single-node MATCH"):
        LogicalPlanner().plan(bound, PlanContext())


def test_logical_planner_rejects_distinct_projection_shapes() -> None:
    query = "MATCH (n:Person) RETURN DISTINCT n"
    bound = _bind_query(query)

    with pytest.raises(GFQLValidationError, match="DISTINCT"):
        LogicalPlanner().plan(bound, PlanContext())


def test_logical_planner_rejects_single_alias_non_node_match_shapes() -> None:
    bound_ir = BoundIR(
        query_parts=[BoundQueryPart(clause="MATCH", outputs=frozenset({"r"}))],
        semantic_table=SemanticTable(
            variables={
                "r": BoundVariable(
                    name="r",
                    logical_type=ScalarType(kind="string", nullable=False),
                    nullable=False,
                    null_extended_from=frozenset(),
                    entity_kind="edge",
                )
            }
        ),
    )

    with pytest.raises(GFQLValidationError, match="node alias"):
        LogicalPlanner().plan(bound_ir, PlanContext())


def test_logical_planner_rejects_unwind_without_exactly_one_new_alias() -> None:
    base_vars = {
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
        "y": BoundVariable(
            name="y",
            logical_type=ScalarType(kind="int", nullable=False),
            nullable=False,
            null_extended_from=frozenset(),
            entity_kind="scalar",
        ),
    }

    no_new_alias = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="UNWIND",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n"}),
                metadata={"expression": "[1, 2, 3]"},
            ),
        ],
        semantic_table=SemanticTable(variables=base_vars),
    )
    with pytest.raises(GFQLValidationError, match="exactly one output alias"):
        LogicalPlanner().plan(no_new_alias, PlanContext())

    too_many_new_aliases = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="UNWIND",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n", "x", "y"}),
                metadata={"expression": "[1, 2, 3]"},
            ),
        ],
        semantic_table=SemanticTable(variables=base_vars),
    )
    with pytest.raises(GFQLValidationError, match="exactly one output alias"):
        LogicalPlanner().plan(too_many_new_aliases, PlanContext())


def test_logical_planner_rejects_unwind_without_list_expression() -> None:
    bound_ir = BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="UNWIND",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n", "x"}),
            ),
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

    with pytest.raises(GFQLValidationError, match="requires UNWIND list expression"):
        LogicalPlanner().plan(bound_ir, PlanContext())
