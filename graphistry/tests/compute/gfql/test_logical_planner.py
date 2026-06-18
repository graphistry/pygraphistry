from __future__ import annotations

from copy import deepcopy
from typing import List, Optional, Set, Type, TypeVar, cast

import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher.parser import parse_cypher
from graphistry.compute.gfql.frontends.cypher.binder import FrontendBinder
from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart, BoundVariable, SemanticTable
from graphistry.compute.gfql.ir.compilation import PlanContext
from graphistry.compute.gfql.ir.logical_plan import Filter, LogicalPlan, NodeScan, PatternMatch, Project, Unwind, iter_children
from graphistry.compute.gfql.ir.types import BoundPredicate, LogicalType, NodeRef, ScalarType
from graphistry.compute.gfql.logical_planner import LogicalPlanner

TPlan = TypeVar("TPlan", bound=LogicalPlan)


def _bind_query(query: str) -> BoundIR:
    return FrontendBinder().bind(parse_cypher(query), PlanContext())


def _var(name: str, logical_type: LogicalType, *, entity_kind: str) -> BoundVariable:
    return BoundVariable(
        name=name,
        logical_type=logical_type,
        nullable=False,
        null_extended_from=frozenset(),
        entity_kind=entity_kind,
    )


def _sample_bound_ir() -> BoundIR:
    return BoundIR(
        query_parts=[
            BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
            BoundQueryPart(
                clause="WHERE",
                inputs=frozenset({"n"}),
                outputs=frozenset({"n"}),
                predicates=[BoundPredicate(expression="n.id > 0"), BoundPredicate(expression="n.score > 10")],
            ),
            BoundQueryPart(clause="RETURN", outputs=frozenset({"n"})),
        ],
        semantic_table=SemanticTable(
            variables={"n": _var("n", NodeRef(labels=frozenset({"Person"})), entity_kind="node")}
        ),
    )


def _walk(plan: LogicalPlan) -> List[LogicalPlan]:
    stack = [plan]
    out: List[LogicalPlan] = []
    while stack:
        current = stack.pop()
        out.append(current)
        stack.extend(child for _slot, child in iter_children(current))
    return out


def _find_first(plan: LogicalPlan, target: Type[TPlan]) -> Optional[TPlan]:
    for node in _walk(plan):
        if isinstance(node, target):
            return cast(TPlan, node)
    return None


def test_logical_planner_root_ids_and_purity() -> None:
    bound_ir = _sample_bound_ir()
    before = deepcopy(bound_ir)
    planner = LogicalPlanner()

    first = planner.plan(bound_ir, PlanContext())
    second = planner.plan(bound_ir, PlanContext())
    op_ids = [node.op_id for node in _walk(first)]

    assert isinstance(first, Project)
    assert first == second
    assert bound_ir == before
    assert len(op_ids) >= 4
    assert all(op_id > 0 for op_id in op_ids)
    assert len(op_ids) == len(set(op_ids))


def test_logical_planner_keeps_typed_nodes_and_predicates_in_schema() -> None:
    root = LogicalPlanner().plan(_sample_bound_ir(), PlanContext())
    scan = _find_first(root, NodeScan)
    filters = [node for node in _walk(root) if isinstance(node, Filter)]

    assert scan is not None
    assert isinstance(scan.output_schema.columns["n"], NodeRef)
    assert [flt.predicate.expression for flt in filters] == ["n.score > 10", "n.id > 0"]
    assert all(isinstance(flt.output_schema.columns["n"], NodeRef) for flt in filters)


@pytest.mark.parametrize(
    ("query", "expected_optional", "expected_arm_id"),
    [
        ("OPTIONAL MATCH (n:Person) RETURN n", True, "top_level_optional_0"),
        ("MATCH (n:Person) OPTIONAL MATCH (n)-[:KNOWS]->(m:Person) RETURN n, m", True, "optional_arm_1"),
        ("MATCH (a:A) MATCH (a)-[:KNOWS]->(b:B) RETURN b", False, None),
    ],
)
def test_logical_planner_match_route_shapes(
    query: str,
    expected_optional: bool,
    expected_arm_id: Optional[str],
) -> None:
    root = LogicalPlanner().plan(_bind_query(query), PlanContext())
    pattern = _find_first(root, PatternMatch)

    assert pattern is not None
    assert pattern.optional is expected_optional
    assert pattern.arm_id == expected_arm_id
    if "MATCH (a:A) MATCH" in query:
        assert isinstance(pattern.input, NodeScan)
        assert pattern.pattern == {"aliases": ("a", "b")}


def test_logical_planner_projection_scope_shapes() -> None:
    root = LogicalPlanner().plan(
        _bind_query("MATCH (a) RETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b"),
        PlanContext(),
    )
    scan = _find_first(root, NodeScan)

    assert isinstance(root, Project)
    assert scan is not None
    assert isinstance(scan.output_schema.columns["a"], NodeRef)
    assert root.output_schema.columns == {"b": ScalarType(kind="unknown", nullable=False)}


@pytest.mark.parametrize(
    ("query", "expected_columns"),
    [
        ("MATCH (a:Person)-[r:KNOWS]->(b:Person) RETURN a, r, b", {"a", "r", "b"}),
        ("MATCH p = (n)-[r]->(b) RETURN count(r) AS cnt", {"b", "n", "r"}),
        ("MATCH path = shortestPath((a)-[*]-(b)) MATCH (b)-->(c) RETURN c", {"a", "b", "c"}),
    ],
)
def test_logical_planner_match_schema_shapes(query: str, expected_columns: Set[str]) -> None:
    root = LogicalPlanner().plan(_bind_query(query), PlanContext())
    pattern = _find_first(root, PatternMatch)

    assert pattern is not None
    assert set(pattern.output_schema.columns) == expected_columns


def test_logical_planner_unwind_contract() -> None:
    root = LogicalPlanner().plan(_bind_query("MATCH (n:Person) UNWIND [1, 2, 3] AS x RETURN n, x"), PlanContext())
    unwind = _find_first(root, Unwind)

    assert unwind is not None
    assert str(unwind.list_expr).replace(" ", "") == "[1,2,3]"
    assert unwind.variable == "x"


@pytest.mark.parametrize("allow_unknown", [False, True])
def test_logical_planner_unknown_alias_match_policy(allow_unknown: bool) -> None:
    bound_ir = BoundIR(
        query_parts=[BoundQueryPart(clause="MATCH", outputs=frozenset({"ghost"}))],
        semantic_table=SemanticTable(variables={}),
    )
    if not allow_unknown:
        with pytest.raises(GFQLValidationError, match="present in semantic scope"):
            LogicalPlanner().plan(bound_ir, PlanContext())
        return

    root = LogicalPlanner(allow_unknown_match_aliases=allow_unknown).plan(bound_ir, PlanContext())
    pattern = _find_first(root, PatternMatch)
    assert pattern is not None
    assert pattern.output_schema.columns == {}


@pytest.mark.parametrize(
    "bound_ir",
    [
        BoundIR(query_parts=[BoundQueryPart(clause="MERGE")]),
        BoundIR(
            query_parts=[
                BoundQueryPart(clause="MATCH", outputs=frozenset({"n"})),
                BoundQueryPart(clause="UNWIND", inputs=frozenset({"n"}), outputs=frozenset({"n"})),
            ],
            semantic_table=SemanticTable(
                variables={"n": _var("n", NodeRef(labels=frozenset({"Person"})), entity_kind="node")}
            ),
        ),
    ],
)
def test_logical_planner_rejects_unsupported_skeleton_shapes(bound_ir: BoundIR) -> None:
    with pytest.raises(GFQLValidationError):
        LogicalPlanner().plan(bound_ir, PlanContext())
