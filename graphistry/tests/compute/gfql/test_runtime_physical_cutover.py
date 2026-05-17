from dataclasses import replace
from typing import cast

import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.execution_context import ExecutionContext
import graphistry.compute.gfql_unified as gfql_unified
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.lowering import CompiledCypherExecutionExtras, CompiledCypherQuery
from graphistry.compute.gfql.ir.compilation import PhysicalPlan
from graphistry.compute.gfql.ir.logical_plan import PatternMatch, iter_children
from graphistry.compute.gfql.physical_planner import (
    PhysicalPlanner,
    ProcedureCallExecutorWrapper,
    RowPipelineExecutorWrapper,
    SamePathExecutorWrapper,
    WavefrontExecutorWrapper,
)


def _mk_graph():
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "label__Missing": [False, False, False]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
    return graphistry.bind(source="s", destination="d", node="id").nodes(nodes).edges(edges)


def _has_optional_pattern_match(logical_plan):
    stack = [logical_plan]
    while stack:
        node = stack.pop()
        if isinstance(node, PatternMatch) and node.optional:
            return True
        stack.extend(child for _slot, child in iter_children(node))
    return False


def test_gfql_invokes_physical_planner_for_planned_route(monkeypatch):
    g = _mk_graph()
    planner_calls = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        planner_calls.append(type(logical_plan).__name__)
        return original_plan(self, logical_plan, ctx)

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("MATCH (n) RETURN n.id AS id ORDER BY id")

    assert planner_calls
    assert result._nodes["id"].tolist() == ["a", "b", "c"]


def test_gfql_runs_logical_pass_pipeline_before_physical_planner(monkeypatch):
    g = _mk_graph()
    calls = []
    original_pipeline = gfql_unified._run_logical_pass_pipeline
    original_plan = PhysicalPlanner.plan

    def _spy_pipeline(logical_plan, ctx):
        calls.append("passes")
        return original_pipeline(logical_plan, ctx)

    def _spy_plan(self, logical_plan, ctx):
        calls.append("physical")
        return original_plan(self, logical_plan, ctx)

    monkeypatch.setattr(gfql_unified, "_run_logical_pass_pipeline", _spy_pipeline)
    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    g.gfql("MATCH (n) RETURN n.id AS id ORDER BY id")

    assert calls[:2] == ["passes", "physical"]


def test_gfql_wavefront_optional_match_parity():
    g = _mk_graph()

    result = g.gfql(
        "MATCH (a)-->(b) OPTIONAL MATCH (b)-->(c) RETURN b.id AS bid, c.id AS cid ORDER BY bid, cid"
    )
    rows = result._nodes.reset_index(drop=True)

    assert rows["bid"].tolist() == ["b", "c"]
    assert rows["cid"].iloc[0] == "c"
    assert pd.isna(rows["cid"].iloc[1])


def test_call_route_uses_procedure_physical_route(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], ProcedureCallExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("CALL graphistry.degree() RETURN degree ORDER BY degree")

    assert planner_routes == ["procedure_call"]
    assert result._nodes["degree"].tolist() == [1, 1, 2]


def test_top_level_optional_match_matched_case_uses_same_path_physical_route(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("OPTIONAL MATCH (n) RETURN n.id AS id ORDER BY id")

    assert planner_routes == ["same_path"]
    assert result._nodes["id"].tolist() == ["a", "b", "c"]


def test_top_level_optional_match_unmatched_case_null_extends_via_same_path_route(monkeypatch):
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "label__Missing": [False, False, False]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
    g = graphistry.bind(source="s", destination="d", node="id").nodes(nodes).edges(edges)
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("OPTIONAL MATCH (n:Missing) RETURN n.id AS id")
    rows = result._nodes.reset_index(drop=True)

    assert planner_routes == ["same_path"]
    assert len(rows) == 1
    assert pd.isna(rows["id"].iloc[0])


def test_optional_reentry_route_uses_same_path_and_null_fills(monkeypatch):
    g = _mk_graph()
    optional_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        if _has_optional_pattern_match(logical_plan):
            optional_routes.append(physical_plan.route)
            assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql(
        "MATCH (a) WITH a, a.id AS aid "
        "OPTIONAL MATCH (a)-->(b) "
        "RETURN aid, b.id AS bid "
        "ORDER BY aid, bid"
    )
    rows = result._nodes.to_dict(orient="records")

    assert optional_routes == ["same_path"]
    assert rows[:2] == [{"aid": "a", "bid": "b"}, {"aid": "b", "bid": "c"}]
    assert len(rows) == 3
    assert rows[2]["aid"] == "c"
    assert pd.isna(rows[2]["bid"])


def test_optional_reentry_route_uses_same_path_when_all_rows_match(monkeypatch):
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "a"]})
    g = graphistry.bind(source="s", destination="d", node="id").nodes(nodes).edges(edges)
    optional_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        if _has_optional_pattern_match(logical_plan):
            optional_routes.append(physical_plan.route)
            assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql(
        "MATCH (a) WITH a "
        "OPTIONAL MATCH (a)-->(b) "
        "RETURN a.id AS aid, b.id AS bid "
        "ORDER BY aid, bid"
    )

    assert optional_routes == ["same_path"]
    assert result._nodes.to_dict(orient="records") == [
        {"aid": "a", "bid": "b"},
        {"aid": "b", "bid": "a"},
    ]


def test_same_path_route_uses_same_path_physical_route(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("MATCH (n) RETURN n.id AS id ORDER BY id")

    assert planner_routes == ["same_path"]
    assert result._nodes["id"].tolist() == ["a", "b", "c"]


def test_row_pipeline_route_uses_row_pipeline_physical_route(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], RowPipelineExecutorWrapper)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("RETURN 1 AS x")

    assert planner_routes == ["row_pipeline"]
    assert result._nodes["x"].tolist() == [1]


def test_distinct_projection_route_uses_physical_route(monkeypatch):
    nodes = pd.DataFrame({"id": ["a", "b", "c"], "group": ["x", "x", "y"]})
    edges = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})
    g = graphistry.bind(source="s", destination="d", node="id").nodes(nodes).edges(edges)
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        return physical_plan

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql("MATCH (n) RETURN DISTINCT n.group AS group ORDER BY group")

    assert planner_routes == ["same_path"]
    assert result._nodes["group"].tolist() == ["x", "y"]


@pytest.mark.parametrize(
    ("query", "defer_code", "expected_rows"),
    [
        (
            "MATCH () RETURN count(*) * 10 AS c",
            "anonymous_match",
            [{"c": 30}],
        ),
        (
            "MATCH (a) OPTIONAL MATCH (a)-->(b) RETURN count(b) AS c",
            "non_top_level_optional_match",
            [{"c": 2}],
        ),
        (
            "MATCH (a) MATCH (a)-->(b) RETURN count(b) AS c",
            "multiple_match_stages",
            [{"c": 2}],
        ),
        (
            "MATCH (a) RETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b",
            "scalar_projection_alias_match",
            [{"a": True, "b": True}, {"a": True, "b": True}, {"a": True, "b": True}],
        ),
    ],
)
def test_classified_unplanned_chain_fallback_uses_chain_route(
    monkeypatch,
    query,
    defer_code,
    expected_rows,
):
    g = _mk_graph()
    planner_calls = []
    compiled = cast(CompiledCypherQuery, compile_cypher(query, _warn_deprecated=False))
    assert compiled.logical_plan is None
    assert compiled.logical_plan_defer_code == defer_code
    assert compiled.logical_plan_defer_reason is not None

    def _spy_plan(self, logical_plan, ctx):
        planner_calls.append(type(logical_plan).__name__)
        return PhysicalPlan(route="unexpected", operators=(), logical_op_ids=(), metadata={})

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)

    result = g.gfql(query)

    assert planner_calls == []
    assert result._nodes.to_dict(orient="records") == expected_rows


def test_unplanned_chain_fallback_requires_compile_defer_reason():
    g = _mk_graph()
    compiled = cast(
        CompiledCypherQuery,
        compile_cypher("MATCH () RETURN count(*) * 10 AS c", _warn_deprecated=False),
    )
    assert compiled.logical_plan is None
    assert compiled.logical_plan_defer_code == "anonymous_match"
    assert compiled.logical_plan_defer_reason is not None

    unclassified = replace(compiled, execution_extras=None)

    with pytest.raises(GFQLValidationError, match="approved deferred chain route"):
        gfql_unified._execute_compiled_query_non_union(
            g,
            compiled_query=unclassified,
            engine="pandas",
            policy=None,
            context=ExecutionContext(),
        )


def test_unknown_deferred_logical_plan_reason_rejected_before_chain_fallback():
    g = _mk_graph()
    compiled = cast(
        CompiledCypherQuery,
        compile_cypher("MATCH () RETURN count(*) * 10 AS c", _warn_deprecated=False),
    )
    unknown_defer = replace(
        compiled,
        execution_extras=CompiledCypherExecutionExtras(
            logical_plan_defer_reason=(
                "LogicalPlanner skeleton requires MATCH aliases to be present in semantic scope"
            ),
            logical_plan_defer_code=None,
        ),
    )

    with pytest.raises(GFQLValidationError, match="approved deferred chain route"):
        gfql_unified._execute_compiled_query_non_union(
            g,
            compiled_query=unknown_defer,
            engine="pandas",
            policy=None,
            context=ExecutionContext(),
        )


def test_unapproved_deferred_logical_plan_code_rejected_before_chain_fallback():
    g = _mk_graph()
    compiled = cast(
        CompiledCypherQuery,
        compile_cypher("MATCH () RETURN count(*) * 10 AS c", _warn_deprecated=False),
    )
    unknown_defer = replace(
        compiled,
        execution_extras=CompiledCypherExecutionExtras(
            logical_plan_defer_reason="Synthetic unplanned route for regression coverage",
            logical_plan_defer_code="bogus_code",
        ),
    )

    with pytest.raises(GFQLValidationError, match="approved deferred chain route"):
        gfql_unified._execute_compiled_query_non_union(
            g,
            compiled_query=unknown_defer,
            engine="pandas",
            policy=None,
            context=ExecutionContext(),
        )


def test_wavefront_route_without_join_payload_raises(monkeypatch):
    g = _mk_graph()

    def _force_wavefront_plan(self, logical_plan, ctx):
        return PhysicalPlan(
            route="wavefront",
            operators=(WavefrontExecutorWrapper(),),
            logical_op_ids=(),
            metadata={},
        )

    monkeypatch.setattr(PhysicalPlanner, "plan", _force_wavefront_plan)

    with pytest.raises(GFQLValidationError, match="wavefront physical route selected"):
        g.gfql("MATCH (n) RETURN n.id AS id")


def test_connected_match_join_uses_physical_route(monkeypatch):
    nodes = pd.DataFrame({
        "id": ["p1", "p2", "c1"],
        "label__Person": [True, True, False],
        "label__Place": [False, False, True],
    })
    edges = pd.DataFrame({
        "s": ["p1", "p2"],
        "d": ["c1", "c1"],
        "type": ["IS_LOCATED_IN", "IS_LOCATED_IN"],
    })
    g = graphistry.bind(source="s", destination="d", node="id").nodes(nodes).edges(edges)

    planner_calls = []

    def _force_row_pipeline_plan(self, logical_plan, ctx):
        planner_calls.append(type(logical_plan).__name__)
        return PhysicalPlan(
            route="row_pipeline",
            operators=(RowPipelineExecutorWrapper(),),
            logical_op_ids=(),
            metadata={},
        )

    monkeypatch.setattr(PhysicalPlanner, "plan", _force_row_pipeline_plan)

    result = g.gfql(
        "MATCH "
        "(person:Person {id: 'p1'})-[:IS_LOCATED_IN]->(city:Place), "
        "(friend:Person)-[:IS_LOCATED_IN]->(city) "
        "RETURN city.id AS cityId, count(friend) AS friendCount"
    )

    assert planner_calls
    assert result._nodes.to_dict(orient="records") == [{"cityId": "c1", "friendCount": 2}]


def test_connected_optional_match_uses_physical_route(monkeypatch):
    g = _mk_graph()
    planner_calls = []

    def _force_row_pipeline_plan(self, logical_plan, ctx):
        planner_calls.append(type(logical_plan).__name__)
        return PhysicalPlan(
            route="row_pipeline",
            operators=(RowPipelineExecutorWrapper(),),
            logical_op_ids=(),
            metadata={},
        )

    monkeypatch.setattr(PhysicalPlanner, "plan", _force_row_pipeline_plan)

    result = g.gfql(
        "MATCH (a)-->(b) "
        "OPTIONAL MATCH (b)-->(c) "
        "RETURN b.id AS bid, c.id AS cid "
        "ORDER BY bid, cid"
    )
    rows = result._nodes.reset_index(drop=True)

    assert planner_calls
    assert rows["bid"].tolist() == ["b", "c"]
    assert rows["cid"].iloc[0] == "c"
    assert pd.isna(rows["cid"].iloc[1])
