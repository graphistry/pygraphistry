import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError
import graphistry.compute.gfql_unified as gfql_unified
from graphistry.compute.gfql.ir.compilation import PhysicalPlan
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


def test_call_route_bypasses_compat_executor(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], ProcedureCallExecutorWrapper)
        return physical_plan

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("CALL physical route should not use the legacy compat executor")

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)
    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql("CALL graphistry.degree() RETURN degree ORDER BY degree")

    assert planner_routes == ["procedure_call"]
    assert result._nodes["degree"].tolist() == [1, 1, 2]


def test_top_level_optional_match_matched_case_bypasses_compat_executor(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("top-level OPTIONAL MATCH physical route should not use the legacy compat executor")

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)
    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql("OPTIONAL MATCH (n) RETURN n.id AS id ORDER BY id")

    assert planner_routes == ["same_path"]
    assert result._nodes["id"].tolist() == ["a", "b", "c"]


def test_top_level_optional_match_unmatched_case_null_extends_without_compat_executor(monkeypatch):
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

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("top-level OPTIONAL MATCH null-extension route should not use the legacy compat executor")

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)
    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql("OPTIONAL MATCH (n:Missing) RETURN n.id AS id")
    rows = result._nodes.reset_index(drop=True)

    assert planner_routes == ["same_path"]
    assert len(rows) == 1
    assert pd.isna(rows["id"].iloc[0])


def test_deferred_optional_reentry_shape_uses_documented_compat_fallback(monkeypatch):
    g = _mk_graph()
    fallback_reasons = []
    original_compat_executor = gfql_unified._execute_compiled_query_compat_non_union

    def _spy_compat_executor(*args, **kwargs):
        compiled_query = kwargs["compiled_query"]
        assert compiled_query.logical_plan is None
        fallback_reasons.append(compiled_query.logical_plan_defer_reason)
        return original_compat_executor(*args, **kwargs)

    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _spy_compat_executor)

    result = g.gfql("MATCH (a) WITH a OPTIONAL MATCH (a)-->(b) RETURN b")

    assert fallback_reasons
    assert "OPTIONAL MATCH" in fallback_reasons[0]
    assert result._nodes is not None


def test_same_path_route_bypasses_compat_executor(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], SamePathExecutorWrapper)
        return physical_plan

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("same_path physical route should not use the legacy compat executor")

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)
    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql("MATCH (n) RETURN n.id AS id ORDER BY id")

    assert planner_routes == ["same_path"]
    assert result._nodes["id"].tolist() == ["a", "b", "c"]


def test_row_pipeline_route_bypasses_compat_executor(monkeypatch):
    g = _mk_graph()
    planner_routes = []
    original_plan = PhysicalPlanner.plan

    def _spy_plan(self, logical_plan, ctx):
        physical_plan = original_plan(self, logical_plan, ctx)
        planner_routes.append(physical_plan.route)
        assert isinstance(physical_plan.operators[0], RowPipelineExecutorWrapper)
        return physical_plan

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("row_pipeline physical route should not use the legacy compat executor")

    monkeypatch.setattr(PhysicalPlanner, "plan", _spy_plan)
    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql("RETURN 1 AS x")

    assert planner_routes == ["row_pipeline"]
    assert result._nodes["x"].tolist() == [1]


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


def test_connected_match_join_bypasses_compat_executor(monkeypatch):
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

    def _force_row_pipeline_plan(self, logical_plan, ctx):
        return PhysicalPlan(
            route="row_pipeline",
            operators=(RowPipelineExecutorWrapper(),),
            logical_op_ids=(),
            metadata={},
        )

    monkeypatch.setattr(PhysicalPlanner, "plan", _force_row_pipeline_plan)

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("connected_match_join should not use the legacy compat executor")

    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql(
        "MATCH "
        "(person:Person {id: 'p1'})-[:IS_LOCATED_IN]->(city:Place), "
        "(friend:Person)-[:IS_LOCATED_IN]->(city) "
        "RETURN city.id AS cityId, count(friend) AS friendCount"
    )

    assert result._nodes.to_dict(orient="records") == [{"cityId": "c1", "friendCount": 2}]


def test_connected_optional_match_bypasses_compat_executor(monkeypatch):
    g = _mk_graph()

    def _force_row_pipeline_plan(self, logical_plan, ctx):
        return PhysicalPlan(
            route="row_pipeline",
            operators=(RowPipelineExecutorWrapper(),),
            logical_op_ids=(),
            metadata={},
        )

    monkeypatch.setattr(PhysicalPlanner, "plan", _force_row_pipeline_plan)

    def _fail_compat_executor(*args, **kwargs):
        raise AssertionError("connected_optional_match should not use the legacy compat executor")

    monkeypatch.setattr(gfql_unified, "_execute_compiled_query_compat_non_union", _fail_compat_executor)

    result = g.gfql(
        "MATCH (a)-->(b) "
        "OPTIONAL MATCH (b)-->(c) "
        "RETURN b.id AS bid, c.id AS cid "
        "ORDER BY bid, cid"
    )
    rows = result._nodes.reset_index(drop=True)

    assert rows["bid"].tolist() == ["b", "c"]
    assert rows["cid"].iloc[0] == "c"
    assert pd.isna(rows["cid"].iloc[1])
