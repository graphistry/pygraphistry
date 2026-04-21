import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.ir.compilation import PhysicalPlan
from graphistry.compute.gfql.physical_planner import PhysicalPlanner, WavefrontExecutorWrapper


def _mk_graph():
    nodes = pd.DataFrame({"id": ["a", "b", "c"]})
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


def test_gfql_wavefront_optional_match_parity():
    g = _mk_graph()

    result = g.gfql(
        "MATCH (a)-->(b) OPTIONAL MATCH (b)-->(c) RETURN b.id AS bid, c.id AS cid ORDER BY bid, cid"
    )
    rows = result._nodes.reset_index(drop=True)

    assert rows["bid"].tolist() == ["b", "c"]
    assert rows["cid"].iloc[0] == "c"
    assert pd.isna(rows["cid"].iloc[1])


def test_gfql_call_compatibility_shim_when_physical_planner_not_covered():
    g = _mk_graph()

    result = g.gfql("CALL graphistry.degree() RETURN degree ORDER BY degree")

    assert result._nodes["degree"].tolist() == [1, 1, 2]


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
