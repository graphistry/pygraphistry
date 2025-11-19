import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward
from graphistry.compute.gfql.cudf_executor import build_same_path_inputs
from graphistry.gfql.same_path_types import col, compare
from graphistry.tests.test_compute import CGFull


def _make_graph():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1"},
            {"id": "user1", "type": "user"},
        ]
    )
    edges = pd.DataFrame([{"src": "acct1", "dst": "user1"}])
    return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")


def test_build_inputs_collects_alias_metadata():
    chain = [n({"type": "account"}, name="a"), e_forward(name="r"), n(name="c")]
    where = [compare(col("a", "owner_id"), "==", col("c", "owner_id"))]
    graph = _make_graph()

    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)

    assert set(inputs.alias_bindings) == {"a", "r", "c"}
    assert inputs.column_requirements["a"] == {"owner_id"}
    assert inputs.column_requirements["c"] == {"owner_id"}
    assert inputs.plan.bitsets


def test_missing_alias_raises():
    chain = [n(name="a"), e_forward(name="r"), n(name="c")]
    where = [compare(col("missing", "x"), "==", col("c", "owner_id"))]
    graph = _make_graph()

    with pytest.raises(ValueError):
        build_same_path_inputs(graph, chain, where, Engine.PANDAS)
