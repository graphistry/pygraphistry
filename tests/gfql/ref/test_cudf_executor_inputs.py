import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward
from graphistry.compute.gfql.cudf_executor import (
    build_same_path_inputs,
    CuDFSamePathExecutor,
    execute_same_path_chain,
)
from graphistry.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull


def _make_graph():
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1", "score": 5},
            {"id": "acct2", "type": "account", "owner_id": "user2", "score": 9},
            {"id": "user1", "type": "user", "score": 7},
            {"id": "user2", "type": "user", "score": 3},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": "acct1", "dst": "user1"},
            {"src": "acct2", "dst": "user2"},
        ]
    )
    return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")


def test_build_inputs_collects_alias_metadata():
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user", "id": "user1"}, name="c"),
    ]
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


def test_forward_captures_alias_frames_and_prunes():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user", "id": "user1"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = CuDFSamePathExecutor(inputs)
    executor._forward()

    assert "a" in executor.alias_frames
    a_nodes = executor.alias_frames["a"]
    assert set(a_nodes.columns) == {"id", "owner_id"}
    assert list(a_nodes["id"]) == ["acct1"]


def test_forward_matches_oracle_tags_on_equality():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = CuDFSamePathExecutor(inputs)
    executor._forward()

    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert oracle.tags is not None
    assert set(executor.alias_frames["a"]["id"]) == oracle.tags["a"]
    assert set(executor.alias_frames["c"]["id"]) == oracle.tags["c"]


def test_run_materializes_oracle_sets():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

    result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )

    assert result._nodes is not None
    assert result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])


def test_forward_minmax_prune_matches_oracle():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "score"), "<", col("c", "score"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = CuDFSamePathExecutor(inputs)
    executor._forward()
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert oracle.tags is not None
    assert set(executor.alias_frames["a"]["id"]) == oracle.tags["a"]
    assert set(executor.alias_frames["c"]["id"]) == oracle.tags["c"]
