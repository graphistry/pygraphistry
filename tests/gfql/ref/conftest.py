"""Shared test fixtures and helpers for GFQL ref tests."""

import os
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
)
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull

# Environment variable to enable cudf parity testing (set in CI GPU tests)
TEST_CUDF = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"


def make_simple_graph():
    """Create a simple account->user graph for basic tests."""
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


def make_hop_graph():
    """Create a multi-hop graph for traversal tests."""
    nodes = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "u1", "score": 1},
            {"id": "user1", "type": "user", "owner_id": "u1", "score": 5},
            {"id": "user2", "type": "user", "owner_id": "u1", "score": 7},
            {"id": "acct2", "type": "account", "owner_id": "u1", "score": 9},
            {"id": "user3", "type": "user", "owner_id": "u3", "score": 2},
        ]
    )
    edges = pd.DataFrame(
        [
            {"src": "acct1", "dst": "user1"},
            {"src": "user1", "dst": "user2"},
            {"src": "user2", "dst": "acct2"},
            {"src": "acct1", "dst": "user3"},
        ]
    )
    return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")


def assert_executor_parity(graph, chain, where):
    """Assert executor parity with oracle. Tests pandas, and cudf if TEST_CUDF=1."""
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = DFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_native()
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=50, max_edges=50),
    )
    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"]), \
        f"pandas nodes mismatch: got {set(result._nodes['id'])}, expected {set(oracle.nodes['id'])}"
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])

    if not TEST_CUDF:
        return

    import cudf  # type: ignore

    cudf_nodes = cudf.DataFrame(graph._nodes)
    cudf_edges = cudf.DataFrame(graph._edges)
    cudf_graph = CGFull().nodes(cudf_nodes, graph._node).edges(cudf_edges, graph._source, graph._destination)

    cudf_inputs = build_same_path_inputs(cudf_graph, chain, where, Engine.CUDF)
    cudf_executor = DFSamePathExecutor(cudf_inputs)
    cudf_executor._forward()
    cudf_result = cudf_executor._run_native()

    assert cudf_result._nodes is not None and cudf_result._edges is not None
    assert set(cudf_result._nodes["id"].to_pandas()) == set(oracle.nodes["id"]), \
        f"cudf nodes mismatch: got {set(cudf_result._nodes['id'].to_pandas())}, expected {set(oracle.nodes['id'])}"
    assert set(cudf_result._edges["src"].to_pandas()) == set(oracle.edges["src"])
    assert set(cudf_result._edges["dst"].to_pandas()) == set(oracle.edges["dst"])


# Backwards compatibility aliases
_make_graph = make_simple_graph
_make_hop_graph = make_hop_graph
_assert_parity = assert_executor_parity
