"""
Tests for native igraph/cugraph shortest-path backends.

Structure:
- Unit tests for igraph_shortest_path_distances (skipped if igraph not installed)
- Unit tests for try_native_shortest_path fallback behavior
- Integration tests: shortestPath Cypher queries produce correct results
  via igraph, and match BFS fallback output
"""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

from graphistry.compute.gfql.same_path.native_shortest_path import (
    igraph_shortest_path_distances,
    try_native_shortest_path,
)
from graphistry.Engine import Engine
from graphistry.tests.test_compute import CGFull

igraph = pytest.importorskip("igraph", reason="igraph not installed")


# ---------------------------------------------------------------------------
# Test graph helpers (mirrors pattern in test_lowering.py)
# ---------------------------------------------------------------------------

class _TestGraph(CGFull):
    _dgl_graph = None

    def search_graph(self, query, scale=0.5, top_n=100, thresh=5000, broader=False, inplace=False):
        raise NotImplementedError

    def search(self, query, cols=None, thresh=5000, fuzzy=True, top_n=10):
        raise NotImplementedError

    def embed(self, relation, *args, **kwargs):
        raise NotImplementedError


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _TestGraph:
    return cast(_TestGraph, _TestGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


def _chain_graph() -> _TestGraph:
    """Linear chain: 1—2—3—4"""
    return _mk_graph(
        pd.DataFrame({"id": [1, 2, 3, 4]}),
        pd.DataFrame({"s": [1, 2, 3], "d": [2, 3, 4]}),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step_pairs(frm, to):
    return pd.DataFrame({"__from__": frm, "__to__": to})


def _hops(result, src, tgt):
    """Return hop distance for (src, tgt), or None if unreachable/missing."""
    row = result[(result["__sp_source__"] == src) & (result["__sp_target__"] == tgt)]
    if row.empty:
        return None
    v = row.iloc[0]["__sp_hops__"]
    return None if pd.isna(v) else int(v)


# ---------------------------------------------------------------------------
# igraph unit tests
# ---------------------------------------------------------------------------

class TestIgraphShortestPathDistances:

    def test_direct_edge(self):
        sp = _step_pairs([1, 2], [2, 3])
        result = igraph_shortest_path_distances(sp, [1], [2], max_hops=None, directed=False)
        assert _hops(result, 1, 2) == 1

    def test_two_hops(self):
        sp = _step_pairs([1, 2], [2, 3])
        result = igraph_shortest_path_distances(sp, [1], [3], max_hops=None, directed=False)
        assert _hops(result, 1, 3) == 2

    def test_disconnected_returns_none(self):
        sp = _step_pairs([1], [2])
        result = igraph_shortest_path_distances(sp, [1], [3], max_hops=None, directed=False)
        assert _hops(result, 1, 3) is None

    def test_self_zero_hops(self):
        sp = _step_pairs([1], [2])
        result = igraph_shortest_path_distances(sp, [1], [1], max_hops=None, directed=False)
        assert _hops(result, 1, 1) == 0

    def test_max_hops_truncates(self):
        # 1—2—3: distance 2; max_hops=1 should return None
        sp = _step_pairs([1, 2], [2, 3])
        result = igraph_shortest_path_distances(sp, [1], [3], max_hops=1, directed=False)
        assert _hops(result, 1, 3) is None

    def test_directed_one_way(self):
        # Edge 1→2 only; reverse lookup should be unreachable when directed
        sp = _step_pairs([1], [2])
        undirected = igraph_shortest_path_distances(sp, [2], [1], max_hops=None, directed=False)
        directed = igraph_shortest_path_distances(sp, [2], [1], max_hops=None, directed=True)
        assert _hops(undirected, 2, 1) == 1
        assert _hops(directed, 2, 1) is None

    def test_multi_row_batched(self):
        # Triangle 1—2—3—1 (undirected)
        sp = _step_pairs([1, 2, 3], [2, 3, 1])
        result = igraph_shortest_path_distances(sp, [1, 1, 2], [2, 3, 3], max_hops=None, directed=False)
        assert _hops(result, 1, 2) == 1
        assert _hops(result, 1, 3) == 1  # direct via 3→1 undirected
        assert _hops(result, 2, 3) == 1

    def test_unknown_source_returns_none(self):
        sp = _step_pairs([1], [2])
        result = igraph_shortest_path_distances(sp, [99], [2], max_hops=None, directed=False)
        assert _hops(result, 99, 2) is None

    def test_unknown_target_returns_none(self):
        sp = _step_pairs([1], [2])
        result = igraph_shortest_path_distances(sp, [1], [99], max_hops=None, directed=False)
        assert _hops(result, 1, 99) is None

    def test_result_schema(self):
        sp = _step_pairs([1], [2])
        result = igraph_shortest_path_distances(sp, [1], [2], max_hops=None, directed=False)
        assert list(result.columns) == ["__sp_source__", "__sp_target__", "__sp_hops__"]
        assert len(result) == 1


# ---------------------------------------------------------------------------
# try_native_shortest_path wrapper tests
# ---------------------------------------------------------------------------

class TestTryNativeShortestPath:

    def test_returns_result_on_pandas_with_igraph(self):
        sp = _step_pairs([1, 2], [2, 3])
        result = try_native_shortest_path(
            sp, [1], [3], max_hops=None, directed=False, engine=Engine.PANDAS
        )
        assert result is not None
        assert _hops(result, 1, 3) == 2

    def test_returns_none_on_cudf_without_cugraph(self):
        # cugraph is not installed in test env; must return None gracefully
        sp = _step_pairs([1], [2])
        result = try_native_shortest_path(
            sp, [1], [2], max_hops=None, directed=False, engine=Engine.CUDF
        )
        assert result is None


# ---------------------------------------------------------------------------
# Integration: Cypher shortestPath via igraph backend
# ---------------------------------------------------------------------------

_SP_QUERY = (
    "MATCH (a:Person {id: $a}), (b:Person {id: $b}), "
    "path = shortestPath((a)-[:KNOWS*]-(b)) "
    "RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS dist"
)


def _mk_person_graph(node_ids, edges_src, edges_dst):
    nodes = pd.DataFrame({"id": node_ids, "label__Person": [True] * len(node_ids)})
    edges = pd.DataFrame({"s": edges_src, "d": edges_dst, "type": ["KNOWS"] * len(edges_src)})
    return _mk_graph(nodes, edges)


class TestCypherShortestPathViaIgraph:

    def _run(self, g, a, b):
        result = g.gfql(_SP_QUERY, params={"a": a, "b": b})
        return result._nodes["dist"].iloc[0]

    def test_connected_length(self):
        # Chain p1—p2—p3: distance 1-to-3 is 2
        g = _mk_person_graph(["p1", "p2", "p3"], ["p1", "p2"], ["p2", "p3"])
        assert self._run(g, "p1", "p3") == 2

    def test_adjacent(self):
        g = _mk_person_graph(["p1", "p2", "p3"], ["p1", "p2"], ["p2", "p3"])
        assert self._run(g, "p1", "p2") == 1

    def test_disconnected_returns_minus_one(self):
        g = _mk_person_graph(["p1", "p2", "p3"], ["p1"], ["p2"])
        assert self._run(g, "p1", "p3") == -1

    def test_self_path_zero(self):
        g = _mk_person_graph(["p1", "p2"], ["p1"], ["p2"])
        assert self._run(g, "p1", "p1") == 0

    def test_parity_with_bfs_fallback(self):
        """igraph result must match BFS for a multi-hop graph."""
        g = _mk_person_graph(
            ["p1", "p2", "p3", "p4", "p5"],
            ["p1", "p1", "p2", "p3"],
            ["p2", "p3", "p4", "p5"],
        )
        # p1→p3→p5 is distance 2
        native = self._run(g, "p1", "p5")

        with patch(
            "graphistry.compute.gfql.row.pipeline.RowPipelineMixin._gfql_shortest_path_scalar_native",
            return_value=None,
        ):
            bfs = self._run(g, "p1", "p5")

        assert native == bfs
