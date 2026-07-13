"""
Tests for native igraph/cugraph shortest-path backends.

Structure:
- Unit tests for igraph_shortest_path_distances (skipped if igraph not installed)
- Unit tests for try_native_shortest_path fallback behavior
- Integration tests: shortestPath Cypher queries produce correct results
  via igraph, and match BFS fallback output
"""

from __future__ import annotations

import sys
import types
from typing import cast
from unittest.mock import patch

import pandas as pd
import pytest

from graphistry.compute.gfql.same_path.native_shortest_path import (
    cugraph_shortest_path_distances,
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
    @pytest.mark.parametrize(
        ("frm", "to", "sources", "targets", "kwargs", "expected"),
        [
            ([1, 2], [2, 3], [1], [2], {"directed": False}, {(1, 2): 1}),
            ([1, 2], [2, 3], [1], [3], {"directed": False}, {(1, 3): 2}),
            ([1], [2], [1], [3], {"directed": False}, {(1, 3): None}),
            ([1], [2], [1], [1], {"min_hops": 0, "directed": False}, {(1, 1): 0}),
            ([1], [2], [1], [1], {"min_hops": 1, "directed": False}, {(1, 1): None}),
            ([1, 1], [2, 1], [1], [1], {"min_hops": 1, "directed": False}, {(1, 1): 1}),
            ([1, 2], [2, 3], [1], [3], {"max_hops": 1, "directed": False}, {(1, 3): None}),
            ([1], [2], [2], [1], {"directed": False}, {(2, 1): 1}),
            ([1], [2], [2], [1], {"directed": True}, {(2, 1): None}),
            ([1, 2, 3], [2, 3, 1], [1, 1, 2], [2, 3, 3], {"directed": False}, {(1, 2): 1, (1, 3): 1, (2, 3): 1}),
            ([1], [2], [99], [2], {"directed": False}, {(99, 2): None}),
            ([1], [2], [1], [99], {"directed": False}, {(1, 99): None}),
        ],
    )
    def test_distance_cases(self, frm, to, sources, targets, kwargs, expected):
        call_kwargs = dict(kwargs)
        result = igraph_shortest_path_distances(
            _step_pairs(frm, to),
            sources,
            targets,
            max_hops=call_kwargs.pop("max_hops", None),
            **call_kwargs,
        )
        for (src, tgt), hops in expected.items():
            assert _hops(result, src, tgt) == hops

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

    def test_reuses_igraph_graph_state_for_same_cache_key(self):
        sp = _step_pairs([1, 2], [2, 3])
        cache = {}
        result1 = try_native_shortest_path(
            sp, [1], [3], max_hops=None, directed=False, engine=Engine.PANDAS, cache=cache, cache_key=("graph", 1)
        )
        result2 = try_native_shortest_path(
            sp, [2], [3], max_hops=None, directed=False, engine=Engine.PANDAS, cache=cache, cache_key=("graph", 1)
        )
        assert result1 is not None
        assert result2 is not None
        assert _hops(result1, 1, 3) == 2
        assert _hops(result2, 2, 3) == 1
        assert len(cache) == 1

    def test_returns_none_on_cudf_without_cugraph(self):
        # cugraph is not installed in test env; must return None gracefully
        sp = _step_pairs([1], [2])
        result = try_native_shortest_path(
            sp, [1], [2], max_hops=None, directed=False, engine=Engine.CUDF
        )
        assert result is None


# ---------------------------------------------------------------------------
# cugraph graph-cache wiring (CPU-testable via fake cudf/cugraph modules)
# ---------------------------------------------------------------------------

class _FakeArrow:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return self._values


class _FakeCudfSeries:
    def __init__(self, values):
        self._values = list(values)

    def unique(self):
        return _FakeCudfSeries(dict.fromkeys(self._values))

    def to_arrow(self):
        return _FakeArrow(self._values)


class _FakeCudfDataFrame:
    def __init__(self, data):
        self._data = {key: list(values) for key, values in data.items()}

    @property
    def columns(self):
        return list(self._data)

    def __getitem__(self, key):
        return _FakeCudfSeries(self._data[key])


class TestCugraphGraphCache:
    """The graph-build side of the cugraph backend is plain CPU logic; fake
    cudf/cugraph modules pin the build-once-per-cache-key contract without a GPU.
    Requests are empty so the BFS loop (real cugraph behavior) never runs."""

    def _fake_modules(self, build_log):
        cudf_mod = types.ModuleType("cudf")
        cudf_mod.DataFrame = _FakeCudfDataFrame  # type: ignore[attr-defined]
        cugraph_mod = types.ModuleType("cugraph")

        class _FakeGraph:
            def __init__(self, directed=False):
                self.directed = directed

            def from_cudf_edgelist(self, edges_gdf, source, destination):
                build_log.append((edges_gdf, source, destination, self.directed))

        cugraph_mod.Graph = _FakeGraph  # type: ignore[attr-defined]
        return {"cudf": cudf_mod, "cugraph": cugraph_mod}

    def test_same_cache_key_builds_graph_once(self):
        build_log: list = []
        sp = _step_pairs([1, 2], [2, 3])
        cache: dict = {}
        with patch.dict(sys.modules, self._fake_modules(build_log)):
            result1 = cugraph_shortest_path_distances(
                sp, [], [], max_hops=None, directed=True, cache=cache, cache_key="g1"
            )
            result2 = cugraph_shortest_path_distances(
                sp, [], [], max_hops=None, directed=True, cache=cache, cache_key="g1"
            )
        assert len(build_log) == 1
        assert list(cache) == [("cugraph", "g1", True)]
        # empty request -> empty result frame with the canonical schema
        assert result1.columns == ["__sp_source__", "__sp_target__", "__sp_hops__"]
        assert result2.columns == ["__sp_source__", "__sp_target__", "__sp_hops__"]

    def test_graph_built_from_step_pairs_with_directedness(self):
        build_log: list = []
        sp = _step_pairs([1, 2], [2, 3])
        with patch.dict(sys.modules, self._fake_modules(build_log)):
            cugraph_shortest_path_distances(
                sp, [], [], max_hops=None, directed=False, cache={}, cache_key="g1"
            )
        edges_gdf, source, destination, directed = build_log[0]
        assert edges_gdf._data == {"src": [1, 2], "dst": [2, 3]}
        assert (source, destination) == ("src", "dst")
        assert directed is False

    def test_no_cache_key_builds_graph_every_call(self):
        build_log: list = []
        sp = _step_pairs([1], [2])
        with patch.dict(sys.modules, self._fake_modules(build_log)):
            cugraph_shortest_path_distances(sp, [], [], max_hops=None, directed=True)
            cugraph_shortest_path_distances(sp, [], [], max_hops=None, directed=True)
        assert len(build_log) == 2

    def test_distinct_cache_keys_build_separate_graphs(self):
        build_log: list = []
        cache: dict = {}
        with patch.dict(sys.modules, self._fake_modules(build_log)):
            cugraph_shortest_path_distances(
                _step_pairs([1], [2]), [], [], max_hops=None, directed=True, cache=cache, cache_key="g1"
            )
            cugraph_shortest_path_distances(
                _step_pairs([3], [4]), [], [], max_hops=None, directed=True, cache=cache, cache_key="g2"
            )
        assert len(build_log) == 2
        assert set(cache) == {("cugraph", "g1", True), ("cugraph", "g2", True)}


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
