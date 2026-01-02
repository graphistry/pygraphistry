"""Core parity tests for df_executor - standalone tests and feature composition."""

import os
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
    execute_same_path_chain,
    _CUDF_MODE_ENV,
)
from graphistry.compute.gfql_unified import gfql
from graphistry.compute.chain import Chain
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull

# Import shared helpers - pytest auto-loads conftest.py
from tests.gfql.ref.conftest import (
    _make_graph,
    _make_hop_graph,
    _assert_parity,
    TEST_CUDF,
)

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
    executor = DFSamePathExecutor(inputs)
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
    executor = DFSamePathExecutor(inputs)
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
    executor = DFSamePathExecutor(inputs)
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


def test_strict_mode_without_cudf_raises(monkeypatch):
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    monkeypatch.setenv(_CUDF_MODE_ENV, "strict")
    inputs = build_same_path_inputs(graph, chain, where, Engine.CUDF)
    executor = DFSamePathExecutor(inputs)

    cudf_available = True
    try:
        import cudf  # type: ignore  # noqa: F401
    except Exception:
        cudf_available = False

    if cudf_available:
        # If cudf exists, strict mode should proceed to GPU path (currently routes to oracle)
        executor.run()
    else:
        with pytest.raises(RuntimeError):
            executor.run()


def test_auto_mode_without_cudf_falls_back(monkeypatch):
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    monkeypatch.setenv(_CUDF_MODE_ENV, "auto")
    inputs = build_same_path_inputs(graph, chain, where, Engine.CUDF)
    executor = DFSamePathExecutor(inputs)
    result = executor.run()
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )

    assert set(result._nodes["id"]) == set(oracle.nodes["id"])


def test_gpu_path_parity_equality():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = DFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_gpu()

    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])


def test_gpu_path_parity_inequality():
    graph = _make_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "score"), ">", col("c", "score"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = DFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_gpu()

    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])


@pytest.mark.parametrize(
    "edge_kwargs",
    [
        {"min_hops": 2, "max_hops": 3},
        {"min_hops": 1, "max_hops": 3, "output_min_hops": 3, "output_max_hops": 3},
    ],
    ids=["hop_range", "output_slice"],
)
def test_same_path_hop_params_parity(edge_kwargs):
    graph = _make_hop_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(**edge_kwargs),
        n(name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "owner_id"))]
    _assert_parity(graph, chain, where)


def test_same_path_hop_labels_propagate():
    graph = _make_hop_graph()
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(
            min_hops=1,
            max_hops=2,
            label_node_hops="node_hop",
            label_edge_hops="edge_hop",
            label_seeds=True,
        ),
        n(name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "owner_id"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = DFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_gpu()

    assert result._nodes is not None and result._edges is not None
    assert "node_hop" in result._nodes.columns
    assert "edge_hop" in result._edges.columns
    assert result._nodes["node_hop"].notna().any()
    assert result._edges["edge_hop"].notna().any()


def test_topology_parity_scenarios():
    scenarios = []

    nodes_cycle = pd.DataFrame(
        [
            {"id": "a1", "type": "account", "value": 1},
            {"id": "a2", "type": "account", "value": 3},
            {"id": "b1", "type": "user", "value": 5},
            {"id": "b2", "type": "user", "value": 2},
        ]
    )
    edges_cycle = pd.DataFrame(
        [
            {"src": "a1", "dst": "b1"},
            {"src": "a1", "dst": "b2"},  # branch
            {"src": "b1", "dst": "a2"},  # cycle back
        ]
    )
    chain_cycle = [
        n({"type": "account"}, name="a"),
        e_forward(name="r1"),
        n({"type": "user"}, name="b"),
        e_forward(name="r2"),
        n({"type": "account"}, name="c"),
    ]
    where_cycle = [compare(col("a", "value"), "<", col("c", "value"))]
    scenarios.append((nodes_cycle, edges_cycle, chain_cycle, where_cycle, None))

    nodes_mixed = pd.DataFrame(
        [
            {"id": "a1", "type": "account", "owner_id": "u1", "score": 2},
            {"id": "a2", "type": "account", "owner_id": "u2", "score": 7},
            {"id": "u1", "type": "user", "score": 9},
            {"id": "u2", "type": "user", "score": 1},
            {"id": "u3", "type": "user", "score": 5},
        ]
    )
    edges_mixed = pd.DataFrame(
        [
            {"src": "a1", "dst": "u1"},
            {"src": "a2", "dst": "u2"},
            {"src": "a2", "dst": "u3"},
        ]
    )
    chain_mixed = [
        n({"type": "account"}, name="a"),
        e_forward(name="r1"),
        n({"type": "user"}, name="b"),
        e_forward(name="r2"),
        n({"type": "account"}, name="c"),
    ]
    where_mixed = [
        compare(col("a", "owner_id"), "==", col("b", "id")),
        compare(col("b", "score"), ">", col("c", "score")),
    ]
    scenarios.append((nodes_mixed, edges_mixed, chain_mixed, where_mixed, None))

    nodes_edge_filter = pd.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1"},
            {"id": "acct2", "type": "account", "owner_id": "user2"},
            {"id": "user1", "type": "user"},
            {"id": "user2", "type": "user"},
            {"id": "user3", "type": "user"},
        ]
    )
    edges_edge_filter = pd.DataFrame(
        [
            {"src": "acct1", "dst": "user1", "etype": "owns"},
            {"src": "acct2", "dst": "user2", "etype": "owns"},
            {"src": "acct1", "dst": "user3", "etype": "follows"},
        ]
    )
    chain_edge_filter = [
        n({"type": "account"}, name="a"),
        e_forward({"etype": "owns"}, name="r"),
        n({"type": "user"}, name="c"),
    ]
    where_edge_filter = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    scenarios.append((nodes_edge_filter, edges_edge_filter, chain_edge_filter, where_edge_filter, {"dst": {"user1", "user2"}}))

    for nodes_df, edges_df, chain, where, edge_expect in scenarios:
        graph = CGFull().nodes(nodes_df, "id").edges(edges_df, "src", "dst")
        _assert_parity(graph, chain, where)
        if edge_expect:
            assert graph._edge is None or "etype" in edges_df.columns  # guard unused expectation
            result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
            assert result._edges is not None
            if "dst" in edge_expect:
                assert set(result._edges["dst"]) == edge_expect["dst"]


def test_cudf_gpu_path_if_available():
    cudf = pytest.importorskip("cudf")
    nodes = cudf.DataFrame(
        [
            {"id": "acct1", "type": "account", "owner_id": "user1", "score": 5},
            {"id": "acct2", "type": "account", "owner_id": "user2", "score": 9},
            {"id": "user1", "type": "user", "score": 7},
            {"id": "user2", "type": "user", "score": 3},
        ]
    )
    edges = cudf.DataFrame(
        [
            {"src": "acct1", "dst": "user1"},
            {"src": "acct2", "dst": "user2"},
        ]
    )
    graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]
    inputs = build_same_path_inputs(graph, chain, where, Engine.CUDF)
    executor = DFSamePathExecutor(inputs)
    result = executor.run()

    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"].to_pandas()) == {"acct1", "acct2"}
    assert set(result._edges["src"].to_pandas()) == {"acct1", "acct2"}


def test_dispatch_dict_where_triggers_executor():
    pytest.importorskip("cudf")
    graph = _make_graph()
    query = {
        "chain": [
            {"type": "Node", "name": "a", "filter_dict": {"type": "account"}},
            {"type": "Edge", "name": "r", "direction": "forward", "hops": 1},
            {"type": "Node", "name": "c", "filter_dict": {"type": "user"}},
        ],
        "where": [{"eq": {"left": "a.owner_id", "right": "c.id"}}],
    }
    result = gfql(graph, query, engine=Engine.CUDF)
    oracle = enumerate_chain(
        graph, [n({"type": "account"}, name="a"), e_forward(name="r"), n({"type": "user"}, name="c")],
        where=[compare(col("a", "owner_id"), "==", col("c", "id"))],
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])


def test_dispatch_chain_list_and_single_ast():
    graph = _make_graph()
    chain_ops = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

    for query in [Chain(chain_ops, where=where), chain_ops]:
        result = gfql(graph, query, engine=Engine.PANDAS)
        oracle = enumerate_chain(
            graph,
            chain_ops if isinstance(query, list) else list(chain_ops),
            where=where,
            include_paths=False,
            caps=OracleCaps(max_nodes=20, max_edges=20),
        )
        assert result._nodes is not None and result._edges is not None
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])
        assert set(result._edges["src"]) == set(oracle.edges["src"])
        assert set(result._edges["dst"]) == set(oracle.edges["dst"])


# ============================================================================
# Feature Composition Tests - Multi-hop + WHERE
# ============================================================================
#
# KNOWN LIMITATION: The cuDF same-path executor has architectural limitations
# with multi-hop edges combined with WHERE clauses:
#
# 1. Backward prune assumes single-hop edges where each edge step directly
#    connects adjacent node steps. Multi-hop edges break this assumption.
#
# 2. For multi-hop edges, _is_single_hop() gates WHERE clause filtering,
#    so WHERE between start/end of a multi-hop edge may not be applied
#    during backward prune.
#
# 3. The oracle correctly handles these cases, so oracle parity tests
#    catch the discrepancy.
#
# These tests are marked xfail to document the known limitations.
# See issue #871 for the testing roadmap.
# ============================================================================


class TestP0FeatureComposition:
    """
    Critical tests for hop ranges + WHERE clause composition.
    These catch subtle bugs in feature interactions.

    These tests are currently xfail due to known limitations in the
    cuDF executor's handling of multi-hop + WHERE combinations.
    """

    def test_where_respected_after_min_hops_backtracking(self):
        """
        P0 Test 1: WHERE must be respected after min_hops backtracking.

        Graph:
          a(v=1) -> b -> c -> d(v=10)   (3 hops, valid path)
          a(v=1) -> x -> y(v=0)         (2 hops, dead end for min=3)

        Chain: n(a) -[min_hops=2, max_hops=3]-> n(end)
        WHERE: a.value < end.value

        After backtracking prunes the x->y branch (doesn't reach 3 hops),
        WHERE should still filter: only paths where a.value < end.value.

        Risk: Backtracking may keep paths that violate WHERE.
        """
        nodes = pd.DataFrame([
            {"id": "a", "type": "start", "value": 5},
            {"id": "b", "type": "mid", "value": 3},
            {"id": "c", "type": "mid", "value": 7},
            {"id": "d", "type": "end", "value": 10},  # a.value(5) < d.value(10) ✓
            {"id": "x", "type": "mid", "value": 1},
            {"id": "y", "type": "end", "value": 2},   # a.value(5) < y.value(2) ✗
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "y"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"type": "start"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        _assert_parity(graph, chain, where)

        # Explicit check: y should NOT be in results (violates WHERE)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert result._nodes is not None
        result_ids = set(result._nodes["id"])
        # y violates WHERE (5 < 2 is false), should not be included
        assert "y" not in result_ids, "Node y violates WHERE but was included"
        # d satisfies WHERE (5 < 10 is true), should be included
        assert "d" in result_ids, "Node d satisfies WHERE but was excluded"

    def test_reverse_direction_where_semantics(self):
        """
        P0 Test 2: WHERE semantics must be consistent with reverse direction.

        Graph: a(v=1) -> b(v=5) -> c(v=3) -> d(v=9)

        Chain: n(name='start') -[e_reverse, min_hops=2]-> n(name='end')
        Starting at d, traversing backward.
        WHERE: start.value > end.value

        Reverse traversal from d:
        - hop 1: c (start=d, v=9)
        - hop 2: b (end=b, v=5) -> d.value(9) > b.value(5) ✓
        - hop 3: a (end=a, v=1) -> d.value(9) > a.value(1) ✓

        Risk: Direction swap could flip WHERE semantics.
        """
        nodes = pd.DataFrame([
            {"id": "a", "value": 1},
            {"id": "b", "value": 5},
            {"id": "c", "value": 3},
            {"id": "d", "value": 9},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "d"}, name="start"),
            e_reverse(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), ">", col("end", "value"))]

        _assert_parity(graph, chain, where)

        # Explicit check
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert result._nodes is not None
        result_ids = set(result._nodes["id"])
        # start is d (v=9), end can be b(v=5) or a(v=1)
        # Both satisfy 9 > 5 and 9 > 1
        assert "a" in result_ids or "b" in result_ids, "Valid endpoints excluded"
        # d is start, should be included
        assert "d" in result_ids, "Start node excluded"

    def test_non_adjacent_alias_where(self):
        """
        P0 Test 3: WHERE between non-adjacent aliases must be applied.

        Chain: n(name='a') -> e -> n(name='b') -> e -> n(name='c')
        WHERE: a.id == c.id  (aliases 2 edges apart)

        This tests cycles where we return to the starting node.

        Graph:
          x -> y -> x  (cycle)
          x -> y -> z  (no cycle)

        Only paths where a.id == c.id should be kept.

        Risk: cuDF backward prune only checks adjacent aliases.
        """
        nodes = pd.DataFrame([
            {"id": "x", "type": "node"},
            {"id": "y", "type": "node"},
            {"id": "z", "type": "node"},
        ])
        edges = pd.DataFrame([
            {"src": "x", "dst": "y"},
            {"src": "y", "dst": "x"},  # cycle back
            {"src": "y", "dst": "z"},  # no cycle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "id"), "==", col("c", "id"))]

        _assert_parity(graph, chain, where)

        # Explicit check: only x->y->x path satisfies a.id == c.id
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )

        # z should NOT be in results (x != z)
        assert "z" not in set(oracle.nodes["id"]), "z violates WHERE but oracle included it"
        if result._nodes is not None and not result._nodes.empty:
            assert "z" not in set(result._nodes["id"]), "z violates WHERE but executor included it"

    def test_non_adjacent_alias_where_inequality(self):
        """
        P0 Test 3b: Non-adjacent WHERE with inequality operators (<, >, <=, >=).

        Chain: n(name='a') -> e -> n(name='b') -> e -> n(name='c')
        WHERE: a.v < c.v  (aliases 2 edges apart, inequality)

        Graph with numeric values:
          n1(v=1) -> n2(v=5) -> n3(v=10)
          n1(v=1) -> n2(v=5) -> n4(v=3)

        Paths:
          n1 -> n2 -> n3: a.v=1 < c.v=10 (valid)
          n1 -> n2 -> n4: a.v=1 < c.v=3  (valid)

        All paths satisfy a.v < c.v.
        """
        nodes = pd.DataFrame([
            {"id": "n1", "v": 1},
            {"id": "n2", "v": 5},
            {"id": "n3", "v": 10},
            {"id": "n4", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "n1", "dst": "n2"},
            {"src": "n2", "dst": "n3"},
            {"src": "n2", "dst": "n4"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "v"), "<", col("c", "v"))]

        _assert_parity(graph, chain, where)

    def test_non_adjacent_alias_where_inequality_filters(self):
        """
        P0 Test 3c: Non-adjacent WHERE inequality that actually filters some paths.

        Chain: n(name='a') -> e -> n(name='b') -> e -> n(name='c')
        WHERE: a.v > c.v  (start value must be greater than end value)

        Graph:
          n1(v=10) -> n2(v=5) -> n3(v=1)   a.v=10 > c.v=1  (valid)
          n1(v=10) -> n2(v=5) -> n4(v=20)  a.v=10 > c.v=20 (invalid)

        Only paths where a.v > c.v should be kept.
        """
        nodes = pd.DataFrame([
            {"id": "n1", "v": 10},
            {"id": "n2", "v": 5},
            {"id": "n3", "v": 1},
            {"id": "n4", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "n1", "dst": "n2"},
            {"src": "n2", "dst": "n3"},
            {"src": "n2", "dst": "n4"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "v"), ">", col("c", "v"))]

        _assert_parity(graph, chain, where)

        # Explicit check: n4 should NOT be in results (10 > 20 is false)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )

        assert "n4" not in set(oracle.nodes["id"]), "n4 violates WHERE but oracle included it"
        if result._nodes is not None and not result._nodes.empty:
            assert "n4" not in set(result._nodes["id"]), "n4 violates WHERE but executor included it"
        # n3 should be included (10 > 1 is true)
        assert "n3" in set(oracle.nodes["id"]), "n3 satisfies WHERE but oracle excluded it"

    def test_non_adjacent_alias_where_not_equal(self):
        """
        P0 Test 3d: Non-adjacent WHERE with != operator.

        Chain: n(name='a') -> e -> n(name='b') -> e -> n(name='c')
        WHERE: a.id != c.id  (aliases must be different nodes)

        Graph:
          x -> y -> x  (cycle, a.id == c.id, should be excluded)
          x -> y -> z  (different, a.id != c.id, should be included)

        Only paths where a.id != c.id should be kept.
        """
        nodes = pd.DataFrame([
            {"id": "x", "type": "node"},
            {"id": "y", "type": "node"},
            {"id": "z", "type": "node"},
        ])
        edges = pd.DataFrame([
            {"src": "x", "dst": "y"},
            {"src": "y", "dst": "x"},  # cycle back
            {"src": "y", "dst": "z"},  # no cycle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "id"), "!=", col("c", "id"))]

        _assert_parity(graph, chain, where)

        # Explicit check: x->y->x path should be excluded (x == x)
        # x->y->z path should be included (x != z)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )

        # z should be in results (x != z)
        assert "z" in set(oracle.nodes["id"]), "z satisfies WHERE but oracle excluded it"
        if result._nodes is not None and not result._nodes.empty:
            assert "z" in set(result._nodes["id"]), "z satisfies WHERE but executor excluded it"

    def test_non_adjacent_alias_where_lte_gte(self):
        """
        P0 Test 3e: Non-adjacent WHERE with <= and >= operators.

        Chain: n(name='a') -> e -> n(name='b') -> e -> n(name='c')
        WHERE: a.v <= c.v  (start value must be <= end value)

        Graph:
          n1(v=5) -> n2(v=5) -> n3(v=5)   a.v=5 <= c.v=5  (valid, equal)
          n1(v=5) -> n2(v=5) -> n4(v=10)  a.v=5 <= c.v=10 (valid, less)
          n1(v=5) -> n2(v=5) -> n5(v=1)   a.v=5 <= c.v=1  (invalid)

        Only paths where a.v <= c.v should be kept.
        """
        nodes = pd.DataFrame([
            {"id": "n1", "v": 5},
            {"id": "n2", "v": 5},
            {"id": "n3", "v": 5},
            {"id": "n4", "v": 10},
            {"id": "n5", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "n1", "dst": "n2"},
            {"src": "n2", "dst": "n3"},
            {"src": "n2", "dst": "n4"},
            {"src": "n2", "dst": "n5"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "v"), "<=", col("c", "v"))]

        _assert_parity(graph, chain, where)

        # Explicit check
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )

        # n5 should NOT be in results (5 <= 1 is false)
        assert "n5" not in set(oracle.nodes["id"]), "n5 violates WHERE but oracle included it"
        if result._nodes is not None and not result._nodes.empty:
            assert "n5" not in set(result._nodes["id"]), "n5 violates WHERE but executor included it"
        # n3 and n4 should be included
        assert "n3" in set(oracle.nodes["id"]), "n3 satisfies WHERE but oracle excluded it"
        assert "n4" in set(oracle.nodes["id"]), "n4 satisfies WHERE but oracle excluded it"

    def test_non_adjacent_where_forward_forward(self):
        """
        P0 Test 3f: Non-adjacent WHERE with forward-forward topology (a->b->c).

        This is the base case already covered, but explicit for completeness.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 0},  # a->b->d where 1 > 0
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        # c (v=10) should be included (1 < 10), d (v=0) should be excluded (1 < 0 is false)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert "c" in set(result._nodes["id"]), "c satisfies WHERE but excluded"
        assert "d" not in set(result._nodes["id"]), "d violates WHERE but included"

    def test_non_adjacent_where_reverse_reverse(self):
        """
        P0 Test 3g: Non-adjacent WHERE with reverse-reverse topology (a<-b<-c).

        Graph edges: c->b->a (but we traverse in reverse)
        Chain: n(start) <-e- n(mid) <-e- n(end)
        Semantically: start is where we begin, end is where we finish traversing.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 0},
        ])
        # Edges go c->b->a, but we traverse backwards
        edges = pd.DataFrame([
            {"src": "c", "dst": "b"},
            {"src": "b", "dst": "a"},
            {"src": "d", "dst": "b"},  # d->b, so traversing reverse: b<-d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_reverse(),
            n(name="mid"),
            e_reverse(),
            n(name="end"),
        ]
        # start.v < end.v means the node we start at has smaller v than where we end
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_non_adjacent_where_forward_reverse(self):
        """
        P0 Test 3h: Non-adjacent WHERE with forward-reverse topology (a->b<-c).

        Graph: a->b and c->b (both point to b)
        Chain: n(start) -e-> n(mid) <-e- n(end)
        This finds paths where start reaches mid via forward, and end reaches mid via reverse.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 2},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b (forward from a)
            {"src": "c", "dst": "b"},  # c->b (reverse to reach c from b)
            {"src": "d", "dst": "b"},  # d->b (reverse to reach d from b)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_reverse(),
            n(name="end"),
        ]
        # start.v < end.v: 1 < 10 (a,c valid), 1 < 2 (a,d valid)
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"])
        # Both c and d should be reachable and satisfy the constraint
        assert "c" in result_nodes, "c satisfies WHERE but excluded"
        assert "d" in result_nodes, "d satisfies WHERE but excluded"

    def test_non_adjacent_where_reverse_forward(self):
        """
        P0 Test 3i: Non-adjacent WHERE with reverse-forward topology (a<-b->c).

        Graph: b->a, b->c, b->d (b points to all)
        Chain: n(start) <-e- n(mid) -e-> n(end)

        Valid paths with start.v < end.v:
          a(v=1) -> b -> c(v=10): 1 < 10 valid
          a(v=1) -> b -> d(v=0): 1 < 0 invalid (but d can still be start!)
          d(v=0) -> b -> a(v=1): 0 < 1 valid
          d(v=0) -> b -> c(v=10): 0 < 10 valid
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 0},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # b->a (reverse from a to reach b)
            {"src": "b", "dst": "c"},  # b->c (forward from b)
            {"src": "b", "dst": "d"},  # b->d (reverse from d to reach b)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_reverse(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        # start.v < end.v
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"])
        # All nodes participate in valid paths
        assert "a" in result_nodes, "a can be start (a->b->c) or end (d->b->a)"
        assert "c" in result_nodes, "c can be end for valid paths"
        assert "d" in result_nodes, "d can be start (d->b->a, d->b->c)"

    def test_non_adjacent_where_multihop_forward(self):
        """
        P0 Test 3j: Non-adjacent WHERE with multi-hop edge (a-[1..2]->b->c).

        Chain: n(start) -[hops 1-2]-> n(mid) -e-> n(end)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 3},
            {"id": "e", "v": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # 1 hop: a->b
            {"src": "b", "dst": "c"},  # 1 hop from b, or 2 hops from a
            {"src": "c", "dst": "d"},  # endpoint from c
            {"src": "c", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(min_hops=1, max_hops=2),  # Can reach b (1 hop) or c (2 hops)
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        # start.v < end.v
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_non_adjacent_where_multihop_reverse(self):
        """
        P0 Test 3k: Non-adjacent WHERE with multi-hop reverse edge.

        Chain: n(start) <-[hops 1-2]- n(mid) <-e- n(end)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        # Edges for reverse traversal
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reverse: a <- b
            {"src": "c", "dst": "b"},  # reverse: b <- c (2 hops from a)
            {"src": "d", "dst": "c"},  # reverse: c <- d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="mid"),
            e_reverse(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    # ===== Single-hop topology tests (direct a->c without middle node) =====

    def test_single_hop_forward_where(self):
        """
        P0 Test 4a: Single-hop forward topology (a->c).

        Chain: n(start) -e-> n(end), WHERE start.v < end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 0},  # d.v < all others
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_single_hop_reverse_where(self):
        """
        P0 Test 4b: Single-hop reverse topology (a<-c).

        Chain: n(start) <-e- n(end), WHERE start.v < end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reverse: a <- b
            {"src": "c", "dst": "b"},  # reverse: b <- c
            {"src": "c", "dst": "a"},  # reverse: a <- c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_reverse(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_single_hop_undirected_where(self):
        """
        P0 Test 4c: Single-hop undirected topology (a<->c).

        Chain: n(start) <-e-> n(end), WHERE start.v < end.v
        Tests both directions of each edge.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_undirected(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_single_hop_with_self_loop(self):
        """
        P0 Test 4d: Single-hop with self-loop (a->a).

        Tests that self-loops are handled correctly.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
            {"id": "c", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "b"},  # Self-loop
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        # start.v < end.v: self-loops fail (5 < 5 = false)
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_single_hop_equality_self_loop(self):
        """
        P0 Test 4e: Single-hop equality with self-loop.

        Self-loops satisfy start.v == end.v.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 5},  # Same value as a
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop: 5 == 5
            {"src": "a", "dst": "b"},  # a->b: 5 == 5
            {"src": "a", "dst": "c"},  # a->c: 5 != 10
            {"src": "b", "dst": "b"},  # Self-loop: 5 == 5
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        _assert_parity(graph, chain, where)

    # ===== Cycle topology tests =====

    def test_cycle_single_node(self):
        """
        P0 Test 5a: Self-loop cycle (a->a).

        Tests single-node cycles with WHERE clause.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "a"},  # Creates cycle a->b->a
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v < end.v
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_cycle_triangle(self):
        """
        P0 Test 5b: Triangle cycle (a->b->c->a).

        Tests cycles in multi-hop traversal.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},  # Completes the triangle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_cycle_with_branch(self):
        """
        P0 Test 5c: Cycle with branch (a->b->a and a->c).

        Tests cycles combined with branching topology.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "a"},  # Cycle back
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_oracle_cudf_parity_comprehensive(self):
        """
        P0 Test 4: Oracle and cuDF executor must produce identical results.

        Parametrized across multiple scenarios combining:
        - Different hop ranges
        - Different WHERE operators
        - Different graph topologies
        """
        scenarios = [
            # (nodes, edges, chain, where, description)
            (
                # Linear with inequality WHERE
                pd.DataFrame([
                    {"id": "a", "v": 1}, {"id": "b", "v": 5},
                    {"id": "c", "v": 3}, {"id": "d", "v": 9},
                ]),
                pd.DataFrame([
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                    {"src": "c", "dst": "d"},
                ]),
                # Note: Using explicit start filter - n(name="s") without filter
                # doesn't work with current executor (hop labels don't distinguish paths)
                [n({"id": "a"}, name="s"), e_forward(min_hops=2, max_hops=3), n(name="e")],
                [compare(col("s", "v"), "<", col("e", "v"))],
                "linear_inequality",
            ),
            (
                # Branch with equality WHERE
                pd.DataFrame([
                    {"id": "root", "owner": "u1"},
                    {"id": "left", "owner": "u1"},
                    {"id": "right", "owner": "u2"},
                    {"id": "leaf1", "owner": "u1"},
                    {"id": "leaf2", "owner": "u2"},
                ]),
                pd.DataFrame([
                    {"src": "root", "dst": "left"},
                    {"src": "root", "dst": "right"},
                    {"src": "left", "dst": "leaf1"},
                    {"src": "right", "dst": "leaf2"},
                ]),
                [n({"id": "root"}, name="a"), e_forward(min_hops=1, max_hops=2), n(name="c")],
                [compare(col("a", "owner"), "==", col("c", "owner"))],
                "branch_equality",
            ),
            (
                # Cycle with output slicing
                pd.DataFrame([
                    {"id": "n1", "v": 10},
                    {"id": "n2", "v": 20},
                    {"id": "n3", "v": 30},
                ]),
                pd.DataFrame([
                    {"src": "n1", "dst": "n2"},
                    {"src": "n2", "dst": "n3"},
                    {"src": "n3", "dst": "n1"},
                ]),
                [
                    n({"id": "n1"}, name="a"),
                    e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=3),
                    n(name="c"),
                ],
                [compare(col("a", "v"), "<", col("c", "v"))],
                "cycle_output_slice",
            ),
            (
                # Reverse with hop labels
                pd.DataFrame([
                    {"id": "a", "score": 100},
                    {"id": "b", "score": 50},
                    {"id": "c", "score": 75},
                ]),
                pd.DataFrame([
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                ]),
                [
                    n({"id": "c"}, name="start"),
                    e_reverse(min_hops=1, max_hops=2, label_node_hops="hop"),
                    n(name="end"),
                ],
                [compare(col("start", "score"), ">", col("end", "score"))],
                "reverse_labels",
            ),
        ]

        for nodes_df, edges_df, chain, where, desc in scenarios:
            graph = CGFull().nodes(nodes_df, "id").edges(edges_df, "src", "dst")
            inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
            executor = DFSamePathExecutor(inputs)
            executor._forward()
            result = executor._run_gpu()

            oracle = enumerate_chain(
                graph, chain, where=where, include_paths=False,
                caps=OracleCaps(max_nodes=50, max_edges=50),
            )

            assert result._nodes is not None, f"{desc}: result nodes is None"
            assert set(result._nodes["id"]) == set(oracle.nodes["id"]), \
                f"{desc}: node mismatch - executor={set(result._nodes['id'])}, oracle={set(oracle.nodes['id'])}"

            if result._edges is not None and not result._edges.empty:
                assert set(result._edges["src"]) == set(oracle.edges["src"]), \
                    f"{desc}: edge src mismatch"
                assert set(result._edges["dst"]) == set(oracle.edges["dst"]), \
                    f"{desc}: edge dst mismatch"


# ============================================================================
# P1 TESTS: High Confidence - Important but not blocking
# ============================================================================


class TestP1FeatureComposition:
    """
    Important tests for edge cases in feature composition.

    These tests are currently xfail due to known limitations in the
    cuDF executor's handling of multi-hop + WHERE combinations.
    """

    def test_multi_hop_edge_where_filtering(self):
        """
        P1 Test 5: WHERE must be applied even for multi-hop edges.

        The cuDF executor has `_is_single_hop()` check that may skip
        WHERE filtering for multi-hop edges.

        Graph: a(v=1) -> b(v=5) -> c(v=3) -> d(v=9)
        Chain: n(a) -[min_hops=2, max_hops=3]-> n(end)
        WHERE: a.value < end.value

        Risk: WHERE skipped for multi-hop edges.
        """
        nodes = pd.DataFrame([
            {"id": "a", "value": 5},
            {"id": "b", "value": 3},
            {"id": "c", "value": 7},
            {"id": "d", "value": 2},  # a.value(5) < d.value(2) is FALSE
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert result._nodes is not None
        result_ids = set(result._nodes["id"])
        # c satisfies 5 < 7, d does NOT satisfy 5 < 2
        assert "c" in result_ids, "c satisfies WHERE but excluded"
        # d should be excluded (5 < 2 is false)
        # But d might be included as intermediate - check oracle behavior
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_output_slicing_with_where(self):
        """
        P1 Test 6: Output slicing must interact correctly with WHERE.

        Graph: a(v=1) -> b(v=2) -> c(v=3) -> d(v=4)
        Chain: n(a) -[max_hops=3, output_min=2, output_max=2]-> n(end)
        WHERE: a.value < end.value

        Output slice keeps only hop 2 (node c).
        WHERE: a.value(1) < c.value(3) ✓

        Risk: Slicing applied before/after WHERE could give different results.
        """
        nodes = pd.DataFrame([
            {"id": "a", "value": 1},
            {"id": "b", "value": 2},
            {"id": "c", "value": 3},
            {"id": "d", "value": 4},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        _assert_parity(graph, chain, where)

    def test_label_seeds_with_output_min_hops(self):
        """
        P1 Test 7: label_seeds=True with output_min_hops > 0.

        Seeds are at hop 0, but output_min_hops=2 excludes hop 0.
        This is a potential conflict.

        Graph: seed -> b -> c -> d
        Chain: n(seed) -[output_min=2, label_seeds=True]-> n(end)
        """
        nodes = pd.DataFrame([
            {"id": "seed", "value": 1},
            {"id": "b", "value": 2},
            {"id": "c", "value": 3},
            {"id": "d", "value": 4},
        ])
        edges = pd.DataFrame([
            {"src": "seed", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "seed"}, name="start"),
            e_forward(
                min_hops=1,
                max_hops=3,
                output_min_hops=2,
                output_max_hops=3,
                label_node_hops="hop",
                label_seeds=True,
            ),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        _assert_parity(graph, chain, where)

    def test_multiple_where_mixed_hop_ranges(self):
        """
        P1 Test 8: Multiple WHERE clauses with different hop ranges per edge.

        Chain: n(a) -[hops=1]-> n(b) -[min_hops=1, max_hops=2]-> n(c)
        WHERE: a.v < b.v AND b.v < c.v

        Graph:
          a1(v=1) -> b1(v=5) -> c1(v=10)
          a1(v=1) -> b2(v=2) -> c2(v=3) -> c3(v=4)

        Both paths should satisfy the WHERE clauses.
        """
        nodes = pd.DataFrame([
            {"id": "a1", "type": "A", "v": 1},
            {"id": "b1", "type": "B", "v": 5},
            {"id": "b2", "type": "B", "v": 2},
            {"id": "c1", "type": "C", "v": 10},
            {"id": "c2", "type": "C", "v": 3},
            {"id": "c3", "type": "C", "v": 4},
        ])
        edges = pd.DataFrame([
            {"src": "a1", "dst": "b1"},
            {"src": "a1", "dst": "b2"},
            {"src": "b1", "dst": "c1"},
            {"src": "b2", "dst": "c2"},
            {"src": "c2", "dst": "c3"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"type": "A"}, name="a"),
            e_forward(name="e1"),
            n({"type": "B"}, name="b"),
            e_forward(min_hops=1, max_hops=2),  # No alias - oracle doesn't support edge aliases for multi-hop
            n({"type": "C"}, name="c"),
        ]
        where = [
            compare(col("a", "v"), "<", col("b", "v")),
            compare(col("b", "v"), "<", col("c", "v")),
        ]

        _assert_parity(graph, chain, where)


# ============================================================================
# UNFILTERED START TESTS - Known limitations of native Yannakakis path
# ============================================================================
#
# The native Yannakakis implementation (_run_native) has limitations with:
# - Unfiltered start nodes (n() with no predicates) combined with multi-hop
# - Complex path patterns where forward pass doesn't capture all valid starts
#
# These tests are marked xfail to document the limitation. The oracle path
# handles these correctly but is O(n!) and not suitable for production.
# TODO: Fix _run_native to handle unfiltered starts properly
# ============================================================================


class TestUnfilteredStarts:
    """
    Tests for unfiltered start nodes.

    The native path handles unfiltered start + multihop by using alias frames
    instead of hop labels (which become ambiguous when all nodes can be starts).
    """

    def test_unfiltered_start_node_multihop(self):
        """
        Unfiltered start node with multi-hop works via public API.

        Chain: n() -[min_hops=2, max_hops=3]-> n()
        WHERE: start.v < end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),  # No filter - all nodes can be start
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        # Use public API which handles this correctly
        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_unfiltered_start_single_hop(self):
        """
        Unfiltered start node with single-hop.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},  # Cycle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),  # No filter
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_unfiltered_start_with_cycle(self):
        """
        Unfiltered start with cycle in graph.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_unfiltered_start_multihop_reverse(self):
        """
        Unfiltered start node with multi-hop REVERSE traversal + WHERE.

        Tests the reverse direction code path with unfiltered starts.
        Chain: n() <-[min_hops=2, max_hops=2]- n()
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),  # No filter
            e_reverse(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_unfiltered_start_multihop_undirected(self):
        """
        Unfiltered start node with multi-hop UNDIRECTED traversal + WHERE.

        Tests undirected edges with unfiltered starts.
        Chain: n() -[undirected, min_hops=2, max_hops=2]- n()
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),  # No filter
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_filtered_start_multihop_reverse_where(self):
        """
        Filtered start node with multi-hop REVERSE + WHERE.

        Ensures hop labels work correctly for reverse direction.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "d"}, name="start"),  # Filtered to 'd'
            e_reverse(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])

    def test_filtered_start_multihop_undirected_where(self):
        """
        Filtered start with multi-hop UNDIRECTED + WHERE.

        Ensures hop labels work correctly for undirected edges.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),  # Filtered to 'a'
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        oracle = enumerate_chain(
            graph, chain, where=where, include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])


# ============================================================================
# ORACLE LIMITATIONS - These are actual oracle limitations, not executor bugs
# ============================================================================


class TestOracleLimitations:
    """
    Tests for oracle limitations (not executor bugs).

    These test features the oracle doesn't support.
    """

    @pytest.mark.xfail(
        reason="Oracle doesn't support edge aliases on multi-hop edges",
        strict=True,
    )
    def test_edge_alias_on_multihop(self):
        """
        ORACLE LIMITATION: Edge alias on multi-hop edge.

        The oracle raises an error when an edge alias is used on a multi-hop edge.
        This is documented in enumerator.py:109.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 1},
            {"src": "b", "dst": "c", "weight": 2},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2, name="e"),  # Edge alias on multi-hop
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        # Oracle raises error for edge alias on multi-hop
        _assert_parity(graph, chain, where)


# ============================================================================
# P0 ADDITIONAL TESTS: Reverse + Multi-hop
# ============================================================================


class TestP0ReverseMultihop:
    """
    P0 Tests: Reverse direction with multi-hop edges.

    These test combinations that revealed bugs during session 3.
    """

    def test_reverse_multihop_basic(self):
        """
        P0: Reverse multi-hop basic case.

        Chain: n(start) <-[min_hops=1, max_hops=2]- n(end)
        WHERE: start.v < end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        # For reverse traversal: edges point "forward" but we traverse backward
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reverse: a <- b
            {"src": "c", "dst": "b"},  # reverse: b <- c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"])
        # start=a(v=1), end can be b(v=5) or c(v=10)
        # Both satisfy 1 < 5 and 1 < 10
        assert "b" in result_ids, "b satisfies WHERE but excluded"
        assert "c" in result_ids, "c satisfies WHERE but excluded"

    def test_reverse_multihop_filters_correctly(self):
        """
        P0: Reverse multi-hop that actually filters some paths.

        Chain: n(start) <-[min_hops=1, max_hops=2]- n(end)
        WHERE: start.v > end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},  # start has high value
            {"id": "b", "v": 5},   # 10 > 5 valid
            {"id": "c", "v": 15},  # 10 > 15 invalid
            {"id": "d", "v": 1},   # 10 > 1 valid
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # a <- b
            {"src": "c", "dst": "b"},  # b <- c (so a <- b <- c)
            {"src": "d", "dst": "b"},  # b <- d (so a <- b <- d)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"])
        # c violates (10 > 15 is false), b and d satisfy
        assert "c" not in result_ids, "c violates WHERE but included"
        assert "b" in result_ids, "b satisfies WHERE but excluded"
        assert "d" in result_ids, "d satisfies WHERE but excluded"

    def test_reverse_multihop_with_cycle(self):
        """
        P0: Reverse multi-hop with cycle in graph.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # a <- b
            {"src": "c", "dst": "b"},  # b <- c
            {"src": "a", "dst": "c"},  # c <- a (creates cycle)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_reverse_multihop_undirected_comparison(self):
        """
        P0: Compare reverse multi-hop with equivalent undirected.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Reverse from c
        chain_rev = [
            n({"id": "c"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain_rev, where)


# ============================================================================
# P0 ADDITIONAL TESTS: Multiple Valid Starts
# ============================================================================


class TestP0MultipleStarts:
    """
    P0 Tests: Multiple valid start nodes (not all, not one).

    This tests the middle ground between single filtered start and all-as-starts.
    """

    def test_two_valid_starts(self):
        """
        P0: Two nodes match start filter.

        Graph:
          a1(v=1) -> b -> c(v=10)
          a2(v=2) -> b -> c(v=10)
        """
        nodes = pd.DataFrame([
            {"id": "a1", "type": "start", "v": 1},
            {"id": "a2", "type": "start", "v": 2},
            {"id": "b", "type": "mid", "v": 5},
            {"id": "c", "type": "end", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a1", "dst": "b"},
            {"src": "a2", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"type": "start"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_multiple_starts_different_paths(self):
        """
        P0: Multiple starts with different path outcomes.

        start1 -> path1 (satisfies WHERE)
        start2 -> path2 (violates WHERE)
        """
        nodes = pd.DataFrame([
            {"id": "s1", "type": "start", "v": 1},
            {"id": "s2", "type": "start", "v": 100},  # High value
            {"id": "m1", "type": "mid", "v": 5},
            {"id": "m2", "type": "mid", "v": 50},
            {"id": "e1", "type": "end", "v": 10},   # s1.v < e1.v (valid)
            {"id": "e2", "type": "end", "v": 60},   # s2.v > e2.v (invalid for <)
        ])
        edges = pd.DataFrame([
            {"src": "s1", "dst": "m1"},
            {"src": "m1", "dst": "e1"},
            {"src": "s2", "dst": "m2"},
            {"src": "m2", "dst": "e2"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"type": "start"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n({"type": "end"}, name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"])
        # s1->m1->e1 satisfies (1 < 10), s2->m2->e2 violates (100 < 60)
        assert "s1" in result_ids, "s1 satisfies WHERE but excluded"
        assert "e1" in result_ids, "e1 satisfies WHERE but excluded"
        # s2/e2 should be excluded
        assert "s2" not in result_ids, "s2 path violates WHERE but s2 included"
        assert "e2" not in result_ids, "e2 path violates WHERE but e2 included"

    def test_multiple_starts_shared_intermediate(self):
        """
        P0: Multiple starts sharing intermediate nodes.

        s1 -> shared -> end1
        s2 -> shared -> end2
        """
        nodes = pd.DataFrame([
            {"id": "s1", "type": "start", "v": 1},
            {"id": "s2", "type": "start", "v": 2},
            {"id": "shared", "type": "mid", "v": 5},
            {"id": "end1", "type": "end", "v": 10},
            {"id": "end2", "type": "end", "v": 0},  # s1.v > end2.v, s2.v > end2.v
        ])
        edges = pd.DataFrame([
            {"src": "s1", "dst": "shared"},
            {"src": "s2", "dst": "shared"},
            {"src": "shared", "dst": "end1"},
            {"src": "shared", "dst": "end2"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"type": "start"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n({"type": "end"}, name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


# ============================================================================
# ENTRYPOINT TESTS: Verify production paths use Yannakakis, NOT oracle
# ============================================================================


class TestProductionEntrypointsUseNative:
    """Verify g.gfql() and g.chain() with WHERE use native Yannakakis executor.

    These are "no-shit" tests - if they fail, production is either:
    1. Using the O(n!) oracle enumerator instead of vectorized Yannakakis
    2. Not using the same-path executor at all (skipping WHERE optimization)
    """

    def test_gfql_pandas_where_uses_yannakakis_executor(self, monkeypatch):
        """Production g.gfql() with pandas + WHERE must use Yannakakis executor."""
        native_called = False

        original_run_native = DFSamePathExecutor._run_native

        def spy_run_native(self):
            nonlocal native_called
            native_called = True
            return original_run_native(self)

        monkeypatch.setattr(DFSamePathExecutor, "_run_native", spy_run_native)

        graph = _make_graph()
        query = Chain(
            chain=[
                n({"type": "account"}, name="a"),
                e_forward(name="r"),
                n({"type": "user"}, name="c"),
            ],
            where=[compare(col("a", "owner_id"), "==", col("c", "id"))],
        )
        result = gfql(graph, query, engine="pandas")

        assert native_called, (
            "Production g.gfql(engine='pandas') with WHERE did not use Yannakakis executor! "
            "The same-path executor should be used for pandas+WHERE, not just cudf."
        )
        # Sanity check: result should have data
        assert result._nodes is not None
        assert len(result._nodes) > 0

    # NOTE: test_chain_pandas_where_uses_yannakakis_executor was removed because:
    # - chain() is deprecated (use gfql() instead)
    # - chain() never supported WHERE clauses - it extracts only ops.chain, discarding where
    # - Users should use gfql() for WHERE support, which is tested by test_gfql_pandas_where_uses_yannakakis_executor

    def test_executor_run_pandas_uses_native_not_oracle(self, monkeypatch):
        """DFSamePathExecutor.run() with pandas must use _run_native, not oracle."""
        oracle_called = False

        import graphistry.compute.gfql.df_executor as df_executor_module
        original_enumerate = df_executor_module.enumerate_chain

        def spy_enumerate(*args, **kwargs):
            nonlocal oracle_called
            oracle_called = True
            return original_enumerate(*args, **kwargs)

        monkeypatch.setattr(df_executor_module, "enumerate_chain", spy_enumerate)

        graph = _make_graph()
        chain = [
            n({"type": "account"}, name="a"),
            e_forward(name="r"),
            n({"type": "user"}, name="c"),
        ]
        where = [compare(col("a", "owner_id"), "==", col("c", "id"))]

        inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
        executor = DFSamePathExecutor(inputs)
        result = executor.run()  # This is the method that currently falls back to oracle!

        assert not oracle_called, (
            "DFSamePathExecutor.run() with Engine.PANDAS called oracle! "
            "Should use _run_native() for pandas too."
        )
        assert result._nodes is not None


# ============================================================================
# P1 TESTS: Operators × Single-hop Systematic
# ============================================================================


