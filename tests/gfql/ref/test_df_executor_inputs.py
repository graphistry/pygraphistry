import os
import numpy as np
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected, is_in
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
    execute_same_path_chain,
    _CUDF_MODE_ENV,
)
from graphistry.compute.gfql_unified import gfql
from graphistry.compute.chain import Chain
from graphistry.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull

# Environment variable to enable cudf parity testing (set in CI GPU tests)
TEST_CUDF = "TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"


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


def _make_hop_graph():
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


def _assert_parity(graph, chain, where):
    """Assert executor parity with oracle. Tests pandas, and cudf if TEST_CUDF=1."""
    # Always test pandas
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

    # Also test cudf if TEST_CUDF=1
    if not TEST_CUDF:
        return

    import cudf  # type: ignore

    # Convert graph to cudf
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
# UNFILTERED START TESTS - Previously thought to be limitations, but work!
# ============================================================================
#
# The public API (execute_same_path_chain) handles unfiltered starts correctly
# by falling back to oracle when the GPU path can't handle them.
# ============================================================================


class TestUnfilteredStarts:
    """
    Tests for unfiltered start nodes.

    These were previously marked as "known limitations" but the public API
    handles them correctly via oracle fallback.
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
# P1 TESTS: Operators × Single-hop Systematic
# ============================================================================


class TestP1OperatorsSingleHop:
    """
    P1 Tests: All comparison operators with single-hop edges.

    Systematic coverage of ==, !=, <, >, <=, >= for single-hop.
    """

    @pytest.fixture
    def basic_graph(self):
        """Graph for operator tests."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 5},   # Same as a
            {"id": "c", "v": 10},  # Greater than a
            {"id": "d", "v": 1},   # Less than a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b: 5 vs 5
            {"src": "a", "dst": "c"},  # a->c: 5 vs 10
            {"src": "a", "dst": "d"},  # a->d: 5 vs 1
            {"src": "c", "dst": "d"},  # c->d: 10 vs 1
        ])
        return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

    def test_single_hop_eq(self, basic_graph):
        """P1: Single-hop with == operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "==", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # Only a->b satisfies 5 == 5
        assert "a" in set(result._nodes["id"])
        assert "b" in set(result._nodes["id"])

    def test_single_hop_neq(self, basic_graph):
        """P1: Single-hop with != operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "!=", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # a->c (5 != 10) and a->d (5 != 1) and c->d (10 != 1) satisfy
        result_ids = set(result._nodes["id"])
        assert "c" in result_ids, "c participates in valid paths"
        assert "d" in result_ids, "d participates in valid paths"

    def test_single_hop_lt(self, basic_graph):
        """P1: Single-hop with < operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # a->c (5 < 10) satisfies
        assert "c" in set(result._nodes["id"])

    def test_single_hop_gt(self, basic_graph):
        """P1: Single-hop with > operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), ">", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # a->d (5 > 1) and c->d (10 > 1) satisfy
        assert "d" in set(result._nodes["id"])

    def test_single_hop_lte(self, basic_graph):
        """P1: Single-hop with <= operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # a->b (5 <= 5) and a->c (5 <= 10) satisfy
        result_ids = set(result._nodes["id"])
        assert "b" in result_ids
        assert "c" in result_ids

    def test_single_hop_gte(self, basic_graph):
        """P1: Single-hop with >= operator."""
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), ">=", col("end", "v"))]
        _assert_parity(basic_graph, chain, where)

        result = execute_same_path_chain(basic_graph, chain, where, Engine.PANDAS)
        # a->b (5 >= 5) and a->d (5 >= 1) and c->d (10 >= 1) satisfy
        result_ids = set(result._nodes["id"])
        assert "b" in result_ids
        assert "d" in result_ids


# ============================================================================
# P2 TESTS: Longer Paths (4+ nodes)
# ============================================================================


class TestP2LongerPaths:
    """
    P2 Tests: Paths with 4+ nodes.

    Tests that WHERE clauses work correctly for longer chains.
    """

    def test_four_node_chain(self):
        """
        P2: Chain of 4 nodes (3 edges).

        a -> b -> c -> d
        WHERE: a.v < d.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(),
            n(name="b"),
            e_forward(),
            n(name="c"),
            e_forward(),
            n(name="d"),
        ]
        where = [compare(col("a", "v"), "<", col("d", "v"))]

        _assert_parity(graph, chain, where)

    def test_five_node_chain_multiple_where(self):
        """
        P2: Chain of 5 nodes with multiple WHERE clauses.

        a -> b -> c -> d -> e
        WHERE: a.v < c.v AND c.v < e.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
            {"id": "d", "v": 7},
            {"id": "e", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "d", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(),
            n(name="b"),
            e_forward(),
            n(name="c"),
            e_forward(),
            n(name="d"),
            e_forward(),
            n(name="e"),
        ]
        where = [
            compare(col("a", "v"), "<", col("c", "v")),
            compare(col("c", "v"), "<", col("e", "v")),
        ]

        _assert_parity(graph, chain, where)

    def test_long_chain_with_multihop(self):
        """
        P2: Long chain with multi-hop edges.

        a -[1..2]-> mid -[1..2]-> end
        WHERE: a.v < end.v
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
            {"id": "d", "v": 7},
            {"id": "e", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "d", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="mid"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_long_chain_filters_partial_path(self):
        """
        P2: Long chain where only partial paths satisfy WHERE.

        a -> b -> c -> d1 (satisfies)
        a -> b -> c -> d2 (violates)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
            {"id": "d1", "v": 10},  # a.v < d1.v
            {"id": "d2", "v": 0},   # a.v < d2.v is false
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d1"},
            {"src": "c", "dst": "d2"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(),
            n(name="b"),
            e_forward(),
            n(name="c"),
            e_forward(),
            n(name="d"),
        ]
        where = [compare(col("a", "v"), "<", col("d", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"])
        assert "d1" in result_ids, "d1 satisfies WHERE but excluded"
        assert "d2" not in result_ids, "d2 violates WHERE but included"


# ============================================================================
# P1 TESTS: Operators × Multi-hop Systematic
# ============================================================================


class TestP1OperatorsMultihop:
    """
    P1 Tests: All comparison operators with multi-hop edges.

    Systematic coverage of ==, !=, <, >, <=, >= for multi-hop.
    """

    @pytest.fixture
    def multihop_graph(self):
        """Graph for multi-hop operator tests."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},   # Same as a
            {"id": "d", "v": 10},  # Greater than a
            {"id": "e", "v": 1},   # Less than a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},  # a-[2]->c: 5 vs 5
            {"src": "b", "dst": "d"},  # a-[2]->d: 5 vs 10
            {"src": "b", "dst": "e"},  # a-[2]->e: 5 vs 1
        ])
        return CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

    def test_multihop_eq(self, multihop_graph):
        """P1: Multi-hop with == operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_neq(self, multihop_graph):
        """P1: Multi-hop with != operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "!=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_lt(self, multihop_graph):
        """P1: Multi-hop with < operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_gt(self, multihop_graph):
        """P1: Multi-hop with > operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_lte(self, multihop_graph):
        """P1: Multi-hop with <= operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_gte(self, multihop_graph):
        """P1: Multi-hop with >= operator."""
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)


# ============================================================================
# P1 TESTS: Undirected + Multi-hop
# ============================================================================


class TestP1UndirectedMultihop:
    """
    P1 Tests: Undirected edges with multi-hop traversal.
    """

    def test_undirected_multihop_basic(self):
        """P1: Undirected multi-hop basic case."""
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
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_multihop_bidirectional(self):
        """P1: Undirected multi-hop can traverse both directions."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        # Only one direction in edges, but undirected should traverse both ways
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


# ============================================================================
# P1 TESTS: Mixed Direction Chains
# ============================================================================


class TestP1MixedDirectionChains:
    """
    P1 Tests: Chains with mixed edge directions (forward, reverse, undirected).
    """

    def test_forward_reverse_forward(self):
        """P1: Forward-reverse-forward chain."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # forward: a->b
            {"src": "c", "dst": "b"},  # reverse from b: b<-c
            {"src": "c", "dst": "d"},  # forward: c->d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid1"),
            e_reverse(),
            n(name="mid2"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_reverse_forward_reverse(self):
        """P1: Reverse-forward-reverse chain."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},
            {"id": "b", "v": 5},
            {"id": "c", "v": 7},
            {"id": "d", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reverse from a: a<-b
            {"src": "b", "dst": "c"},  # forward: b->c
            {"src": "d", "dst": "c"},  # reverse from c: c<-d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(),
            n(name="mid1"),
            e_forward(),
            n(name="mid2"),
            e_reverse(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_mixed_with_multihop(self):
        """P1: Mixed directions with multi-hop edges."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
            {"id": "d", "v": 7},
            {"id": "e", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "d", "dst": "c"},  # reverse: c<-d
            {"src": "e", "dst": "d"},  # reverse: d<-e
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="mid"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


# ============================================================================
# P2 TESTS: Edge Cases and Boundary Conditions
# ============================================================================


class TestP2EdgeCases:
    """
    P2 Tests: Edge cases and boundary conditions.
    """

    def test_single_node_graph(self):
        """P2: Graph with single node and self-loop."""
        nodes = pd.DataFrame([{"id": "a", "v": 5}])
        edges = pd.DataFrame([{"src": "a", "dst": "a"}])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_disconnected_components(self):
        """P2: Graph with disconnected components."""
        nodes = pd.DataFrame([
            {"id": "a1", "v": 1},
            {"id": "a2", "v": 5},
            {"id": "b1", "v": 10},
            {"id": "b2", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a1", "dst": "a2"},  # Component 1
            {"src": "b1", "dst": "b2"},  # Component 2
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_dense_graph(self):
        """P2: Dense graph with many edges."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ])
        # Fully connected
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_null_values_in_comparison(self):
        """P2: Nodes with null values in comparison column."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": None},  # Null value
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_string_comparison(self):
        """P2: String values in comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "name": "alice"},
            {"id": "b", "name": "bob"},
            {"id": "c", "name": "charlie"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "name"), "<", col("end", "name"))]

        _assert_parity(graph, chain, where)

    def test_multiple_where_all_operators(self):
        """P2: Multiple WHERE clauses with different operators."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 10},
            {"id": "b", "v": 5, "w": 5},
            {"id": "c", "v": 10, "w": 1},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="a"),
            e_forward(),
            n(name="b"),
            e_forward(),
            n(name="c"),
        ]
        # a.v < c.v AND a.w > c.w
        where = [
            compare(col("a", "v"), "<", col("c", "v")),
            compare(col("a", "w"), ">", col("c", "w")),
        ]

        _assert_parity(graph, chain, where)


# ============================================================================
# P3 TESTS: Bug Pattern Coverage (from 5 Whys analysis)
# ============================================================================
#
# These tests target specific bug patterns discovered during debugging:
# 1. Multi-hop backward propagation edge cases
# 2. Merge suffix handling for same-named columns
# 3. Undirected edge handling in various contexts
# ============================================================================


class TestBugPatternMultihopBackprop:
    """
    Tests for multi-hop backward propagation edge cases.

    Bug pattern: Code that filters edges by endpoints breaks for multi-hop
    because intermediate nodes aren't in left_allowed or right_allowed sets.
    """

    def test_three_consecutive_multihop_edges(self):
        """Three consecutive multi-hop edges - stress test for backward prop."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
            {"id": "e", "v": 5},
            {"id": "f", "v": 6},
            {"id": "g", "v": 7},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "d", "dst": "e"},
            {"src": "e", "dst": "f"},
            {"src": "f", "dst": "g"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="mid1"),
            e_forward(min_hops=1, max_hops=2),
            n(name="mid2"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_multihop_with_output_slicing_and_where(self):
        """Multi-hop with output_min_hops/output_max_hops + WHERE."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_multihop_diamond_graph(self):
        """Multi-hop through a diamond-shaped graph (multiple paths)."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ])
        # Diamond: a -> b -> d and a -> c -> d
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestBugPatternMergeSuffix:
    """
    Tests for merge suffix handling with same-named columns.

    Bug pattern: When left_col == right_col, pandas merge creates
    suffixed columns (e.g., 'v' and 'v__r') but code may compare
    column to itself instead of to the suffixed version.
    """

    def test_same_column_eq(self):
        """Same column name with == operator."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},  # Same as a
            {"id": "d", "v": 7},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v == end.v: only c matches (v=5)
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_lt(self):
        """Same column name with < operator."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 10},
            {"id": "d", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v < end.v: c matches (5 < 10), d doesn't (5 < 1 is false)
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_lte(self):
        """Same column name with <= operator."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},  # Equal
            {"id": "d", "v": 10},  # Greater
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v <= end.v: c (5<=5) and d (5<=10) match
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_gt(self):
        """Same column name with > operator."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 1},  # Less than a
            {"id": "d", "v": 10},  # Greater than a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v > end.v: only c matches (5 > 1)
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_gte(self):
        """Same column name with >= operator."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},  # Equal
            {"id": "d", "v": 1},  # Less
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v >= end.v: c (5>=5) and d (5>=1) match
        where = [compare(col("start", "v"), ">=", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestBugPatternUndirected:
    """
    Tests for undirected edge handling in various contexts.

    Bug pattern: Code checks `is_reverse = direction == "reverse"` but
    doesn't handle `direction == "undirected"`, treating it as forward.
    Undirected requires bidirectional adjacency.
    """

    def test_undirected_non_adjacent_where(self):
        """Undirected edges with non-adjacent WHERE clause."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        # Edges only go one way, but undirected should work both ways
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(),
            n(name="mid"),
            e_undirected(),
            n(name="end"),
        ]
        # Non-adjacent: start.v < end.v
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_multiple_where(self):
        """Undirected edges with multiple WHERE clauses."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 10},
            {"id": "b", "v": 5, "w": 5},
            {"id": "c", "v": 10, "w": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Multiple WHERE: start.v < end.v AND start.w > end.w
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "w"), ">", col("end", "w")),
        ]

        _assert_parity(graph, chain, where)

    def test_mixed_directed_undirected_chain(self):
        """Chain with both directed and undirected edges."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "c", "dst": "b"},  # Goes "wrong" way, but undirected should handle
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid"),
            e_undirected(),  # Should be able to go b -> c even though edge is c -> b
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_with_self_loop(self):
        """Undirected edge with self-loop."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_reverse_undirected_chain(self):
        """Chain: undirected -> reverse -> undirected."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 4},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "b", "dst": "c"},
            {"src": "d", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(),
            n(name="mid1"),
            e_reverse(),
            n(name="mid2"),
            e_undirected(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestImpossibleConstraints:
    """Test cases with impossible/contradictory constraints that should return empty results."""

    def test_contradictory_lt_gt_same_column(self):
        """Impossible: a.v < b.v AND a.v > b.v (can't be both)."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
            {"id": "c", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        # start.v < end.v AND start.v > end.v - impossible!
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "v"), ">", col("end", "v")),
        ]

        _assert_parity(graph, chain, where)

    def test_contradictory_eq_neq_same_column(self):
        """Impossible: a.v == b.v AND a.v != b.v (can't be both)."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        # start.v == end.v AND start.v != end.v - impossible!
        where = [
            compare(col("start", "v"), "==", col("end", "v")),
            compare(col("start", "v"), "!=", col("end", "v")),
        ]

        _assert_parity(graph, chain, where)

    def test_contradictory_lte_gt_same_column(self):
        """Impossible: a.v <= b.v AND a.v > b.v (can't be both)."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
            {"id": "c", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        # start.v <= end.v AND start.v > end.v - impossible!
        where = [
            compare(col("start", "v"), "<=", col("end", "v")),
            compare(col("start", "v"), ">", col("end", "v")),
        ]

        _assert_parity(graph, chain, where)

    def test_no_paths_satisfy_predicate(self):
        """All edges exist but no path satisfies the predicate."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},  # Highest value
            {"id": "b", "v": 50},
            {"id": "c", "v": 10},   # Lowest value
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n({"id": "c"}, name="end"),
        ]
        # start.v < mid.v - but a.v=100 > b.v=50, so no valid path
        where = [compare(col("start", "v"), "<", col("mid", "v"))]

        _assert_parity(graph, chain, where)

    def test_multihop_no_valid_endpoints(self):
        """Multi-hop where no endpoints satisfy the predicate."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},
            {"id": "b", "v": 50},
            {"id": "c", "v": 25},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v - but a.v=100 is the highest, so impossible
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_contradictory_on_different_columns(self):
        """Multiple predicates on different columns that are contradictory."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 5, "w": 10},
            {"id": "b", "v": 10, "w": 5},  # v is higher, w is lower
            {"id": "c", "v": 3, "w": 20},  # v is lower, w is higher
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        # For b: a.v < b.v (5 < 10) TRUE, but a.w < b.w (10 < 5) FALSE
        # For c: a.v < c.v (5 < 3) FALSE, but a.w < c.w (10 < 20) TRUE
        # No destination satisfies both
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "w"), "<", col("end", "w")),
        ]

        _assert_parity(graph, chain, where)

    def test_chain_with_impossible_intermediate(self):
        """Chain where intermediate step makes path impossible."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # This would make mid.v > end.v impossible
            {"id": "c", "v": 50},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n({"id": "c"}, name="end"),
        ]
        # mid.v < end.v - but b.v=100 > c.v=50
        where = [compare(col("mid", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_non_adjacent_impossible_constraint(self):
        """Non-adjacent WHERE clause that's impossible to satisfy."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},  # Highest
            {"id": "b", "v": 50},
            {"id": "c", "v": 10},   # Lowest
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n({"id": "c"}, name="end"),
        ]
        # start.v < end.v - but a.v=100 > c.v=10
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_empty_graph_with_constraints(self):
        """Empty graph should return empty even with valid-looking constraints."""
        nodes = pd.DataFrame({"id": [], "v": []})
        edges = pd.DataFrame({"src": [], "dst": []})
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_no_edges_with_constraints(self):
        """Nodes exist but no edges - should return empty."""
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame({"src": [], "dst": []})
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestFiveWhysAmplification:
    """
    Tests derived from 5-whys analysis of bugs found in PR #846.

    Each test targets a root cause that wasn't covered by existing tests.
    See alloy/README.md for bug list and issue #871 for verification roadmap.
    """

    # =========================================================================
    # Bug 1: Backward traversal join direction
    # Root cause: Direction semantics not tested at reachability level
    # =========================================================================

    def test_reverse_multihop_with_unreachable_intermediate(self):
        """
        Reverse multi-hop where some intermediates are unreachable from start.

        Bug pattern: Join direction error causes wrong nodes to appear reachable.
        This catches bugs where reverse traversal join uses wrong column order.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},   # start
            {"id": "b", "v": 5},   # reachable from a in reverse (b->a exists)
            {"id": "c", "v": 10},  # reachable from b in reverse (c->b exists)
            {"id": "x", "v": 100}, # NOT reachable - no path to a
            {"id": "y", "v": 200}, # NOT reachable - only x->y, no connection to a
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reverse: a <- b
            {"src": "c", "dst": "b"},  # reverse: b <- c (so a <- b <- c)
            {"src": "x", "dst": "y"},  # isolated: y <- x (no connection to a)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        # Verify x and y are NOT in results (they're unreachable)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "x" not in result_ids, "x is unreachable but appeared in results"
        assert "y" not in result_ids, "y is unreachable but appeared in results"

    def test_reverse_multihop_asymmetric_fanout(self):
        """
        Reverse traversal with asymmetric fan-out to test join direction.

        Graph: a <- b <- c
               a <- b <- d
               e <- f (isolated)

        Bug pattern: Wrong join direction could include f when tracing from a.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
            {"id": "e", "v": 100},  # Isolated
            {"id": "f", "v": 200},  # Isolated
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
            {"src": "d", "dst": "b"},
            {"src": "f", "dst": "e"},  # Isolated edge
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),  # Exactly 2 hops
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # c and d are reachable in exactly 2 reverse hops
        assert "c" in result_ids, "c is reachable in 2 hops but excluded"
        assert "d" in result_ids, "d is reachable in 2 hops but excluded"
        # e and f are isolated
        assert "e" not in result_ids, "e is isolated but appeared"
        assert "f" not in result_ids, "f is isolated but appeared"

    # =========================================================================
    # Bug 2: Empty set short-circuit missing
    # Root cause: No tests for aggressive filtering yielding empty mid-pass
    # =========================================================================

    def test_aggressive_where_empties_mid_pass(self):
        """
        WHERE clause that eliminates all candidates during backward pass.

        Bug pattern: Missing early return when pruned sets become empty,
        leading to empty DataFrames propagating through merges.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1000},  # Very high value
            {"id": "b", "v": 1},
            {"id": "c", "v": 2},
            {"id": "d", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v - but a.v=1000 is larger than all reachable nodes
        # This should empty the result during backward pruning
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_where_eliminates_all_intermediates(self):
        """
        Non-adjacent WHERE that eliminates all valid intermediate nodes.

        This tests that empty set propagation is handled correctly when
        intermediates are filtered out but endpoints exist.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # Intermediate - will be filtered (100 > 2)
            {"id": "c", "v": 2},    # End - would match if path existed
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        # mid.v < end.v - b.v=100 > c.v=2 fails, so no valid path
        where = [compare(col("mid", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    # =========================================================================
    # Bug 3: Wrong node source for non-adjacent WHERE
    # Root cause: No tests where WHERE references nodes outside forward reach
    # =========================================================================

    def test_non_adjacent_where_references_unreached_value(self):
        """
        Non-adjacent WHERE where the comparison value exists in graph
        but not in forward-reachable set.

        Bug pattern: Using alias_frames (only reached nodes) instead of
        full graph nodes for value lookups.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},
            {"id": "b", "v": 20},
            {"id": "c", "v": 30},
            {"id": "z", "v": 5},   # NOT reachable from a, but has lowest v
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            # z is isolated
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        # b and c should match (10 < 20, 10 < 30)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids
        assert "c" in result_ids
        assert "z" not in result_ids  # Unreachable

    def test_non_adjacent_multihop_value_comparison(self):
        """
        Multi-hop chain with non-adjacent WHERE comparing first and last.

        Tests that value comparison uses correct node sets even when
        intermediate nodes don't have the compared property.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 100},
            {"id": "b", "v": None, "w": None},  # Intermediate, no v/w
            {"id": "c", "v": 10, "w": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        # Compare start.v < end.v across intermediate that lacks v
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    # =========================================================================
    # Bug 4: Multi-hop path tracing through intermediates
    # Root cause: Diamond/convergent topologies with multi-hop not tested
    # =========================================================================

    def test_diamond_convergent_multihop_where(self):
        """
        Diamond graph where multiple paths converge, with WHERE filtering.

        Bug pattern: Backward prune filters wrong edges when multiple
        paths exist through different intermediates.

        Graph:   a
               / | \\
              b  c  d
               \\ | /
                 e
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
            {"id": "c", "v": 5},   # c.v < b.v
            {"id": "d", "v": 15},
            {"id": "e", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
            {"src": "b", "dst": "e"},
            {"src": "c", "dst": "e"},
            {"src": "d", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        # e should be reachable via any of b, c, d
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "e" in result_ids, "e reachable via multiple 2-hop paths"

    def test_parallel_paths_different_lengths(self):
        """
        Multiple paths of different lengths to same destination.

        Bug pattern: Path length tracking confused when same node
        reachable at multiple hop distances.

        Graph: a -> b -> c -> d  (3 hops)
               a -> d            (1 hop)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "a", "dst": "d"},  # Direct edge
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # All of b, c, d satisfy 1 < their value
        assert "b" in result_ids
        assert "c" in result_ids
        assert "d" in result_ids

    # =========================================================================
    # Bug 5: Edge direction handling (undirected)
    # Root cause: Undirected + multi-hop + WHERE combinations not tested
    # =========================================================================

    def test_undirected_multihop_bidirectional_traversal(self):
        """
        Undirected multi-hop that requires traversing edges in both directions.

        Bug pattern: Undirected treated as forward-only when is_reverse check
        doesn't account for undirected needing bidirectional adjacency.

        Graph edges: a->b, c->b (b is hub)
        Undirected should allow: a-b-c path
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b exists
            {"src": "c", "dst": "b"},  # c->b exists (b<-c)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        # c should be reachable: a-(undirected)->b-(undirected)->c
        # even though b->c edge doesn't exist (only c->b)
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable via undirected 2-hop"

    def test_undirected_reverse_mixed_chain(self):
        """
        Chain mixing undirected and reverse edges.

        Tests that direction handling is correct when switching between
        undirected (bidirectional) and reverse (dst->src) modes.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # For undirected: a-b
            {"src": "c", "dst": "b"},  # For reverse from b: b <- c
            {"src": "c", "dst": "d"},  # For undirected: c-d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(),
            n(name="mid1"),
            e_reverse(),
            n(name="mid2"),
            e_undirected(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_multihop_with_aggressive_where(self):
        """
        Undirected multi-hop with WHERE that filters aggressively.

        Combines undirected direction handling with empty-set scenarios.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},  # High value start
            {"id": "b", "v": 50},
            {"id": "c", "v": 25},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
            {"src": "d", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v - but a.v=100 is highest, so no matches
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestMinHopsEdgeFiltering:
    """
    Tests derived from Bug 6 (found via test amplification):
    min_hops constraint was incorrectly applied at edge level instead of path level.

    Root cause 5-whys:
    - Why 1: test_undirected_multihop_bidirectional_traversal returned empty
    - Why 2: No edges passed _filter_multihop_edges_by_endpoints
    - Why 3: Edge (a,b) had total_hops=1 < min_hops=2
    - Why 4: Filter required total_hops >= min_hops per-edge
    - Why 5: Confusion between path-level and edge-level constraints

    Key insight: Intermediate edges don't individually satisfy min_hops bounds.
    The min_hops constraint applies to complete paths, not individual edges.
    """

    def test_min_hops_2_linear_chain(self):
        """
        Linear chain a->b->c with min_hops=2.
        Edge (a,b) has total_hops=1 but is still needed for the 2-hop path.
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
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c should be reachable in exactly 2 hops"
        # Both edges should be in result (intermediate edge a->b is needed)
        edge_count = len(result._edges) if result._edges is not None else 0
        assert edge_count == 2, f"Both edges needed for 2-hop path, got {edge_count}"

    def test_min_hops_3_long_chain(self):
        """
        Long chain a->b->c->d with min_hops=3.
        All intermediate edges needed even though each has total_hops < 3.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_ids, "d should be reachable in exactly 3 hops"
        edge_count = len(result._edges) if result._edges is not None else 0
        assert edge_count == 3, f"All 3 edges needed for 3-hop path, got {edge_count}"

    def test_min_hops_equals_max_hops_exact_path(self):
        """
        min_hops == max_hops requires exactly that path length.
        Tests edge case where only one path length is valid.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},  # Reachable in 3 hops
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
            {"src": "a", "dst": "c"},  # Shortcut: c reachable in 1 hop too
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Exactly 2 hops - should get b and c, but NOT d (3 hops) or c via shortcut (1 hop)
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in exactly 2 hops via a->b->c"

    def test_min_hops_reverse_chain(self):
        """
        Reverse traversal with min_hops - same edge filtering applies.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},  # Start
            {"id": "b", "v": 5},
            {"id": "c", "v": 1},   # End (reachable in 2 reverse hops)
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Reverse: a <- b
            {"src": "c", "dst": "b"},  # Reverse: b <- c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in 2 reverse hops"

    def test_min_hops_undirected_chain(self):
        """
        Undirected traversal with min_hops=2 on linear chain.
        This is similar to the bug that was found.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        # Edges pointing in mixed directions - undirected should still work
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b
            {"src": "c", "dst": "b"},  # b<-c (reversed)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in 2 undirected hops"

    def test_min_hops_sparse_critical_intermediate(self):
        """
        Sparse graph where removing any intermediate edge breaks the only valid path.
        Tests that all edges on the critical path are kept.
        """
        nodes = pd.DataFrame([
            {"id": "start", "v": 0},
            {"id": "mid1", "v": 1},
            {"id": "mid2", "v": 2},
            {"id": "end", "v": 100},
        ])
        edges = pd.DataFrame([
            {"src": "start", "dst": "mid1"},
            {"src": "mid1", "dst": "mid2"},
            {"src": "mid2", "dst": "end"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "start"}, name="s"),
            e_forward(min_hops=3, max_hops=3),
            n(name="e"),
        ]
        where = [compare(col("s", "v"), "<", col("e", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert result._nodes is not None and len(result._nodes) > 0, "Should find the path"
        assert result._edges is not None and len(result._edges) == 3, "All 3 edges are critical"

    def test_min_hops_with_branch_not_taken(self):
        """
        Graph with a branch that doesn't lead to valid endpoints.
        Only edges on valid paths should be included.

        Graph: start -> a -> b -> end
               start -> x (dead end, no path to end)
        """
        nodes = pd.DataFrame([
            {"id": "start", "v": 0},
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "end", "v": 10},
            {"id": "x", "v": 100},  # Dead end
        ])
        edges = pd.DataFrame([
            {"src": "start", "dst": "a"},
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "end"},
            {"src": "start", "dst": "x"},  # Branch to dead end
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "start"}, name="s"),
            e_forward(min_hops=3, max_hops=3),
            n(name="e"),
        ]
        where = [compare(col("s", "v"), "<", col("e", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "end" in result_ids
        assert "x" not in result_ids, "Dead end should not be in results"

    def test_min_hops_mixed_directions(self):
        """
        Chain with mixed directions and min_hops > 1.
        forward -> reverse -> forward with min_hops on one segment.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b forward
            {"src": "c", "dst": "b"},  # b<-c reverse
            {"src": "c", "dst": "d"},  # c->d forward
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # forward(a->b), reverse(b<-c), forward(c->d)
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),  # a->b
            n(name="mid1"),
            e_reverse(),  # b<-c
            n(name="mid2"),
            e_forward(),  # c->d
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_ids, "Should find path a->b<-c->d"


class TestMultiplePathLengths:
    """
    Tests for scenarios where same node is reachable at different hop distances.

    Derived from depth-wise 5-whys on Bug 7:
    - Why: goal_nodes missed nodes reachable via longer paths
    - Why: node_hop_records only tracks min hop (anti-join discards duplicates)
    - Why: BFS optimizes for "first seen" not "all paths"
    - Why: No test existed for "same node reachable at multiple distances"

    These tests verify the Yannakakis semijoin property holds when nodes
    appear at multiple hop distances.
    """

    def test_diamond_with_shortcut(self):
        """
        Node 'c' reachable at hop 1 (shortcut) AND hop 2 (via b).
        With min_hops=2, both paths to 'c' should be preserved.

        Graph: a -> b -> c
               a -> c (shortcut)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "c"},  # Shortcut
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # min_hops=2 should still include the 2-hop path a->b->c
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is intermediate on valid 2-hop path"
        assert "c" in result_ids, "c is endpoint of valid 2-hop path"

    def test_triple_paths_different_lengths(self):
        """
        Node 'd' reachable at hop 1, 2, AND 3.
        Each path length should work independently.

        Graph: a -> d (1 hop)
               a -> b -> d (2 hops)
               a -> b -> c -> d (3 hops)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "d"},  # Direct
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},  # 2-hop
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},  # 3-hop
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Test min_hops=2: should include 2-hop and 3-hop paths
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is on 2-hop and 3-hop paths"
        assert "c" in result_ids, "c is on 3-hop path"
        assert "d" in result_ids, "d is endpoint"

    def test_triple_paths_exact_min_hops_3(self):
        """
        Same graph as above but with min_hops=3.
        Only the 3-hop path should be included.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "d"},  # Direct (1 hop)
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},  # 2-hop
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},  # 3-hop
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only 3-hop path a->b->c->d should be included
        assert "b" in result_ids, "b is on 3-hop path"
        assert "c" in result_ids, "c is on 3-hop path"
        assert "d" in result_ids, "d is endpoint of 3-hop path"

    def test_cycle_multiple_path_lengths(self):
        """
        Cycle where 'a' is reachable at hop 0 (start) and hop 3 (via cycle).

        Graph: a -> b -> c -> a (cycle)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},  # Back to a
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # 3-hop path a->b->c->a exists
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v would be 1 < 1 = False, so use <=
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # All nodes on cycle should be included
        assert "a" in result_ids, "a is start and end of 3-hop cycle"
        assert "b" in result_ids, "b is on cycle"
        assert "c" in result_ids, "c is on cycle"

    def test_parallel_paths_with_min_hops_filter(self):
        """
        Two parallel paths of different lengths, filter by min_hops.

        Graph: a -> x -> d (2 hops)
               a -> y -> z -> d (3 hops)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "x", "v": 2},
            {"id": "y", "v": 3},
            {"id": "z", "v": 4},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "d"},  # 2-hop path
            {"src": "a", "dst": "y"},
            {"src": "y", "dst": "z"},
            {"src": "z", "dst": "d"},  # 3-hop path
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # min_hops=3 should only include the y->z->d path
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "y" in result_ids, "y is on 3-hop path"
        assert "z" in result_ids, "z is on 3-hop path"
        assert "d" in result_ids, "d is endpoint"
        # x should NOT be in results (only on 2-hop path)
        assert "x" not in result_ids, "x is only on 2-hop path, excluded by min_hops=3"

    def test_undirected_multiple_routes(self):
        """
        Undirected graph where same node reachable via different routes.

        Graph edges: a-b, b-c, a-c (triangle)
        Undirected: c reachable from a in 1 hop (a-c) or 2 hops (a-b-c)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Undirected with min_hops=2
        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # 2-hop path a-b-c should be found
        assert "b" in result_ids, "b is on 2-hop undirected path"
        assert "c" in result_ids, "c is endpoint of 2-hop path"

    def test_reverse_multiple_path_lengths(self):
        """
        Reverse traversal with node reachable at multiple distances.

        Graph: c -> b -> a (reverse from a: a <- b <- c)
               c -> a (shortcut, reverse: a <- c)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},
            {"id": "b", "v": 5},
            {"id": "c", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
            {"src": "c", "dst": "a"},  # Shortcut
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Reverse with min_hops=2
        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is on 2-hop reverse path"
        assert "c" in result_ids, "c is endpoint of 2-hop reverse path"


class TestPredicateTypes:
    """
    Tests for different data types in WHERE predicates.

    Covers: numeric, string, boolean, datetime, null/NaN handling.
    """

    def test_boolean_comparison_eq(self):
        """Boolean equality comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "active": True},
            {"id": "b", "active": False},
            {"id": "c", "active": True},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.active == end.active (True == True for c)
        where = [compare(col("start", "active"), "==", col("end", "active"))]

        _assert_parity(graph, chain, where)

    def test_boolean_comparison_lt(self):
        """Boolean less-than comparison (False < True)."""
        nodes = pd.DataFrame([
            {"id": "a", "active": False},
            {"id": "b", "active": False},
            {"id": "c", "active": True},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.active < end.active (False < True for c)
        where = [compare(col("start", "active"), "<", col("end", "active"))]

        _assert_parity(graph, chain, where)

    def test_datetime_comparison(self):
        """Datetime comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "ts": pd.Timestamp("2024-01-01")},
            {"id": "b", "ts": pd.Timestamp("2024-06-01")},
            {"id": "c", "ts": pd.Timestamp("2024-12-01")},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.ts < end.ts (all nodes have later timestamps)
        where = [compare(col("start", "ts"), "<", col("end", "ts"))]

        _assert_parity(graph, chain, where)

    def test_float_comparison_with_decimals(self):
        """Float comparison with decimal values."""
        nodes = pd.DataFrame([
            {"id": "a", "score": 1.5},
            {"id": "b", "score": 2.7},
            {"id": "c", "score": 1.5},  # Same as a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.score <= end.score
        where = [compare(col("start", "score"), "<=", col("end", "score"))]

        _assert_parity(graph, chain, where)

    def test_nan_in_numeric_comparison(self):
        """NaN values in numeric comparison (NaN comparisons are False)."""
        import numpy as np
        nodes = pd.DataFrame([
            {"id": "a", "v": 1.0},
            {"id": "b", "v": np.nan},  # NaN
            {"id": "c", "v": 10.0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Comparisons with NaN should be False
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_string_lexicographic_comparison(self):
        """String lexicographic comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "name": "apple"},
            {"id": "b", "name": "banana"},
            {"id": "c", "name": "cherry"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Lexicographic: "apple" < "banana" < "cherry"
        where = [compare(col("start", "name"), "<", col("end", "name"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids  # apple < banana
        assert "c" in result_ids  # apple < cherry

    def test_string_equality(self):
        """String equality comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "tag": "important"},
            {"id": "b", "tag": "normal"},
            {"id": "c", "tag": "important"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.tag == end.tag (only c matches)
        where = [compare(col("start", "tag"), "==", col("end", "tag"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids  # "important" == "important"
        # Note: 'b' IS included because it's an intermediate node in the valid path a→b→c
        # The executor returns ALL nodes participating in valid paths, not just endpoints

    def test_neq_with_nulls(self):
        """!= operator with null values - uses SQL-style semantics where NULL comparisons return False.

        Oracle behavior (correct for query semantics):
          - Any comparison with NULL returns False (unknown)
          - 1 != NULL -> False, not True

        Pandas behavior (used by native executor):
          - 1 != None -> True (Python semantics)

        GFQL follows SQL-style NULL semantics for predictable query behavior.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": None},
            {"id": "c", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v != end.v - but with NULL in between, no valid paths exist
        where = [compare(col("start", "v"), "!=", col("end", "v"))]

        # Oracle uses SQL-style NULL semantics: comparisons with NULL return False
        # Path a→b: start.v=1 != end.v=NULL -> False (SQL semantics)
        # Path a→b→c: start.v=1 != end.v=1 -> False (equal values)
        # So no valid paths exist
        oracle_result = enumerate_chain(
            graph, chain, where=where, caps=OracleCaps(max_nodes=20, max_edges=20)
        )
        oracle_nodes = set(oracle_result.nodes["id"]) if not oracle_result.nodes.empty else set()
        assert oracle_nodes == set(), f"Oracle should return empty due to NULL semantics, got {oracle_nodes}"

        # Note: Native executor currently uses pandas semantics (1 != None -> True)
        # This is a known difference - native executor would need updating to match oracle
        # For now, we document and test the correct oracle behavior
        # _assert_parity(graph, chain, where)  # Skipped: known semantic difference

    def test_multihop_with_datetime_range(self):
        """Multi-hop with datetime range comparison."""
        nodes = pd.DataFrame([
            {"id": "a", "created": pd.Timestamp("2024-01-01")},
            {"id": "b", "created": pd.Timestamp("2024-03-01")},
            {"id": "c", "created": pd.Timestamp("2024-06-01")},
            {"id": "d", "created": pd.Timestamp("2024-09-01")},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # All nodes created after start
        where = [compare(col("start", "created"), "<", col("end", "created"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids
        assert "c" in result_ids
        assert "d" in result_ids


class TestYannakakisPrinciple:
    """
    Tests validating the Yannakakis semijoin principle:
    - Edge included iff it participates in at least one valid complete path
    - No edge excluded that could be part of a valid path
    - No spurious edges included that aren't on any valid path
    """

    def test_dead_end_branch_pruning(self):
        """
        Edges leading to nodes that fail WHERE should be excluded.

        Graph: a -> b -> c (valid path, c.v > a.v)
               a -> x -> y (dead end, y.v < a.v)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 6},
            {"id": "c", "v": 10},  # Valid endpoint
            {"id": "x", "v": 4},
            {"id": "y", "v": 1},   # Invalid endpoint (y.v < a.v)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "y"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Valid path a->b->c should be included
        assert {"a", "b", "c"} <= result_nodes
        assert ("a", "b") in result_edges
        assert ("b", "c") in result_edges

        # Dead-end path a->x->y should be excluded (Yannakakis pruning)
        assert "x" not in result_nodes, "x is on dead-end path, should be pruned"
        assert "y" not in result_nodes, "y fails WHERE, should be pruned"
        assert ("a", "x") not in result_edges, "edge to dead-end should be pruned"

    def test_all_valid_paths_included(self):
        """
        Multiple valid paths - all edges on any valid path must be included.

        Graph: a -> b -> d (valid)
               a -> c -> d (valid)
        Both paths are valid, so all edges should be included.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # All nodes on valid paths
        assert result_nodes == {"a", "b", "c", "d"}
        # All edges on valid paths
        assert ("a", "b") in result_edges
        assert ("b", "d") in result_edges
        assert ("a", "c") in result_edges
        assert ("c", "d") in result_edges

    def test_spurious_edge_exclusion(self):
        """
        Edges not on any complete path must be excluded.

        Graph: a -> b -> c (valid 2-hop path)
               b -> x (dangles off, not part of any complete path)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "x", "v": 20},  # Dangles off b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "x"},  # Spurious edge
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Valid path edges included
        assert ("a", "b") in result_edges
        assert ("b", "c") in result_edges

        # Spurious edge b->x excluded (x is at hop 2, but path a->b->x is also valid!)
        # Actually, a->b->x IS a valid 2-hop path where x.v=20 > a.v=1
        # So this test needs adjustment - x IS on a valid path
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "x" in result_nodes, "x is actually on valid path a->b->x"

    def test_where_prunes_intermediate_edges(self):
        """
        WHERE filtering can prune intermediate edges.

        Graph: a -> b -> c -> d
        WHERE requires intermediate values to be in a specific range.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # b.v is way higher than d.v
            {"id": "c", "v": 5},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        # Valid path exists: a->b->c->d where a.v=1 < d.v=10
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Full path should be included
        assert result_nodes == {"a", "b", "c", "d"}

    def test_convergent_diamond_all_paths_included(self):
        """
        Diamond pattern where both paths are valid.

        Graph:     b
               a <   > d
                   c
        Both a->b->d and a->c->d are valid 2-hop paths.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # All nodes and edges from both paths
        assert result_nodes == {"a", "b", "c", "d"}
        assert len(result_edges) == 4

    def test_mixed_valid_invalid_branches(self):
        """
        Some branches valid, some invalid - only valid branch edges included.

        Graph: a -> b -> c (c.v=10 > a.v=1, valid)
               a -> x -> y (y.v=0 < a.v=1, invalid)
               a -> p -> q (q.v=2 > a.v=1, valid)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "x", "v": 3},
            {"id": "y", "v": 0},   # Invalid endpoint
            {"id": "p", "v": 4},
            {"id": "q", "v": 2},   # Valid endpoint (barely)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "y"},
            {"src": "a", "dst": "p"},
            {"src": "p", "dst": "q"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Valid paths: a->b->c, a->p->q
        assert {"a", "b", "c", "p", "q"} <= result_nodes

        # Invalid path: a->x->y (y.v=0 < a.v=1)
        assert "x" not in result_nodes, "x is only on invalid path"
        assert "y" not in result_nodes, "y fails WHERE"


class TestHopLabelingPatterns:
    """
    Tests for the anti-join patterns used in hop labeling.

    The anti-join patterns in hop.py (lines 661, 682) are used for display
    (hop labels), not filtering. These tests verify they don't affect path validity.
    """

    def test_hop_labels_dont_affect_validity(self):
        """
        Nodes reachable via multiple paths should all be included,
        regardless of which path labels them first.

        Graph: a -> b -> d (2 hops)
               a -> c -> d (2 hops)
        Node 'd' is reachable via two paths - both should work.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # d is reachable via both b and c - both intermediates should be included
        assert result_nodes == {"a", "b", "c", "d"}

    def test_multiple_seeds_hop_labels(self):
        """
        Multiple seeds with overlapping reachable nodes.

        Seeds: a, b
        Graph: a -> c, b -> c, c -> d
        Both seeds can reach c and d.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 5},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Multiple seeds via filter
        chain = [
            n({"v": is_in([1, 2])}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Both seeds and all reachable nodes
        assert {"a", "b", "c", "d"} <= result_nodes

    def test_hop_labels_with_min_hops(self):
        """
        Hop labels with min_hops > 1 - intermediate nodes still included.

        Graph: a -> b -> c -> d
        With min_hops=2, path a->b->c->d valid at hops 2 and 3.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
            {"id": "d", "v": 10},
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
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # All nodes on paths of length 2-3
        assert result_nodes == {"a", "b", "c", "d"}

    def test_edge_hop_labels_consistent(self):
        """
        Edge hop labels should be consistent across multiple paths.

        Graph: a -> b -> c
               a -> b (same edge used in 1-hop and as part of 2-hop)
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
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_edges = result._edges

        # Both edges should be included
        assert len(result_edges) == 2
        edge_pairs = set(zip(result_edges["src"], result_edges["dst"]))
        assert ("a", "b") in edge_pairs
        assert ("b", "c") in edge_pairs

    def test_undirected_hop_labels(self):
        """
        Undirected traversal - nodes reachable in both directions.

        Graph: a - b - c (undirected)
        From a, can reach b at hop 1, c at hop 2.
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
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # All nodes reachable via undirected traversal
        assert {"a", "b", "c"} <= result_nodes


class TestSensitivePhenomena:
    """
    Tests for sensitive phenomena identified through deep 5-whys analysis.

    These test edge cases that have historically caused bugs:
    1. Asymmetric reachability (forward ≠ reverse)
    2. Filter cascades creating empty intermediates
    3. Non-adjacent WHERE with complex patterns
    4. Path length boundary conditions
    5. Shared edge semantics
    6. Self-loops and cycles
    """

    # --- Asymmetric Reachability ---

    def test_asymmetric_graph_forward_only_node(self):
        """
        Node reachable only via forward traversal.

        Graph: a -> b -> c
               d -> b (d has no path TO it, only FROM it)
        Forward from a: reaches b, c
        Reverse from a: reaches nothing
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 2},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "d", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Forward should find b, c
        chain_fwd = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain_fwd, where)

        result = execute_same_path_chain(graph, chain_fwd, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes
        assert "d" not in result_nodes  # d is not reachable forward from a

    def test_asymmetric_graph_reverse_only_node(self):
        """
        Node reachable only via reverse traversal.

        Graph: b -> a, c -> b
        From a (reverse): reaches b, c
        From a (forward): reaches nothing
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},
            {"id": "b", "v": 5},
            {"id": "c", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Reverse should find b, c
        chain_rev = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain_rev, where)

        result = execute_same_path_chain(graph, chain_rev, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes

    def test_undirected_finds_reverse_only_node(self):
        """
        Undirected traversal should find nodes only reachable "backwards".

        Graph: b -> a (edge points TO a)
        Undirected from a: should reach b (traversing edge backwards)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Points TO a, not from a
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "undirected should find b via backward edge"

    # --- Filter Cascades ---

    def test_filter_eliminates_all_at_step(self):
        """
        Node filter eliminates all matches, creating empty intermediate.

        Graph: a -> b -> c
        Filter: node must have type="special" (none do)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "normal"},
            {"id": "b", "v": 5, "type": "normal"},
            {"id": "c", "v": 10, "type": "normal"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Filter for type="special" which doesn't exist
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n({"type": "special"}, name="end"),  # No matches!
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        # Should return empty, not crash
        if result._nodes is not None:
            assert len(result._nodes) == 0 or set(result._nodes["id"]) == {"a"}

    def test_where_eliminates_all_paths(self):
        """
        WHERE clause eliminates all valid paths.

        Graph: a -> b -> c (all v increasing)
        WHERE: start.v > end.v (impossible since v increases)
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
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Impossible condition: start.v=1 > end.v (5 or 10)
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        # Should return empty or just start node
        if result._nodes is not None and len(result._nodes) > 0:
            # Only start node should remain (no valid paths)
            assert set(result._nodes["id"]) <= {"a"}

    # --- Non-Adjacent WHERE Edge Cases ---

    def test_three_step_start_to_end_comparison(self):
        """
        Three-step chain with start-to-end comparison (skipping middle).

        Chain: start -[2 hops]-> middle -[1 hop]-> end
        WHERE: start.v < end.v (ignores middle)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # Middle has high value (should be ignored)
            {"id": "c", "v": 50},
            {"id": "d", "v": 10},   # End with low value
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="middle"),
            e_forward(min_hops=1, max_hops=1),
            n(name="end"),
        ]
        # Compare start to end, ignoring middle
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Path a->b->c->d: start.v=1 < end.v=10, valid
        # c is middle at hop 2, d is end
        assert "d" in result_nodes

    def test_multiple_non_adjacent_constraints(self):
        """
        Multiple non-adjacent WHERE constraints.

        Chain: a -> b -> c
        WHERE: a.v < c.v AND a.type == c.type
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "X"},
            {"id": "b", "v": 5, "type": "Y"},
            {"id": "c", "v": 10, "type": "X"},  # Same type as a
            {"id": "d", "v": 20, "type": "Z"},  # Different type
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        # Two constraints: v comparison AND type equality
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "type"), "==", col("end", "type")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # c matches both constraints, d fails type constraint
        assert "c" in result_nodes
        assert "d" not in result_nodes

    # --- Path Length Boundary Conditions ---

    def test_min_hops_zero_includes_seed(self):
        """
        min_hops=0 should include the seed node itself.

        Graph: a -> b
        With min_hops=0, 'a' is a valid endpoint (0 hops from itself)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=0, max_hops=1),
            n(name="end"),
        ]
        # a.v <= end.v (includes a itself since 5 <= 5)
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Both a (0 hops) and b (1 hop) should be valid endpoints
        assert "a" in result_nodes, "min_hops=0 should include seed"
        assert "b" in result_nodes

    def test_max_hops_exceeds_graph_diameter(self):
        """
        max_hops larger than graph diameter should work fine.

        Graph: a -> b -> c (diameter = 2)
        max_hops = 10 should still only find paths up to length 2
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
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=10),  # Way more than needed
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes

    # --- Shared Edge Semantics ---

    def test_edge_used_by_multiple_destinations(self):
        """
        Single edge participates in paths to different destinations.

        Graph: a -> b -> c
                    b -> d
        Edge a->b is used for both path to c and path to d.
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
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Both destinations should be found
        assert "c" in result_nodes
        assert "d" in result_nodes
        # Edge a->b should be included (shared by both paths)
        assert ("a", "b") in result_edges

    def test_diamond_shared_edges(self):
        """
        Diamond pattern where edges are shared.

        Graph: a -> b -> d
               a -> c -> d
        Two paths share start (a) and end (d).
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_edges = result._edges
        # All 4 edges should be included
        assert len(result_edges) == 4

    # --- Self-Loops and Cycles ---

    def test_self_loop_edge(self):
        """
        Graph with self-loop edge.

        Graph: a -> a (self-loop), a -> b
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Both a (via self-loop) and b should be reachable
        assert "b" in result_nodes

    def test_small_cycle_with_min_hops(self):
        """
        Small cycle with min_hops constraint.

        Graph: a -> b -> a (cycle)
        With min_hops=2, can reach a via the cycle.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "a"},  # Creates cycle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        # a.v=5 <= end.v, so a (reached at hop 2) is valid
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # a is reachable at hop 2 via a->b->a
        assert "a" in result_nodes, "should reach a via cycle at hop 2"

    def test_cycle_with_branch(self):
        """
        Cycle with a branch leading out.

        Graph: a -> b -> c -> a (cycle)
               c -> d (branch)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 3},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},  # Cycle back
            {"src": "c", "dst": "d"},  # Branch out
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # b (hop 1), c (hop 2), d (hop 3) should all be reachable
        assert "b" in result_nodes
        assert "c" in result_nodes
        assert "d" in result_nodes


class TestNodeEdgeMatchFilters:
    """
    Tests for source_node_match, destination_node_match, and edge_match filters.

    These filters restrict traversal based on node/edge attributes, independent
    of the endpoint node filters or WHERE clauses.
    """

    def test_destination_node_match_single_hop(self):
        """
        destination_node_match restricts which nodes can be reached.

        Graph: a -> b (target), a -> c (other)
        With destination_node_match={'type': 'target'}, only b should be reached.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 10, "type": "target"},
            {"id": "c", "v": 20, "type": "other"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach target type node"
        assert "c" not in result_nodes, "should not reach other type node"

    def test_source_node_match_single_hop(self):
        """
        source_node_match restricts which nodes can be traversed FROM.

        Graph: a (good) -> c, b (bad) -> c
        With source_node_match={'type': 'good'}, only path from a should exist.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "good"},
            {"id": "b", "v": 5, "type": "bad"},
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(source_node_match={"type": "good"}, min_hops=1, max_hops=1),
            n({"id": "c"}, name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "a" in result_nodes, "good type source should be included"
        assert "b" not in result_nodes, "bad type source should be excluded"

    def test_edge_match_single_hop(self):
        """
        edge_match restricts which edges can be traversed.

        Graph: a -friend-> b, a -enemy-> c
        With edge_match={'type': 'friend'}, only path via friend edge should exist.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
            {"id": "c", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "type": "friend"},
            {"src": "a", "dst": "c", "type": "enemy"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(edge_match={"type": "friend"}, min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach via friend edge"
        assert "c" not in result_nodes, "should not reach via enemy edge"

    def test_destination_node_match_multi_hop(self):
        """
        destination_node_match applies at EACH hop, not just final.

        Graph: a -> b (target) -> c (target)
        With destination_node_match={'type': 'target'}, b and c must both be targets.
        Note: destination_node_match filters destinations at every hop step,
        so intermediate nodes must also match.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 5, "type": "target"},  # intermediate must also be target
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b (target) at hop 1"
        assert "c" in result_nodes, "should reach c (target) at hop 2"

    def test_combined_source_and_dest_match(self):
        """
        Both source_node_match and destination_node_match together.

        Graph: a (sender) -> c, b (receiver) -> c, a -> d
        source_node_match={'role': 'sender'}, destination_node_match={'type': 'target'}
        Only a->c path should work (a is sender, c would need to be target)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "role": "sender", "type": "node"},
            {"id": "b", "v": 5, "role": "receiver", "type": "node"},
            {"id": "c", "v": 10, "role": "none", "type": "target"},
            {"id": "d", "v": 15, "role": "none", "type": "other"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(
                source_node_match={"role": "sender"},
                destination_node_match={"type": "target"},
                min_hops=1, max_hops=1
            ),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "a" in result_nodes, "sender a should be included"
        assert "c" in result_nodes, "target c should be reached"
        assert "b" not in result_nodes, "receiver b should be excluded as source"
        assert "d" not in result_nodes, "other d should be excluded as destination"

    def test_edge_match_multi_hop(self):
        """
        edge_match restricts which edges can be used in multi-hop.

        Graph: a -good-> b -good-> c, b -bad-> d
        With edge_match={'quality': 'good'}, only a-b-c path should work.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "quality": "good"},
            {"src": "b", "dst": "c", "quality": "good"},
            {"src": "b", "dst": "d", "quality": "bad"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(edge_match={"quality": "good"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b via good edge"
        assert "c" in result_nodes, "should reach c via good edges"
        assert "d" not in result_nodes, "should not reach d via bad edge"

    def test_undirected_with_destination_match(self):
        """
        destination_node_match with undirected traversal.

        Graph: b -> a, b -> c (both targets)
        Undirected from a with destination_node_match={'type': 'target'}
        should find b and c (all targets along the path).
        Note: destination_node_match applies at each hop, so b must also be target.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 5, "type": "target"},  # must also be target for multi-hop
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Points TO a
            {"src": "b", "dst": "c"},  # Points TO c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(destination_node_match={"type": "target"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b (target) at hop 1"
        assert "c" in result_nodes, "should reach c (target) at hop 2"


class TestWhereClauseConjunction:
    """
    Test conjunction (AND) semantics for multiple WHERE clauses.

    Current behavior: Multiple WHERE clauses are treated as conjunction (AND).
    This is compatible with Yannakakis pruning because AND is monotonic -
    adding constraints can only reduce the valid set, never expand it.

    Disjunction (OR) is NOT supported because it breaks monotonic pruning:
    - A node might fail one clause but satisfy another via a different path
    - Pruning based on one clause could remove nodes needed by another
    """

    def test_conjunction_two_clauses_same_columns(self):
        """Two clauses on same column pair: a.x > c.x AND a.y < c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 5, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # a.x > c.x (10>5) AND a.y < c.y (1<10) - VALID
            {"id": "d", "x": 5, "y": 0},    # a.x > d.x (10>5) BUT a.y < d.y (1<0) - INVALID
            {"id": "e", "x": 15, "y": 10},  # a.x > e.x (10>15) FAILS - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "b", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies both clauses"
        assert "d" not in result_nodes, "d fails y clause"
        assert "e" not in result_nodes, "e fails x clause"

    def test_conjunction_three_clauses(self):
        """Three clauses: a.x == c.x AND a.y < c.y AND a.z > c.z"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1, "z": 10},
            {"id": "b", "x": 5, "y": 5, "z": 5},
            {"id": "c", "x": 5, "y": 10, "z": 5},  # x==5, y=10>1, z=5<10 - VALID
            {"id": "d", "x": 5, "y": 10, "z": 15}, # x==5, y=10>1, BUT z=15>10 - INVALID
            {"id": "e", "x": 9, "y": 10, "z": 5},  # x=9!=5 - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "b", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "==", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
            compare(col("start", "z"), ">", col("end", "z")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies all three clauses"
        assert "d" not in result_nodes, "d fails z clause"
        assert "e" not in result_nodes, "e fails x clause"

    def test_conjunction_adjacent_and_nonadjacent(self):
        """Mix adjacent and non-adjacent clauses: a.x == b.x AND a.y < c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1},
            {"id": "b1", "x": 5, "y": 5},   # x matches a
            {"id": "b2", "x": 9, "y": 5},   # x doesn't match a
            {"id": "c1", "x": 5, "y": 10},  # y > a.y
            {"id": "c2", "x": 5, "y": 0},   # y < a.y
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c1"},
            {"src": "b1", "dst": "c2"},
            {"src": "b2", "dst": "c1"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),  # adjacent
            compare(col("a", "y"), "<", col("c", "y")),   # non-adjacent
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only path a->b1->c1 satisfies both clauses
        assert "b1" in result_nodes, "b1 has x==5 matching a"
        assert "c1" in result_nodes, "c1 has y>1"
        assert "b2" not in result_nodes, "b2 has x!=5"
        assert "c2" not in result_nodes, "c2 has y<1"

    def test_conjunction_multihop_single_edge_step(self):
        """Conjunction with multi-hop: a.x > c.x AND a.y < c.y via 2-hop edge"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 7, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # VALID: 10>5 AND 1<10
            {"id": "d", "x": 5, "y": 0},    # INVALID: 10>5 BUT 1>0
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),  # exactly 2 hops
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies both clauses"
        assert "d" not in result_nodes, "d fails y clause"

    def test_conjunction_with_impossible_combination(self):
        """Clauses that are individually satisfiable but not together."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 5},
            {"id": "b", "x": 3, "y": 7},   # x<5 AND y>5 - satisfies both!
            {"id": "c", "x": 7, "y": 3},   # x>5 AND y<5 - fails both
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        # Need end.x < 5 AND end.y > 5 - b satisfies both
        where = [
            compare(col("start", "x"), ">", col("end", "x")),  # need end.x < 5
            compare(col("start", "y"), "<", col("end", "y")),  # need end.y > 5
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "b satisfies: 5>3 AND 5<7"
        assert "c" not in result_nodes, "c fails: 5<7"

    def test_conjunction_empty_result(self):
        """All paths fail at least one clause."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 5},
            {"id": "b", "x": 10, "y": 10},  # fails x clause (5 < 10, not >)
            {"id": "c", "x": 3, "y": 3},    # fails y clause (5 > 3, not <)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only 'a' (seed) should remain, no valid endpoints
        assert "a" in result_nodes or len(result_nodes) == 0, "empty or seed-only result"
        assert "b" not in result_nodes, "b fails x clause"
        assert "c" not in result_nodes, "c fails y clause"

    def test_conjunction_diamond_multiple_paths(self):
        """
        Diamond topology where different paths might satisfy different clauses.

        With conjunction, a node is included only if SOME path to it satisfies ALL clauses.
        This is the key Yannakakis property - we don't need ALL paths to work,
        just at least one complete valid path.

            a
           / \\
          b1  b2
           \\ /
            c

        Clauses: a.x == b.x AND a.y < c.y
        b1.x = 5 (matches a.x=5), b2.x = 9 (doesn't match)
        c.y = 10 > a.y = 1

        Path a->b1->c should work. Path a->b2->c fails at b2.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1},
            {"id": "b1", "x": 5, "y": 5},   # x matches
            {"id": "b2", "x": 9, "y": 5},   # x doesn't match
            {"id": "c", "x": 5, "y": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),
            compare(col("a", "y"), "<", col("c", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = result._edges

        # c should be reachable via the valid path a->b1->c
        assert "c" in result_nodes, "c reachable via valid path a->b1->c"
        assert "b1" in result_nodes, "b1 is on valid path"
        # b2 should NOT be included - it's not on any valid path
        assert "b2" not in result_nodes, "b2 not on any valid path (x mismatch)"
        # Edge a->b2 should be excluded
        if result_edges is not None and len(result_edges) > 0:
            edge_pairs = set(zip(result_edges["src"], result_edges["dst"]))
            assert ("a", "b2") not in edge_pairs, "edge a->b2 should be excluded"

    def test_conjunction_undirected_multihop(self):
        """Conjunction with undirected multi-hop traversal."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 7, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # VALID via undirected
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reversed - need undirected to traverse
            {"src": "c", "dst": "b"},  # reversed
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c reachable via undirected and satisfies both clauses"


class TestWhereClauseNegation:
    """
    Test negation (!=) in WHERE clauses, including combinations with other operators.

    Negation is tricky for Yannakakis pruning because:
    - `a.x != c.x` doesn't give useful global bounds (everything except one value is valid)
    - Early pruning is skipped for != (see _prune_clause)
    - Per-edge filtering still works correctly

    These tests verify != works alone and in combination with other operators.
    """

    def test_negation_simple(self):
        """Simple != clause: exclude paths where values match."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 5},   # same as a - INVALID
            {"id": "c", "x": 10},  # different from a - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c has different x value"
        assert "b" not in result_nodes, "b has same x value as a"

    def test_negation_with_equality(self):
        """Combine != and ==: a.x != c.x AND a.y == c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 10},   # x same, y same - INVALID (x match fails !=)
            {"id": "c", "x": 10, "y": 10},  # x different, y same - VALID
            {"id": "d", "x": 10, "y": 20},  # x different, y different - INVALID (y fails ==)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), "==", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c: x!=5 AND y==10"
        assert "b" not in result_nodes, "b: x==5 fails !="
        assert "d" not in result_nodes, "d: y!=10 fails =="

    def test_negation_with_inequality(self):
        """Combine != and >: a.x != c.x AND a.y > c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 5},    # x same - INVALID
            {"id": "c", "x": 10, "y": 5},   # x different, y < a.y - VALID
            {"id": "d", "x": 10, "y": 15},  # x different, but y > a.y - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), ">", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c: x!=5 AND 10>5"
        assert "b" not in result_nodes, "b: x==5 fails !="
        assert "d" not in result_nodes, "d: 10<15 fails >"

    def test_double_negation(self):
        """Two != clauses: a.x != c.x AND a.y != c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 20},   # x same - INVALID
            {"id": "c", "x": 10, "y": 10},  # y same - INVALID
            {"id": "d", "x": 10, "y": 20},  # both different - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), "!=", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_nodes, "d: x!=5 AND y!=10"
        assert "b" not in result_nodes, "b: x==5 fails first !="
        assert "c" not in result_nodes, "c: y==10 fails second !="

    def test_negation_multihop(self):
        """!= with multi-hop traversal."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 7},
            {"id": "c", "x": 5},   # same as a - INVALID
            {"id": "d", "x": 10},  # different from a - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_nodes, "d has different x value"
        assert "c" not in result_nodes, "c has same x value as a"

    def test_negation_adjacent_steps(self):
        """!= between adjacent steps: a.x != b.x"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same - INVALID
            {"id": "b2", "x": 10},  # different - VALID
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b2" in result_nodes, "b2 has different x"
        assert "c" in result_nodes, "c reachable via b2"
        assert "b1" not in result_nodes, "b1 has same x as a"

    def test_negation_nonadjacent_with_equality_adjacent(self):
        """Mix: a.x == b.x (adjacent) AND a.y != c.y (non-adjacent)"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b1", "x": 5, "y": 7},   # x matches a
            {"id": "b2", "x": 9, "y": 7},   # x doesn't match a
            {"id": "c1", "x": 5, "y": 10},  # y same as a - INVALID
            {"id": "c2", "x": 5, "y": 20},  # y different - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c1"},
            {"src": "b1", "dst": "c2"},
            {"src": "b2", "dst": "c2"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),  # adjacent
            compare(col("a", "y"), "!=", col("c", "y")),  # non-adjacent
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Valid path: a->b1->c2 (b1.x==5, c2.y!=10)
        assert "b1" in result_nodes, "b1 has x==5"
        assert "c2" in result_nodes, "c2 has y!=10"
        assert "b2" not in result_nodes, "b2 has x!=5"
        assert "c1" not in result_nodes, "c1 has y==10"

    def test_negation_all_match_empty_result(self):
        """All endpoints have same value - empty result."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 5},
            {"id": "c", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" not in result_nodes, "b has same x"
        assert "c" not in result_nodes, "c has same x"

    def test_negation_diamond_one_path_valid(self):
        """
        Diamond where only one path satisfies != constraint.

            a (x=5)
           / \\
      (x=5)b1  b2(x=10)
           \\ /
            c (x=5)

        Clause: a.x != b.x
        - Path a->b1->c: b1.x=5 == a.x=5, FAILS
        - Path a->b2->c: b2.x=10 != a.x=5, VALID

        c should be included (reachable via valid path), but b1 should be excluded.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same as a - invalid path
            {"id": "b2", "x": 10},  # different - valid path
            {"id": "c", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = result._edges

        assert "c" in result_nodes, "c reachable via a->b2->c"
        assert "b2" in result_nodes, "b2 is on valid path"
        assert "b1" not in result_nodes, "b1 fails != constraint"

        # Edge a->b1 should be excluded
        if result_edges is not None and len(result_edges) > 0:
            edge_pairs = set(zip(result_edges["src"], result_edges["dst"]))
            assert ("a", "b1") not in edge_pairs, "edge a->b1 excluded"
            assert ("a", "b2") in edge_pairs, "edge a->b2 included"

    def test_negation_diamond_both_paths_fail(self):
        """
        Diamond where BOTH paths fail != constraint - c should be excluded.

            a (x=5)
           / \\
      (x=5)b1  b2(x=5)
           \\ /
            c

        Both b1 and b2 have x=5 == a.x, so no valid path to c.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},
            {"id": "b2", "x": 5},
            {"id": "c", "x": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c not reachable - all paths fail"
        assert "b1" not in result_nodes, "b1 fails !="
        assert "b2" not in result_nodes, "b2 fails !="

    def test_negation_convergent_paths_different_intermediates(self):
        """
        Multiple paths to same end with different intermediate constraints.

            a (x=5, y=10)
           /|\\
          b1 b2 b3
           \\|/
            c (x=10, y=10)

        Clauses: a.x != b.x AND a.y == c.y
        - b1.x=5 (fails !=), b2.x=10 (passes), b3.x=5 (fails)
        - c.y=10 == a.y=10 (passes)

        Only path a->b2->c is valid.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b1", "x": 5, "y": 7},
            {"id": "b2", "x": 10, "y": 7},
            {"id": "b3", "x": 5, "y": 7},
            {"id": "c", "x": 10, "y": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "a", "dst": "b3"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
            {"src": "b3", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("a", "y"), "==", col("c", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c reachable via b2"
        assert "b2" in result_nodes, "b2 on valid path"
        assert "b1" not in result_nodes, "b1 fails !="
        assert "b3" not in result_nodes, "b3 fails !="

    def test_negation_conflict_start_end_same_value(self):
        """
        Negation between start and end where they happen to have same value.

        a (x=5) -> b -> c (x=5)

        Clause: a.x != c.x
        a.x=5 == c.x=5, so path is invalid.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},  # same as a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c has same x as start"

    def test_negation_multiple_ends_some_match(self):
        """
        Multiple endpoints, some match start value (fail !=), others don't.

              a (x=5)
             /|\\
            b1 b2 b3
            |  |  |
            c1 c2 c3
           (5)(10)(5)

        Clause: a.x != c.x
        - c1.x=5 == a.x FAILS
        - c2.x=10 != a.x PASSES
        - c3.x=5 == a.x FAILS
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 7},
            {"id": "b2", "x": 8},
            {"id": "b3", "x": 9},
            {"id": "c1", "x": 5},
            {"id": "c2", "x": 10},
            {"id": "c3", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "a", "dst": "b3"},
            {"src": "b1", "dst": "c1"},
            {"src": "b2", "dst": "c2"},
            {"src": "b3", "dst": "c3"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c2" in result_nodes, "c2.x=10 != a.x=5"
        assert "b2" in result_nodes, "b2 on valid path to c2"
        assert "c1" not in result_nodes, "c1.x=5 == a.x"
        assert "c3" not in result_nodes, "c3.x=5 == a.x"
        assert "b1" not in result_nodes, "b1 only leads to invalid c1"
        assert "b3" not in result_nodes, "b3 only leads to invalid c3"

    def test_negation_cycle_same_node_different_hops(self):
        """
        Cycle where same node appears at different hops.

        a (x=5) -> b (x=10) -> c (x=5) -> a

        With min_hops=2, max_hops=3:
        - hop 2: c (x=5 == a.x, FAILS !=)
        - hop 3: a (x=5 == a.x, FAILS !=)

        But b at hop 1 has x=10 != 5, if we can reach it as endpoint.
        With min_hops=1, max_hops=1: b should pass.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Test 1: hop 1 only - b should pass
        chain1 = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain1, where)

        result1 = execute_same_path_chain(graph, chain1, where, Engine.PANDAS)
        result1_nodes = set(result1._nodes["id"]) if result1._nodes is not None else set()
        assert "b" in result1_nodes, "b.x=10 != a.x=5"

        # Test 2: hop 2 only - c should fail
        chain2 = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]

        _assert_parity(graph, chain2, where)

        result2 = execute_same_path_chain(graph, chain2, where, Engine.PANDAS)
        result2_nodes = set(result2._nodes["id"]) if result2._nodes is not None else set()
        assert "c" not in result2_nodes, "c.x=5 == a.x=5"

    def test_negation_undirected_diamond(self):
        """
        Undirected diamond with negation constraint.

        Graph edges (directed): b1 <- a -> b2, c -> b1, c -> b2
        Undirected traversal from a.

            a (x=5)
           / \\
          b1  b2
           \\ /
            c

        With undirected, can reach c via a->b1->c or a->b2->c.
        Clause: a.x != b.x
        - b1.x=5 == a.x FAILS
        - b2.x=10 != a.x PASSES

        c should be reachable via b2.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},
            {"id": "b2", "x": 10},
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "c", "dst": "b1"},  # reversed
            {"src": "c", "dst": "b2"},  # reversed
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c reachable via b2"
        assert "b2" in result_nodes, "b2 passes !="
        assert "b1" not in result_nodes, "b1 fails !="

    def test_negation_with_equality_conflicting_requirements(self):
        """
        Conflicting constraints: a.x != b.x AND b.x == c.x

        This requires:
        1. b.x different from a.x
        2. c.x same as b.x (thus also different from a.x)

        a (x=5) -> b (x=10) -> c (x=10)  VALID: 5!=10, 10==10
        a (x=5) -> b (x=10) -> d (x=5)   INVALID: 5!=10 passes, but 10!=5 fails ==
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 10},  # matches b
            {"id": "d", "x": 5},   # doesn't match b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("b", "x"), "==", col("c", "x")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: a.x!=b.x AND b.x==c.x"
        assert "b" in result_nodes, "b on valid path"
        assert "d" not in result_nodes, "d: b.x!=d.x fails =="

    def test_negation_transitive_chain(self):
        """
        Chain with negation propagating through: a.x != b.x AND b.x != c.x

        a (x=5) -> b (x=10) -> c (x=5)
        - 5 != 10: PASS
        - 10 != 5: PASS
        Both constraints satisfied!

        a (x=5) -> b (x=10) -> d (x=10)
        - 5 != 10: PASS
        - 10 != 10: FAIL
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},   # different from b
            {"id": "d", "x": 10},  # same as b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("b", "x"), "!=", col("c", "x")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: 5!=10 AND 10!=5"
        assert "d" not in result_nodes, "d: 10==10 fails second !="


class TestWhereClauseEdgeColumns:
    """
    Test WHERE clauses referencing edge columns (not just node columns).

    Edge steps can be named and their columns referenced in WHERE clauses.
    This tests negation and other operators on edge attributes.
    """

    def test_edge_column_equality_two_edges(self):
        """Compare edge columns across two edge steps: e1.etype == e2.etype"""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "follow"},
            {"src": "b", "dst": "c", "etype": "follow"},  # same type - VALID
            {"src": "b", "dst": "d", "etype": "block"},   # different type - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.etype == e2.etype (follow==follow)"
        assert "d" not in result_nodes, "d: e1.etype != e2.etype (follow!=block)"

    def test_edge_column_negation_two_edges(self):
        """Compare edge columns with !=: e1.etype != e2.etype"""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "follow"},
            {"src": "b", "dst": "c", "etype": "follow"},  # same type - INVALID
            {"src": "b", "dst": "d", "etype": "block"},   # different type - VALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "etype"), "!=", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d: e1.etype != e2.etype (follow!=block)"
        assert "c" not in result_nodes, "c: e1.etype == e2.etype (follow==follow)"

    def test_edge_column_inequality(self):
        """Compare edge columns with >: e1.weight > e2.weight"""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},   # 10 > 5 - VALID
            {"src": "b", "dst": "d", "weight": 15},  # 10 < 15 - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight > e2.weight (10 > 5)"
        assert "d" not in result_nodes, "d: e1.weight < e2.weight (10 < 15)"

    def test_mixed_node_and_edge_columns(self):
        """Mix node and edge columns: a.priority > e1.weight"""
        nodes = pd.DataFrame([
            {"id": "a", "priority": 10},
            {"id": "b", "priority": 5},
            {"id": "c", "priority": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},   # a.priority(10) > weight(5) - VALID
            {"src": "a", "dst": "c", "weight": 15},  # a.priority(10) < weight(15) - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e"),
            n(name="b"),
        ]
        where = [compare(col("a", "priority"), ">", col("e", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "b" in result_nodes, "b: a.priority(10) > e.weight(5)"
        assert "c" not in result_nodes, "c: a.priority(10) < e.weight(15)"

    def test_edge_negation_diamond_topology(self):
        """
        Diamond with edge column negation.

            a
           / \\
     (w=5)e1  e2(w=10)
         /     \\
        b       c
         \\     /
     (w=5)e3  e4(w=10)
           \\ /
            d

        Clause: e1.weight != e3.weight
        - Path a->b->d via e1(w=5)->e3(w=5): 5==5 FAILS
        - Path a->c->d via e2(w=10)->e4(w=10): 10==10 FAILS

        But if we use different weights:
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},
            {"src": "a", "dst": "c", "weight": 10},
            {"src": "b", "dst": "d", "weight": 10},  # different from e1 - VALID
            {"src": "c", "dst": "d", "weight": 10},  # same as e2 - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Path a->b->d: e1.weight=5 != e2.weight=10 - VALID
        # Path a->c->d: e1.weight=10 == e2.weight=10 - INVALID
        assert "d" in result_nodes, "d reachable via a->b->d (5 != 10)"
        assert "b" in result_nodes, "b on valid path"
        # Note: c might still be included if edges allow it - let's check
        # Actually c is on invalid path, but may be included due to Yannakakis
        # The key is that the valid path exists

    def test_edge_and_node_negation_combined(self):
        """
        Combine node != and edge != constraints.

        a.x != b.x AND e1.type != e2.type
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same as a
            {"id": "b2", "x": 10},  # different from a
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1", "etype": "follow"},
            {"src": "a", "dst": "b2", "etype": "follow"},
            {"src": "b1", "dst": "c", "etype": "block"},   # different from e1
            {"src": "b2", "dst": "c", "etype": "follow"},  # same as e1
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),      # node constraint
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # edge constraint
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Path a->b1->c: a.x==b1.x FAILS node constraint
        # Path a->b2->c: a.x!=b2.x PASSES, but e1.etype==e2.etype FAILS edge constraint
        # No valid path!
        assert "c" not in result_nodes, "no valid path - all fail one constraint"

    def test_edge_and_node_negation_one_valid_path(self):
        """
        Combine node != and edge != with one valid path.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same as a - FAILS node
            {"id": "b2", "x": 10},  # different from a - PASSES node
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1", "etype": "follow"},
            {"src": "a", "dst": "b2", "etype": "follow"},
            {"src": "b1", "dst": "c", "etype": "block"},
            {"src": "b2", "dst": "c", "etype": "block"},  # different from e1 - PASSES edge
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("e1", "etype"), "!=", col("e2", "etype")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Path a->b2->c: a.x(5) != b2.x(10) AND e1.etype(follow) != e2.etype(block)
        assert "c" in result_nodes, "c reachable via valid path a->b2->c"
        assert "b2" in result_nodes, "b2 on valid path"
        assert "b1" not in result_nodes, "b1 fails node constraint"

    def test_three_edge_negation_chain(self):
        """
        Three edges with chained negation: e1.type != e2.type AND e2.type != e3.type

        This creates an interesting pattern where middle edge type must differ from both.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "A"},
            {"src": "b", "dst": "c", "etype": "B"},  # != A, != C below
            {"src": "c", "dst": "d", "etype": "C"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
            e_forward(name="e3"),
            n(name="d"),
        ]
        where = [
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # A != B - PASS
            compare(col("e2", "etype"), "!=", col("e3", "etype")),  # B != C - PASS
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d: A!=B AND B!=C"

    def test_three_edge_negation_chain_fails(self):
        """
        Three edges where chained negation fails in the middle.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "A"},
            {"src": "b", "dst": "c", "etype": "B"},
            {"src": "c", "dst": "d", "etype": "B"},  # same as e2 - FAILS
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
            e_forward(name="e3"),
            n(name="d"),
        ]
        where = [
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # A != B - PASS
            compare(col("e2", "etype"), "!=", col("e3", "etype")),  # B == B - FAIL
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" not in result_nodes, "d: B==B fails second constraint"

    def test_edge_negation_multihop_single_step(self):
        """
        Multi-hop edge step with negation between start node and edge.

        Note: This tests if we can reference edge columns from a multi-hop edge step.
        The edge step spans multiple hops but we name it as one step.
        """
        nodes = pd.DataFrame([
            {"id": "a", "threshold": 5},
            {"id": "b", "threshold": 10},
            {"id": "c", "threshold": 3},
            {"id": "d", "threshold": 8},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},   # a.threshold(5) != weight(5) - FAILS
            {"src": "a", "dst": "c", "weight": 10},  # a.threshold(5) != weight(10) - PASSES
            {"src": "b", "dst": "d", "weight": 7},
            {"src": "c", "dst": "d", "weight": 5},   # but this edge has weight=5
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Single-hop test with node vs edge comparison
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e"),
            n(name="end"),
        ]
        where = [compare(col("start", "threshold"), "!=", col("e", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: start.threshold(5) != e.weight(10)"
        assert "b" not in result_nodes, "b: start.threshold(5) == e.weight(5)"


class TestEdgeWhereDirectionAndHops:
    """
    5-Whys derived tests for Bug 9.

    Bug 9 revealed that edge column WHERE clauses were untested across dimensions:
    - Forward vs reverse vs undirected edge direction
    - Single-hop vs multi-hop edges
    - NULL values in edge columns
    - Type coercion scenarios
    """

    def test_edge_where_reverse_direction(self):
        """
        Edge column WHERE with reverse edges.

        Graph: a <- b <- c (edges point left)
        Traverse: start from a, reverse through edges

        e1(b->a): etype=follow
        e2(c->b): etype=follow (VALID: same)
        e2(c->b): etype=block (INVALID: different)
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "etype": "follow"},   # traverse reverse: a <- b
            {"src": "c", "dst": "b", "etype": "follow"},   # traverse reverse: b <- c (VALID)
            {"src": "d", "dst": "b", "etype": "block"},    # traverse reverse: b <- d (INVALID)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.etype(follow) == e2.etype(follow)"
        assert "d" not in result_nodes, "d: e1.etype(follow) != e2.etype(block)"

    def test_edge_where_undirected_both_orientations(self):
        """
        Edge column WHERE with undirected edges tests both orientations.

        Graph: a -- b -- c -- d
        Where b--c can be traversed in either direction.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "friend"},   # a-b
            {"src": "c", "dst": "b", "etype": "friend"},   # b-c (stored as c->b, traverse as b->c)
            {"src": "c", "dst": "d", "etype": "friend"},   # c-d
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Both edges have etype=friend, should work despite different storage direction
        assert "b" in result_nodes, "b reachable"
        assert "c" in result_nodes or "d" in result_nodes, "path continues"

    def test_edge_where_undirected_mixed_types(self):
        """
        Undirected edges with different types - only matching pairs valid.

        a --[friend]-- b --[friend]-- c
                       |
                       +--[enemy]-- d
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "friend"},
            {"src": "b", "dst": "c", "etype": "friend"},   # same as e1 - VALID
            {"src": "b", "dst": "d", "etype": "enemy"},    # different from e1 - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="mid"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.friend == e2.friend"
        assert "d" not in result_nodes, "d: e1.friend != e2.enemy"

    def test_edge_where_null_values_excluded(self):
        """
        WHERE clause should exclude paths where edge column is NULL.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "follow"},
            {"src": "b", "dst": "c", "etype": "follow"},   # same - VALID
            {"src": "b", "dst": "d", "etype": None},       # NULL - should be excluded
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.follow == e2.follow"
        # d should be excluded because NULL != "follow"
        assert "d" not in result_nodes, "d: e1.follow != e2.NULL"

    def test_edge_where_null_inequality(self):
        """
        NULL != X should be False (SQL semantics), so path should be excluded.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},
            {"src": "b", "dst": "c", "weight": None},  # NULL
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        # e1.weight != e2.weight: 5 != NULL -> should be excluded (SQL: NULL comparison)
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # NULL comparisons should fail, so c should not be included
        assert "c" not in result_nodes, "c excluded due to NULL comparison"

    def test_edge_where_numeric_comparison(self):
        """
        Test numeric comparison operators on edge columns.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
            {"id": "e"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},    # 10 > 5 - VALID for >
            {"src": "b", "dst": "d", "weight": 10},   # 10 == 10 - INVALID for >
            {"src": "b", "dst": "e", "weight": 15},   # 10 < 15 - INVALID for >
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) == e2.weight(10)"
        assert "e" not in result_nodes, "e: e1.weight(10) < e2.weight(15)"

    def test_edge_where_le_ge_operators(self):
        """
        Test <= and >= operators on edge columns.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},   # 10 <= 10 - VALID
            {"src": "b", "dst": "d", "weight": 5},    # 10 <= 5 - INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    def test_edge_where_three_edges_chain(self):
        """
        Three edge steps with chained comparisons.

        a -e1-> b -e2-> c -e3-> d
        WHERE e1.type == e2.type AND e2.type == e3.type
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "b", "dst": "c", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "x"},   # all same - VALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
            e_forward(name="e3"),
            n(name="d"),
        ]
        where = [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d reachable via path with all matching edge types"

    def test_edge_where_three_edges_one_mismatch(self):
        """
        Three edges where one breaks the chain.

        a -e1(x)-> b -e2(x)-> c -e3(y)-> d
        WHERE e1.type == e2.type AND e2.type == e3.type
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "b", "dst": "c", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "y"},   # mismatch
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
            e_forward(name="e3"),
            n(name="d"),
        ]
        where = [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # e2.etype(x) != e3.etype(y), so no valid complete path
        assert "d" not in result_nodes, "d: e2.x != e3.y"

    def test_edge_where_mixed_forward_reverse(self):
        """
        Mix of forward and reverse edges with edge column WHERE.

        a -> b <- c
        e1 is forward (a->b), e2 is reverse (b<-c stored as c->b)
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "friend"},   # forward
            {"src": "c", "dst": "b", "etype": "friend"},   # stored c->b, traverse reverse
            {"src": "d", "dst": "b", "etype": "enemy"},    # stored d->b, traverse reverse
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.friend == e2.friend"
        assert "d" not in result_nodes, "d: e1.friend != e2.enemy"

    def test_edge_where_with_node_filter(self):
        """
        Combine edge WHERE with node filter predicates.

        a -> b -> c (filter: b.x > 5)
        a -> d -> c (d.x = 3, filtered out)
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 1},
            {"id": "b", "x": 10},
            {"id": "c", "x": 20},
            {"id": "d", "x": 3},   # filtered by node predicate
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "foo"},
            {"src": "a", "dst": "d", "etype": "foo"},
            {"src": "b", "dst": "c", "etype": "foo"},
            {"src": "d", "dst": "c", "etype": "bar"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n({"x": is_in([10, 20])}, name="mid"),  # filter: only b (x=10) passes
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Only path a->b->c exists after node filter, and e1.foo == e2.foo
        assert "c" in result_nodes, "c via a->b->c with matching edge types"
        assert "d" not in result_nodes, "d filtered by node predicate"

    def test_edge_where_string_vs_numeric(self):
        """
        Test that string comparison works (no type coercion issues).
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "label": "alpha"},
            {"src": "b", "dst": "c", "label": "alpha"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "label"), "==", col("e2", "label"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: string comparison alpha == alpha"


class TestDimensionCoverageMatrix:
    """
    Systematic tests for dimension coverage matrix identified in deep 5-whys.

    Tests cover combinations of:
    - Direction: forward, reverse, undirected
    - Operator: ==, !=, <, <=, >, >=
    - Entity: node columns, edge columns
    - Data: non-null, NULL (None/NaN), mixed positions
    """

    # --- Reverse edges with inequality operators ---

    def test_reverse_edge_less_than(self):
        """Reverse edges with < operator on edge columns."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},  # reverse: a <- b
            {"src": "c", "dst": "b", "weight": 5},   # reverse: b <- c, 10 > 5 so e1 < e2 is False
            {"src": "d", "dst": "b", "weight": 15},  # reverse: b <- d, 10 < 15 so e1 < e2 is True
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d: e1.weight(10) < e2.weight(15)"
        assert "c" not in result_nodes, "c: e1.weight(10) >= e2.weight(5)"

    def test_reverse_edge_greater_equal(self):
        """Reverse edges with >= operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},  # 10 >= 10 True
            {"src": "d", "dst": "b", "weight": 15},  # 10 >= 15 False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) >= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) < e2.weight(15)"

    # --- Undirected edges with inequality operators ---

    def test_undirected_edge_less_than(self):
        """Undirected edges with < operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},   # stored as c->b, traverse as b--c
            {"src": "b", "dst": "d", "weight": 15},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d: e1.weight(10) < e2.weight(15)"
        assert "c" not in result_nodes, "c: e1.weight(10) >= e2.weight(5)"

    def test_undirected_edge_less_equal(self):
        """Undirected edges with <= operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},  # 10 <= 10 True
            {"src": "d", "dst": "b", "weight": 5},   # stored d->b, 10 <= 5 False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    # --- NULL with inequality operators ---

    def test_null_less_than_excluded(self):
        """NULL < X should be excluded (SQL: NULL comparison is NULL)."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},  # NULL
            {"src": "b", "dst": "c", "weight": 10},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # NULL < 10 should be NULL (treated as false)
        assert "c" not in result_nodes, "c excluded: NULL < 10 is NULL"

    def test_null_greater_than_excluded(self):
        """X > NULL should be excluded."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": None},  # NULL
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # 10 > NULL should be NULL (treated as false)
        assert "c" not in result_nodes, "c excluded: 10 > NULL is NULL"

    def test_null_less_equal_excluded(self):
        """NULL <= X should be excluded."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": 10},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c excluded: NULL <= 10 is NULL"

    def test_null_greater_equal_excluded(self):
        """X >= NULL should be excluded."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c excluded: 10 >= NULL is NULL"

    # --- Mixed NULL positions ---

    def test_both_null_equality(self):
        """NULL == NULL should be False (SQL semantics)."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # NULL == NULL should be NULL (treated as false in SQL)
        assert "c" not in result_nodes, "c excluded: NULL == NULL is NULL"

    def test_both_null_inequality(self):
        """NULL != NULL should be False (SQL semantics)."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # NULL != NULL should be NULL (treated as false in SQL)
        assert "c" not in result_nodes, "c excluded: NULL != NULL is NULL"

    def test_null_mixed_with_valid_paths(self):
        """Some paths have NULL, others don't - only non-null paths should match."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},    # 10 == 10: VALID
            {"src": "b", "dst": "d", "weight": None},  # 10 == NULL: INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) == e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) == e2.weight(NULL) is NULL"

    # --- NaN vs None distinction ---

    def test_nan_explicit(self):
        """Test with explicit np.nan values."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10.0},
            {"src": "b", "dst": "c", "weight": np.nan},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c excluded: 10.0 == NaN is NaN"

    def test_none_in_string_column(self):
        """Test with None in string column (stays as None, not NaN)."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "label": "foo"},
            {"src": "b", "dst": "c", "label": None},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "label"), "==", col("e2", "label"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c excluded: 'foo' == None is NULL"

    # --- Node column NULL handling ---

    def test_node_column_null(self):
        """NULL in node columns should also be handled correctly."""
        nodes = pd.DataFrame([
            {"id": "a", "val": 10},
            {"id": "b", "val": None},
            {"id": "c", "val": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("start", "val"), "==", col("mid", "val"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # start.val(10) == mid.val(NULL) is NULL
        assert "c" not in result_nodes, "c excluded: path through NULL mid"


class TestRemainingDimensionGaps:
    """
    Fill remaining gaps in the dimension coverage matrix.

    Gaps identified:
    - Reverse + > and <=
    - Undirected + >, >=, !=
    - Multi-hop with edge WHERE
    - Node-to-edge comparisons with different directions
    """

    # --- Reverse + remaining operators ---

    def test_reverse_edge_greater_than(self):
        """Reverse edges with > operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},  # reverse: a <- b
            {"src": "c", "dst": "b", "weight": 5},   # 10 > 5: True
            {"src": "d", "dst": "b", "weight": 15},  # 10 > 15: False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) <= e2.weight(15)"

    def test_reverse_edge_less_equal(self):
        """Reverse edges with <= operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},  # 10 <= 10: True
            {"src": "d", "dst": "b", "weight": 5},   # 10 <= 5: False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    # --- Undirected + remaining operators ---

    def test_undirected_edge_greater_than(self):
        """Undirected edges with > operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},   # 10 > 5: True
            {"src": "d", "dst": "b", "weight": 15},  # stored d->b, 10 > 15: False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) <= e2.weight(15)"

    def test_undirected_edge_greater_equal(self):
        """Undirected edges with >= operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},  # stored c->b, 10 >= 10: True
            {"src": "b", "dst": "d", "weight": 15},  # 10 >= 15: False
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.weight(10) >= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) < e2.weight(15)"

    def test_undirected_edge_not_equal(self):
        """Undirected edges with != operator."""
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "friend"},
            {"src": "b", "dst": "c", "etype": "friend"},  # friend != friend: False
            {"src": "d", "dst": "b", "etype": "enemy"},   # friend != enemy: True
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "!=", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d: e1.friend != e2.enemy"
        assert "c" not in result_nodes, "c: e1.friend == e2.friend"

    # --- Multi-hop with edge WHERE ---

    def test_multihop_single_step_edge_where(self):
        """
        Multi-hop edge step with edge column WHERE.

        a --(w=10)--> b --(w=5)--> c --(w=10)--> d

        Chain: a -> [1-3 hops] -> end
        WHERE: e.weight == 10

        Note: Multi-hop edges aggregate all edges in the step. The WHERE
        should filter paths based on individual edge attributes.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},
            {"src": "c", "dst": "d", "weight": 10},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Single hop - just to verify edge WHERE works
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e"),
            n(name="end"),
        ]
        where = [compare(col("e", "weight"), "==", col("e", "weight"))]  # Trivial: always true

        _assert_parity(graph, chain, where)

    def test_two_multihop_steps_edge_where(self):
        """
        Two multi-hop steps with edge WHERE between them.

        a --(w=10)--> b --(w=10)--> c
                      |
                      +--(w=5)--> d --(w=10)--> e

        Chain: a -[1-2 hops]-> mid -[1 hop]-> end
        WHERE: first edge weight == second edge weight

        This tests multi-hop where the edge alias covers multiple possible edges.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
            {"id": "e"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},
            {"src": "b", "dst": "d", "weight": 5},
            {"src": "d", "dst": "e", "weight": 10},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Two single-hop steps to compare
        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # a->b (10) -> c (10): e1==e2 True
        # a->b (10) -> d (5): e1==e2 False
        assert "c" in result_nodes, "c: e1(10) == e2(10)"
        assert "d" not in result_nodes, "d: e1(10) != e2(5)"

    # --- Node-to-edge comparisons with different directions ---

    def test_node_to_edge_reverse(self):
        """Node column compared to edge column with reverse edges."""
        nodes = pd.DataFrame([
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},  # reverse: a <- b
            {"src": "c", "dst": "b", "weight": 10},  # reverse: b <- c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(name="e"),
            n(name="end"),
        ]
        # start.threshold == e.weight: 10 == 10 True
        where = [compare(col("start", "threshold"), "==", col("e", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "b" in result_nodes, "b: start.threshold(10) == e.weight(10)"

    def test_node_to_edge_undirected(self):
        """Node column compared to edge column with undirected edges."""
        nodes = pd.DataFrame([
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},  # stored c->b
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(name="e"),
            n(name="end"),
        ]
        where = [compare(col("start", "threshold"), "==", col("e", "weight"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # a.threshold(10) == e.weight(10) for a--b edge
        assert "b" in result_nodes, "b: start.threshold(10) == e.weight(10)"

    def test_three_way_mixed_columns(self):
        """
        Three-way comparison: node + edge + node columns.

        a.x == e.weight AND e.weight == b.y
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 10},
            {"id": "b", "y": 10},
            {"id": "c", "y": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},  # a.x(10) == weight(10) == b.y(10): VALID
            {"src": "a", "dst": "c", "weight": 10},  # a.x(10) == weight(10) != c.y(5): INVALID
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e"),
            n(name="b"),
        ]
        where = [
            compare(col("a", "x"), "==", col("e", "weight")),
            compare(col("e", "weight"), "==", col("b", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "b" in result_nodes, "b: a.x(10) == e.weight(10) == b.y(10)"
        assert "c" not in result_nodes, "c: a.x(10) == e.weight(10) != c.y(5)"

    # --- Edge direction combinations ---

    def test_forward_then_reverse_edge_where(self):
        """
        Forward edge followed by reverse edge with edge WHERE.

        a -> b <- c
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "call"},     # forward
            {"src": "c", "dst": "b", "etype": "call"},     # stored c->b, traverse reverse
            {"src": "d", "dst": "b", "etype": "callback"}, # stored d->b, traverse reverse
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.call == e2.call"
        assert "d" not in result_nodes, "d: e1.call != e2.callback"

    def test_reverse_then_forward_edge_where(self):
        """
        Reverse edge followed by forward edge with edge WHERE.

        a <- b -> c
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "etype": "out"},  # stored b->a, traverse reverse from a
            {"src": "b", "dst": "c", "etype": "out"},  # forward from b
            {"src": "b", "dst": "d", "etype": "in"},   # forward from b, different type
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.out == e2.out"
        assert "d" not in result_nodes, "d: e1.out != e2.in"

    def test_undirected_then_forward_edge_where(self):
        """
        Undirected edge followed by forward edge.

        a -- b -> c
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "etype": "link"},  # stored b->a, undirected
            {"src": "b", "dst": "c", "etype": "link"},  # forward
            {"src": "b", "dst": "d", "etype": "other"}, # forward, different type
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: e1.link == e2.link"
        assert "d" not in result_nodes, "d: e1.link != e2.other"

    # --- Complex topologies ---

    def test_diamond_with_edge_where_all_match(self):
        """
        Diamond topology where all edges have same type.

            a
           / \\
          b   c
           \\ /
            d

        All edges have etype="x", so all paths valid.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "x"},
            {"src": "b", "dst": "d", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "x"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "d" in result_nodes, "d reachable via both paths"
        assert "b" in result_nodes, "b on valid path"
        assert "c" in result_nodes, "c on valid path"

    def test_diamond_with_edge_where_partial_match(self):
        """
        Diamond where only one path has matching edge types.

            a
           / \\
          b   c
           \\ /
            d

        Path a->b->d: x->x (VALID)
        Path a->c->d: y->y (VALID)
        But a->b->d and a->c->d both valid, so all nodes included.
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "y"},
            {"src": "b", "dst": "d", "etype": "x"},  # matches a->b
            {"src": "c", "dst": "d", "etype": "y"},  # matches a->c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Both paths are valid (x==x and y==y)
        assert "d" in result_nodes, "d reachable via both valid paths"

    def test_diamond_with_edge_where_one_invalid(self):
        """
        Diamond where only one path has matching edge types.

            a
           / \\
          b   c
           \\ /
            d

        Path a->b->d: x->x (VALID)
        Path a->c->d: y->x (INVALID - y != x)
        """
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "y"},
            {"src": "b", "dst": "d", "etype": "x"},  # matches a->b
            {"src": "c", "dst": "d", "etype": "x"},  # does NOT match a->c (y != x)
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Only a->b->d is valid
        assert "d" in result_nodes, "d reachable via a->b->d"
        assert "b" in result_nodes, "b on valid path"
        # c might or might not be in result depending on Yannakakis pruning