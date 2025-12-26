import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse
from graphistry.compute.gfql.cudf_executor import (
    build_same_path_inputs,
    CuDFSamePathExecutor,
    execute_same_path_chain,
)
from graphistry.compute.gfql_unified import gfql
from graphistry.compute.chain import Chain
from graphistry.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from graphistry.tests.test_compute import CGFull
from graphistry.compute.gfql.cudf_executor import _CUDF_MODE_ENV


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
    executor = CuDFSamePathExecutor(inputs)

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
    executor = CuDFSamePathExecutor(inputs)
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
    executor = CuDFSamePathExecutor(inputs)
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
    executor = CuDFSamePathExecutor(inputs)
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
    inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
    executor = CuDFSamePathExecutor(inputs)
    executor._forward()
    result = executor._run_gpu()
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=50, max_edges=50),
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
    executor = CuDFSamePathExecutor(inputs)
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
    executor = CuDFSamePathExecutor(inputs)
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

    @pytest.mark.xfail(
        reason="Multi-hop backward prune doesn't trace through intermediate edges to find start nodes",
        strict=True,
    )
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

    @pytest.mark.xfail(
        reason="Multi-hop backward prune doesn't trace through intermediate edges for reverse direction",
        strict=True,
    )
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

    @pytest.mark.xfail(
        reason="WHERE between non-adjacent aliases not applied during backward prune",
        strict=True,
    )
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

    @pytest.mark.xfail(
        reason="Multi-hop + WHERE parity issues between executor and oracle",
        strict=True,
    )
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
                [n(name="s"), e_forward(min_hops=2, max_hops=3), n(name="e")],
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
            executor = CuDFSamePathExecutor(inputs)
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

    @pytest.mark.xfail(
        reason="Multi-hop edges skip WHERE filtering in _is_single_hop check",
        strict=True,
    )
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

    @pytest.mark.xfail(
        reason="Multiple WHERE + mixed hop ranges interaction issues",
        strict=True,
    )
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
            e_forward(min_hops=1, max_hops=2, name="e2"),
            n({"type": "C"}, name="c"),
        ]
        where = [
            compare(col("a", "v"), "<", col("b", "v")),
            compare(col("b", "v"), "<", col("c", "v")),
        ]

        _assert_parity(graph, chain, where)
