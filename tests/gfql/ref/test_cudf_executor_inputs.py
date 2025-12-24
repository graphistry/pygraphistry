import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward
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
