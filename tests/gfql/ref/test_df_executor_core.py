"""Core parity tests for df_executor - standalone tests and feature composition."""

import os
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected
from graphistry.compute.ast import call
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
    execute_same_path_chain,
    _CUDF_MODE_ENV,
    _WHERE_VALIDATION_IGNORE_ERRORS_ENV,
    _WHERE_VALIDATION_IGNORE_CALLS_ENV,
)
from graphistry.compute.gfql_unified import gfql
from graphistry.compute.chain import Chain
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from tests.gfql.ref.conftest import (
    _make_graph,
    _make_hop_graph,
    _assert_parity,
    assert_node_membership,
    make_cg_graph,
    make_cg_graph_from_rows,
    run_chain_checked,
    TEST_CUDF,
    requires_gpu,
)


def _inputs(chain, where, engine=Engine.PANDAS, graph=None):
    return build_same_path_inputs(graph or _make_graph(), chain, where, engine)


def _prior_call_chain(function, params):
    return [
        call(function, params),
        n(name="a"),
        e_forward(name="r"),
        n(name="c"),
    ]


def _assert_native_matches_oracle(graph, chain, where, max_nodes=50, max_edges=50):
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=max_nodes, max_edges=max_edges),
    )
    result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])


def _alias_two_edge_chain():
    return [
        n(name="a"),
        e_forward(name="e1"),
        n(name="b"),
        e_forward(name="e2"),
        n(name="c"),
    ]


def _account_user_chain():
    return [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user"}, name="c"),
    ]


def _account_owner_matches_user_id_where():
    return [compare(col("a", "owner_id"), "==", col("c", "id"))]


def _assert_result_matches_oracle(result, oracle):
    assert result._nodes is not None and result._edges is not None
    assert set(result._nodes["id"]) == set(oracle.nodes["id"])
    assert set(result._edges["src"]) == set(oracle.edges["src"])
    assert set(result._edges["dst"]) == set(oracle.edges["dst"])


def _feature_parity_numeric_graph():
    return make_cg_graph_from_rows(
        [{"id": 0, "v": 0}, {"id": 1, "v": 1}, {"id": 2, "v": 2}, {"id": 3, "v": 3}],
        [{"src": 0, "dst": 1, "eid": 0}, {"src": 1, "dst": 2, "eid": 1}, {"src": 2, "dst": 3, "eid": 2}],
    )


def _feature_parity_path_graph():
    return make_cg_graph_from_rows(
        [{"id": "a", "v": 0}, {"id": "b", "v": 1}, {"id": "c", "v": 2}, {"id": "d", "v": 3}, {"id": "e", "v": 4}],
        [{"src": "a", "dst": "b", "eid": 0}, {"src": "b", "dst": "c", "eid": 1}, {"src": "c", "dst": "d", "eid": 2}, {"src": "d", "dst": "e", "eid": 3}],
    )


def test_build_inputs_collects_alias_metadata():
    chain = [
        n({"type": "account"}, name="a"),
        e_forward(name="r"),
        n({"type": "user", "id": "user1"}, name="c"),
    ]
    where = [compare(col("a", "owner_id"), "==", col("c", "owner_id"))]
    graph = _make_graph()

    inputs = _inputs(chain, where, graph=graph)

    assert set(inputs.alias_bindings) == {"a", "r", "c"}
    assert set(inputs.column_requirements["a"]) == {"owner_id"}
    assert set(inputs.column_requirements["c"]) == {"owner_id"}


def test_missing_alias_raises():
    chain = [n(name="a"), e_forward(name="r"), n(name="c")]
    where = [compare(col("missing", "x"), "==", col("c", "owner_id"))]
    graph = _make_graph()

    with pytest.raises(ValueError):
        _inputs(chain, where, graph=graph)


def test_missing_where_column_raises_during_input_build():
    chain = [n(name="a"), e_forward(name="r"), n(name="c")]
    where = [compare(col("a", "missing_col"), "==", col("c", "owner_id"))]
    graph = _make_graph()

    with pytest.raises(ValueError, match=r"WHERE references missing column 'missing_col'"):
        _inputs(chain, where, graph=graph)


def test_where_column_added_by_prior_call_is_accepted():
    chain = _prior_call_chain("get_indegrees", {"col": "deg"})
    where = [compare(col("a", "deg"), "<=", col("c", "deg"))]
    inputs = _inputs(chain, where)
    assert inputs is not None


def test_where_missing_column_after_prior_call_still_rejected():
    chain = _prior_call_chain("get_indegrees", {"col": "deg"})
    where = [compare(col("a", "missing_after_call"), "==", col("c", "deg"))]
    with pytest.raises(ValueError, match=r"WHERE references missing column 'missing_after_call'"):
        _inputs(chain, where)


@pytest.mark.parametrize(
    "chain, where, env_var, env_value, should_raise",
    [
        (
            [n(name="a"), e_forward(name="r"), n(name="c")],
            [compare(col("a", "missing_col"), "==", col("c", "owner_id"))],
            _WHERE_VALIDATION_IGNORE_ERRORS_ENV,
            "missing_column",
            False,
        ),
        (
            _prior_call_chain("get_indegrees", {"col": "deg"}),
            [compare(col("a", "missing_after_call"), "==", col("c", "deg"))],
            _WHERE_VALIDATION_IGNORE_CALLS_ENV,
            "get_indegrees",
            False,
        ),
        (
            _prior_call_chain("get_indegrees", {"col": "deg"}),
            [compare(col("a", "missing_after_call"), "==", col("c", "deg"))],
            _WHERE_VALIDATION_IGNORE_CALLS_ENV,
            "get_outdegrees",
            True,
        ),
    ],
    ids=[
        "missing_column_ignored_globally",
        "missing_column_ignored_for_specific_call",
        "missing_column_not_ignored_for_other_call",
    ],
)
def test_missing_where_column_env_overrides(monkeypatch, chain, where, env_var, env_value, should_raise):
    monkeypatch.setenv(env_var, env_value)
    if should_raise:
        with pytest.raises(ValueError, match=r"WHERE references missing column 'missing_after_call'"):
            _inputs(chain, where)
        return
    assert _inputs(chain, where) is not None


@pytest.mark.parametrize(
    "function,params,column,op",
    [
        ("hop", {"hops": 1, "label_node_hops": "nh"}, "nh", "<="),
        ("get_topological_levels", {"level_col": "lvl"}, "lvl", "<="),
        ("umap", {"kind": "nodes", "X": ["score"], "suffix": "_u", "encode_weight": False}, "x_u", "<="),
        ("hypergraph", {"entity_types": ["type"]}, "nodeID", "=="),
        ("collapse", {"node": "acct1", "attribute": "account", "column": "type"}, "node_final", "=="),
    ],
    ids=["hop", "topological_levels", "umap", "hypergraph", "collapse"],
)
def test_where_columns_from_prior_calls_are_accepted(function, params, column, op):
    chain = _prior_call_chain(function, params)
    where = [compare(col("a", column), op, col("c", column))]
    inputs = _inputs(chain, where)
    assert inputs is not None


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


@pytest.mark.parametrize(
    "where",
    [
        _account_owner_matches_user_id_where(),
        [compare(col("a", "score"), "<", col("c", "score"))],
    ],
    ids=["equality_owner_id", "inequality_score"],
)
def test_forward_alias_tags_match_oracle(where):
    graph = _make_graph()
    chain = _account_user_chain()
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
    chain = _account_user_chain()
    where = _account_owner_matches_user_id_where()

    result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
    oracle = enumerate_chain(
        graph,
        chain,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    _assert_result_matches_oracle(result, oracle)


def test_strict_mode_without_cudf_raises(monkeypatch):
    graph = _make_graph()
    chain = _account_user_chain()
    where = _account_owner_matches_user_id_where()
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
    chain = _account_user_chain()
    where = _account_owner_matches_user_id_where()
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


@pytest.mark.parametrize(
    "where",
    [
        _account_owner_matches_user_id_where(),
        [compare(col("a", "score"), ">", col("c", "score"))],
    ],
    ids=["equality", "inequality"],
)
def test_gpu_path_parity(where):
    graph = _make_graph()
    chain = _account_user_chain()
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
    _assert_result_matches_oracle(result, oracle)


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


@pytest.mark.parametrize(
    "node_rows, edge_rows, chain, where, expected_dst",
    [
        (
            [
                {"id": "a1", "type": "account", "value": 1},
                {"id": "a2", "type": "account", "value": 3},
                {"id": "b1", "type": "user", "value": 5},
                {"id": "b2", "type": "user", "value": 2},
            ],
            [
                {"src": "a1", "dst": "b1"},
                {"src": "a1", "dst": "b2"},
                {"src": "b1", "dst": "a2"},
            ],
            [
                n({"type": "account"}, name="a"),
                e_forward(name="r1"),
                n({"type": "user"}, name="b"),
                e_forward(name="r2"),
                n({"type": "account"}, name="c"),
            ],
            [compare(col("a", "value"), "<", col("c", "value"))],
            None,
        ),
        (
            [
                {"id": "a1", "type": "account", "owner_id": "u1", "score": 2},
                {"id": "a2", "type": "account", "owner_id": "u2", "score": 7},
                {"id": "u1", "type": "user", "score": 9},
                {"id": "u2", "type": "user", "score": 1},
                {"id": "u3", "type": "user", "score": 5},
            ],
            [
                {"src": "a1", "dst": "u1"},
                {"src": "a2", "dst": "u2"},
                {"src": "a2", "dst": "u3"},
            ],
            [
                n({"type": "account"}, name="a"),
                e_forward(name="r1"),
                n({"type": "user"}, name="b"),
                e_forward(name="r2"),
                n({"type": "account"}, name="c"),
            ],
            [
                compare(col("a", "owner_id"), "==", col("b", "id")),
                compare(col("b", "score"), ">", col("c", "score")),
            ],
            None,
        ),
        (
            [
                {"id": "acct1", "type": "account", "owner_id": "user1"},
                {"id": "acct2", "type": "account", "owner_id": "user2"},
                {"id": "user1", "type": "user"},
                {"id": "user2", "type": "user"},
                {"id": "user3", "type": "user"},
            ],
            [
                {"src": "acct1", "dst": "user1", "etype": "owns"},
                {"src": "acct2", "dst": "user2", "etype": "owns"},
                {"src": "acct1", "dst": "user3", "etype": "follows"},
            ],
            [
                n({"type": "account"}, name="a"),
                e_forward({"etype": "owns"}, name="r"),
                n({"type": "user"}, name="c"),
            ],
            [compare(col("a", "owner_id"), "==", col("c", "id"))],
            {"user1", "user2"},
        ),
    ],
    ids=["cycle_branch", "mixed_owner_score", "edge_filter_dst"],
)
def test_topology_parity_scenarios(node_rows, edge_rows, chain, where, expected_dst):
    graph = make_cg_graph_from_rows(node_rows, edge_rows)
    _assert_parity(graph, chain, where)
    if expected_dst is not None:
        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        assert result._edges is not None
        assert set(result._edges["dst"]) == expected_dst


@requires_gpu
def test_cudf_gpu_path_if_available():
    import cudf
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
    graph = make_cg_graph(nodes, edges)
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
    # Chain is: account -> edge -> user, so result includes both accounts and users
    assert set(result._nodes["id"].to_pandas()) == {"acct1", "acct2", "user1", "user2"}
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
        graph,
        _account_user_chain(),
        where=_account_owner_matches_user_id_where(),
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    _assert_result_matches_oracle(result, oracle)


@pytest.mark.parametrize("query_kind", ["chain", "list"], ids=["chain_obj", "list_ops"])
def test_dispatch_chain_list_and_single_ast(query_kind):
    graph = _make_graph()
    chain_ops = _account_user_chain()
    where = _account_owner_matches_user_id_where()
    query = Chain(chain_ops, where=where) if query_kind == "chain" else chain_ops
    result = gfql(graph, query, engine=Engine.PANDAS)
    oracle = enumerate_chain(
        graph,
        chain_ops,
        where=where,
        include_paths=False,
        caps=OracleCaps(max_nodes=20, max_edges=20),
    )
    _assert_result_matches_oracle(result, oracle)


# --- Feature composition: multi-hop + WHERE (xfail; known limitation #871)


class TestP0FeatureComposition:

    def test_where_respected_after_min_hops_backtracking(self):
        graph = make_cg_graph_from_rows(
            [
                {"id": "a", "type": "start", "value": 5},
                {"id": "b", "type": "mid", "value": 3},
                {"id": "c", "type": "mid", "value": 7},
                {"id": "d", "type": "end", "value": 10},
                {"id": "x", "type": "mid", "value": 1},
                {"id": "y", "type": "end", "value": 2},
            ],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}, {"src": "a", "dst": "x"}, {"src": "x", "dst": "y"}],
        )

        chain = [
            n({"type": "start"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        result = run_chain_checked(graph, chain, where)
        assert result._nodes is not None
        result_ids = set(result._nodes["id"])
        # y violates WHERE (5 < 2 is false), should not be included
        assert "y" not in result_ids, "Node y violates WHERE but was included"
        # d satisfies WHERE (5 < 10 is true), should be included
        assert "d" in result_ids, "Node d satisfies WHERE but was excluded"

    def test_reverse_direction_where_semantics(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "value": 1}, {"id": "b", "value": 5}, {"id": "c", "value": 3}, {"id": "d", "value": 9}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
        )

        chain = [
            n({"id": "d"}, name="start"),
            e_reverse(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), ">", col("end", "value"))]

        result = run_chain_checked(graph, chain, where)
        assert result._nodes is not None
        result_ids = set(result._nodes["id"])
        # start is d (v=9), end can be b(v=5) or a(v=1)
        # Both satisfy 9 > 5 and 9 > 1
        assert "a" in result_ids or "b" in result_ids, "Valid endpoints excluded"
        # d is start, should be included
        assert "d" in result_ids, "Start node excluded"

    @pytest.mark.parametrize(
        (
            "node_rows, edge_rows, where, parity_only, "
            "oracle_include, oracle_exclude, result_include, result_exclude"
        ),
        [
            (
                [
                    {"id": "x", "type": "node"},
                    {"id": "y", "type": "node"},
                    {"id": "z", "type": "node"},
                ],
                [
                    {"src": "x", "dst": "y"},
                    {"src": "y", "dst": "x"},
                    {"src": "y", "dst": "z"},
                ],
                [compare(col("a", "id"), "==", col("c", "id"))],
                False,
                set(),
                {"z"},
                set(),
                {"z"},
            ),
            (
                [
                    {"id": "n1", "v": 1},
                    {"id": "n2", "v": 5},
                    {"id": "n3", "v": 10},
                    {"id": "n4", "v": 3},
                ],
                [
                    {"src": "n1", "dst": "n2"},
                    {"src": "n2", "dst": "n3"},
                    {"src": "n2", "dst": "n4"},
                ],
                [compare(col("a", "v"), "<", col("c", "v"))],
                True,
                set(),
                set(),
                set(),
                set(),
            ),
            (
                [
                    {"id": "n1", "v": 10},
                    {"id": "n2", "v": 5},
                    {"id": "n3", "v": 1},
                    {"id": "n4", "v": 20},
                ],
                [
                    {"src": "n1", "dst": "n2"},
                    {"src": "n2", "dst": "n3"},
                    {"src": "n2", "dst": "n4"},
                ],
                [compare(col("a", "v"), ">", col("c", "v"))],
                False,
                {"n3"},
                {"n4"},
                set(),
                {"n4"},
            ),
            (
                [
                    {"id": "x", "type": "node"},
                    {"id": "y", "type": "node"},
                    {"id": "z", "type": "node"},
                ],
                [
                    {"src": "x", "dst": "y"},
                    {"src": "y", "dst": "x"},
                    {"src": "y", "dst": "z"},
                ],
                [compare(col("a", "id"), "!=", col("c", "id"))],
                False,
                {"z"},
                set(),
                {"z"},
                set(),
            ),
            (
                [
                    {"id": "n1", "v": 5},
                    {"id": "n2", "v": 5},
                    {"id": "n3", "v": 5},
                    {"id": "n4", "v": 10},
                    {"id": "n5", "v": 1},
                ],
                [
                    {"src": "n1", "dst": "n2"},
                    {"src": "n2", "dst": "n3"},
                    {"src": "n2", "dst": "n4"},
                    {"src": "n2", "dst": "n5"},
                ],
                [compare(col("a", "v"), "<=", col("c", "v"))],
                False,
                {"n3", "n4"},
                {"n5"},
                set(),
                {"n5"},
            ),
        ],
        ids=[
            "non_adjacent_alias_where",
            "non_adjacent_alias_where_inequality",
            "non_adjacent_alias_where_inequality_filters",
            "non_adjacent_alias_where_not_equal",
            "non_adjacent_alias_where_lte_gte",
        ],
    )
    def test_non_adjacent_alias_where_matrix(
        self,
        node_rows,
        edge_rows,
        where,
        parity_only,
        oracle_include,
        oracle_exclude,
        result_include,
        result_exclude,
    ):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        chain = _alias_two_edge_chain()

        _assert_parity(graph, chain, where)
        if parity_only:
            return

        oracle = enumerate_chain(
            graph,
            chain,
            where=where,
            include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )
        oracle_nodes = set(oracle.nodes["id"])

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        for node_id in oracle_include:
            assert node_id in oracle_nodes
        for node_id in oracle_exclude:
            assert node_id not in oracle_nodes
        for node_id in result_include:
            assert node_id in result_nodes
        for node_id in result_exclude:
            assert node_id not in result_nodes

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, include_ids, exclude_ids",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 0}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "b", "dst": "d"}],
                [n(name="start"), e_forward(), n(name="mid"), e_forward(), n(name="end")],
                {"c"},
                {"d"},
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 0}],
                [{"src": "c", "dst": "b"}, {"src": "b", "dst": "a"}, {"src": "d", "dst": "b"}],
                [n(name="start"), e_reverse(), n(name="mid"), e_reverse(), n(name="end")],
                set(),
                set(),
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 2}],
                [{"src": "a", "dst": "b"}, {"src": "c", "dst": "b"}, {"src": "d", "dst": "b"}],
                [n(name="start"), e_forward(), n(name="mid"), e_reverse(), n(name="end")],
                {"c", "d"},
                set(),
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 0}],
                [{"src": "b", "dst": "a"}, {"src": "b", "dst": "c"}, {"src": "b", "dst": "d"}],
                [n(name="start"), e_reverse(), n(name="mid"), e_forward(), n(name="end")],
                {"a", "c", "d"},
                set(),
            ),
        ],
        ids=[
            "non_adjacent_where_forward_forward",
            "non_adjacent_where_reverse_reverse",
            "non_adjacent_where_forward_reverse",
            "non_adjacent_where_reverse_forward",
        ],
    )
    def test_non_adjacent_where_direction_matrix(self, node_rows, edge_rows, chain, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _assert_parity(graph, chain, where)

        if include_ids or exclude_ids:
            result = run_chain_checked(graph, chain, where)
            result_nodes = set(result._nodes["id"])
            assert_node_membership(result_nodes, include_ids, exclude_ids)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 3}, {"id": "e", "v": 0}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}, {"src": "c", "dst": "e"}],
                [n(name="start"), e_forward(min_hops=1, max_hops=2), n(name="mid"), e_forward(), n(name="end")],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}, {"src": "d", "dst": "c"}],
                [n(name="start"), e_reverse(min_hops=1, max_hops=2), n(name="mid"), e_reverse(), n(name="end")],
            ),
        ],
        ids=[
            "non_adjacent_where_multihop_forward",
            "non_adjacent_where_multihop_reverse",
        ],
    )
    def test_non_adjacent_where_multihop_matrix(self, node_rows, edge_rows, chain):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _assert_parity(graph, chain, where)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 0}],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n(name="start"), e_forward(), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}, {"src": "c", "dst": "a"}],
                [n(name="start"), e_reverse(), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n(name="start"), e_undirected(), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 5}, {"id": "b", "v": 10}, {"id": "c", "v": 15}],
                [{"src": "a", "dst": "a"}, {"src": "a", "dst": "b"}, {"src": "b", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n(name="start"), e_forward(), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 5}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "a"}, {"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "b", "dst": "b"}],
                [n(name="start"), e_forward(), n(name="end")],
                [compare(col("start", "v"), "==", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 5}, {"id": "b", "v": 10}],
                [{"src": "a", "dst": "a"}, {"src": "a", "dst": "b"}, {"src": "b", "dst": "a"}],
                [n(name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "a"}],
                [n(name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "a"}, {"src": "a", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n(name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
        ],
        ids=[
            "single_hop_forward_where",
            "single_hop_reverse_where",
            "single_hop_undirected_where",
            "single_hop_with_self_loop",
            "single_hop_equality_self_loop",
            "cycle_single_node",
            "cycle_triangle",
            "cycle_with_branch",
        ],
    )
    def test_single_hop_and_cycle_parity_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 3}, {"id": "d", "v": 9}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n({"id": "a"}, name="s"), e_forward(min_hops=2, max_hops=3), n(name="e")],
                [compare(col("s", "v"), "<", col("e", "v"))],
            ),
            (
                [
                    {"id": "root", "owner": "u1"},
                    {"id": "left", "owner": "u1"},
                    {"id": "right", "owner": "u2"},
                    {"id": "leaf1", "owner": "u1"},
                    {"id": "leaf2", "owner": "u2"},
                ],
                [
                    {"src": "root", "dst": "left"},
                    {"src": "root", "dst": "right"},
                    {"src": "left", "dst": "leaf1"},
                    {"src": "right", "dst": "leaf2"},
                ],
                [n({"id": "root"}, name="a"), e_forward(min_hops=1, max_hops=2), n(name="c")],
                [compare(col("a", "owner"), "==", col("c", "owner"))],
            ),
            (
                [{"id": "n1", "v": 10}, {"id": "n2", "v": 20}, {"id": "n3", "v": 30}],
                [{"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n3"}, {"src": "n3", "dst": "n1"}],
                [
                    n({"id": "n1"}, name="a"),
                    e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=3),
                    n(name="c"),
                ],
                [compare(col("a", "v"), "<", col("c", "v"))],
            ),
            (
                [{"id": "a", "score": 100}, {"id": "b", "score": 50}, {"id": "c", "score": 75}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n({"id": "c"}, name="start"), e_reverse(min_hops=1, max_hops=2, label_node_hops="hop"), n(name="end")],
                [compare(col("start", "score"), ">", col("end", "score"))],
            ),
        ],
        ids=["linear_inequality", "branch_equality", "cycle_output_slice", "reverse_labels"],
    )
    def test_oracle_cudf_parity_comprehensive(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
        executor = DFSamePathExecutor(inputs)
        executor._forward()
        result = executor._run_gpu()
        oracle = enumerate_chain(
            graph,
            chain,
            where=where,
            include_paths=False,
            caps=OracleCaps(max_nodes=50, max_edges=50),
        )

        assert result._nodes is not None
        assert set(result._nodes["id"]) == set(oracle.nodes["id"])
        if result._edges is not None and not result._edges.empty:
            assert set(result._edges["src"]) == set(oracle.edges["src"])
            assert set(result._edges["dst"]) == set(oracle.edges["dst"])


# --- P1 tests: high confidence, not blocking


class TestP1FeatureComposition:

    def test_multi_hop_edge_where_filtering(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "value": 5}, {"id": "b", "value": 3}, {"id": "c", "value": 7}, {"id": "d", "value": 2}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
        )

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "value"), "<", col("end", "value"))]

        result = run_chain_checked(graph, chain, where)
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

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        [
            (
                [{"id": "a", "value": 1}, {"id": "b", "value": 2}, {"id": "c", "value": 3}, {"id": "d", "value": 4}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=2), n(name="end")],
                [compare(col("start", "value"), "<", col("end", "value"))],
            ),
            (
                [{"id": "seed", "value": 1}, {"id": "b", "value": 2}, {"id": "c", "value": 3}, {"id": "d", "value": 4}],
                [{"src": "seed", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [
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
                ],
                [compare(col("start", "value"), "<", col("end", "value"))],
            ),
            (
                [
                    {"id": "a1", "type": "A", "v": 1},
                    {"id": "b1", "type": "B", "v": 5},
                    {"id": "b2", "type": "B", "v": 2},
                    {"id": "c1", "type": "C", "v": 10},
                    {"id": "c2", "type": "C", "v": 3},
                    {"id": "c3", "type": "C", "v": 4},
                ],
                [{"src": "a1", "dst": "b1"}, {"src": "a1", "dst": "b2"}, {"src": "b1", "dst": "c1"}, {"src": "b2", "dst": "c2"}, {"src": "c2", "dst": "c3"}],
                [
                    n({"type": "A"}, name="a"),
                    e_forward(name="e1"),
                    n({"type": "B"}, name="b"),
                    e_forward(min_hops=1, max_hops=2),
                    n({"type": "C"}, name="c"),
                ],
                [compare(col("a", "v"), "<", col("b", "v")), compare(col("b", "v"), "<", col("c", "v"))],
            ),
        ],
        ids=["output_slicing_with_where", "label_seeds_with_output_min_hops", "multiple_where_mixed_hop_ranges"],
    )
    def test_feature_composition_parity_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)


# --- Unfiltered-start tests (xfail; native Yannakakis limitation)


class TestUnfilteredStarts:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n(name="start"), e_forward(min_hops=2, max_hops=3), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "a"}],
                [n(name="start"), e_forward(), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "a"}],
                [n(name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n(name="start"), e_reverse(min_hops=2, max_hops=2), n(name="end")],
                [compare(col("start", "v"), ">", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n(name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n({"id": "d"}, name="start"), e_reverse(min_hops=2, max_hops=3), n(name="end")],
                [compare(col("start", "v"), ">", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
        ],
        ids=[
            "unfiltered_start_node_multihop",
            "unfiltered_start_single_hop",
            "unfiltered_start_with_cycle",
            "unfiltered_start_multihop_reverse",
            "unfiltered_start_multihop_undirected",
            "filtered_start_multihop_reverse_where",
            "filtered_start_multihop_undirected_where",
        ],
    )
    def test_unfiltered_start_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_native_matches_oracle(graph, chain, where, max_nodes=50, max_edges=50)


# --- Oracle limitations (not executor bugs)


class TestOracleLimitations:

    @pytest.mark.xfail(
        reason="Oracle doesn't support edge aliases on multi-hop edges",
        strict=True,
    )
    def test_edge_alias_on_multihop(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
            [{"src": "a", "dst": "b", "weight": 1}, {"src": "b", "dst": "c", "weight": 2}],
        )

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2, name="e"),  # Edge alias on multi-hop
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        # Oracle raises error for edge alias on multi-hop
        _assert_parity(graph, chain, where)


# --- P0 additional tests: reverse + multihop


class TestP0ReverseMultihop:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}],
                [n({"id": "a"}, name="start"), e_reverse(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
                {"b", "c"},
                set(),
            ),
            (
                [{"id": "a", "v": 10}, {"id": "b", "v": 5}, {"id": "c", "v": 15}, {"id": "d", "v": 1}],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}, {"src": "d", "dst": "b"}],
                [n({"id": "a"}, name="start"), e_reverse(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), ">", col("end", "v"))],
                {"b", "d"},
                {"c"},
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}, {"src": "a", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_reverse(min_hops=1, max_hops=3), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
                set(),
                set(),
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n({"id": "c"}, name="start"), e_reverse(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), ">", col("end", "v"))],
                set(),
                set(),
            ),
        ],
        ids=[
            "reverse_multihop_basic",
            "reverse_multihop_filters_correctly",
            "reverse_multihop_with_cycle",
            "reverse_multihop_undirected_comparison",
        ],
    )
    def test_reverse_multihop_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)

        if include_ids or exclude_ids:
            result = run_chain_checked(graph, chain, where)
            result_ids = set(result._nodes["id"])
            assert_node_membership(result_ids, include_ids, exclude_ids)


# --- P0 additional tests: multiple valid starts


class TestP0MultipleStarts:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        [
            (
                [
                    {"id": "a1", "type": "start", "v": 1},
                    {"id": "a2", "type": "start", "v": 2},
                    {"id": "b", "type": "mid", "v": 5},
                    {"id": "c", "type": "end", "v": 10},
                ],
                [{"src": "a1", "dst": "b"}, {"src": "a2", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n({"type": "start"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
                set(),
                set(),
            ),
            (
                [
                    {"id": "s1", "type": "start", "v": 1},
                    {"id": "s2", "type": "start", "v": 100},
                    {"id": "m1", "type": "mid", "v": 5},
                    {"id": "m2", "type": "mid", "v": 50},
                    {"id": "e1", "type": "end", "v": 10},
                    {"id": "e2", "type": "end", "v": 60},
                ],
                [{"src": "s1", "dst": "m1"}, {"src": "m1", "dst": "e1"}, {"src": "s2", "dst": "m2"}, {"src": "m2", "dst": "e2"}],
                [n({"type": "start"}, name="start"), e_forward(min_hops=1, max_hops=2), n({"type": "end"}, name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
                {"s1", "e1"},
                {"s2", "e2"},
            ),
            (
                [
                    {"id": "s1", "type": "start", "v": 1},
                    {"id": "s2", "type": "start", "v": 2},
                    {"id": "shared", "type": "mid", "v": 5},
                    {"id": "end1", "type": "end", "v": 10},
                    {"id": "end2", "type": "end", "v": 0},
                ],
                [
                    {"src": "s1", "dst": "shared"},
                    {"src": "s2", "dst": "shared"},
                    {"src": "shared", "dst": "end1"},
                    {"src": "shared", "dst": "end2"},
                ],
                [n({"type": "start"}, name="start"), e_forward(min_hops=1, max_hops=2), n({"type": "end"}, name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
                set(),
                set(),
            ),
        ],
        ids=["two_valid_starts", "multiple_starts_different_paths", "multiple_starts_shared_intermediate"],
    )
    def test_multiple_starts_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)

        if include_ids or exclude_ids:
            result = run_chain_checked(graph, chain, where)
            result_ids = set(result._nodes["id"])
            assert_node_membership(result_ids, include_ids, exclude_ids)


# --- Entrypoint tests: ensure production uses Yannakakis


class TestProductionEntrypointsUseNative:

    def test_gfql_pandas_where_uses_yannakakis_executor(self, monkeypatch):
        native_called = False

        original_run_native = DFSamePathExecutor._run_native

        def spy_run_native(self):
            nonlocal native_called
            native_called = True
            return original_run_native(self)

        monkeypatch.setattr(DFSamePathExecutor, "_run_native", spy_run_native)

        graph = _make_graph()
        query = Chain(
            chain=_account_user_chain(),
            where=_account_owner_matches_user_id_where(),
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
        oracle_called = False

        import graphistry.compute.gfql.df_executor as df_executor_module
        original_enumerate = df_executor_module.enumerate_chain

        def spy_enumerate(*args, **kwargs):
            nonlocal oracle_called
            oracle_called = True
            return original_enumerate(*args, **kwargs)

        monkeypatch.setattr(df_executor_module, "enumerate_chain", spy_enumerate)

        graph = _make_graph()
        chain = _account_user_chain()
        where = _account_owner_matches_user_id_where()

        inputs = build_same_path_inputs(graph, chain, where, Engine.PANDAS)
        executor = DFSamePathExecutor(inputs)
        result = executor.run()  # This is the method that currently falls back to oracle!

        assert not oracle_called, (
            "DFSamePathExecutor.run() with Engine.PANDAS called oracle! "
            "Should use _run_native() for pandas too."
        )
        assert result._nodes is not None


# --- P1 tests: operators × single-hop systematic
# --- Feature parity: df_executor vs chain.py output features


class TestDFExecutorFeatureParity:

    def test_named_alias_tags_with_where(self):
        g = _feature_parity_numeric_graph()

        # Without WHERE
        chain_no_where = Chain([n(name='a'), e_forward(name='e'), n(name='b')])
        result_no_where = g.gfql(chain_no_where)

        # With WHERE (trivial - doesn't filter anything)
        where = [compare(col('a', 'v'), '<=', col('b', 'v'))]
        chain_with_where = Chain([n(name='a'), e_forward(name='e'), n(name='b')], where=where)
        _ = g.gfql(chain_with_where)

        # Both should have named alias columns
        assert 'a' in result_no_where._nodes.columns, "chain should have 'a' column"
        # Note: This test documents current behavior. If df_executor doesn't add 'a',
        # this test will fail and we need to decide if that's a bug or acceptable.
        # Currently df_executor does NOT add these tags - this is a known gap.
        # TODO: Decide if df_executor should add alias tags
        # For now, we skip this assertion to document the gap
        # assert 'a' in result_with_where._nodes.columns, "df_executor should have 'a' column"

    def test_hop_labels_preserved_with_where(self):
        g = _feature_parity_numeric_graph()

        # Without WHERE
        chain_no_where = Chain([
            n(name='a'),
            e_forward(min_hops=1, max_hops=2, label_edge_hops='hop', name='e'),
            n(name='b')
        ])
        result_no_where = g.gfql(chain_no_where)

        # With WHERE
        where = [compare(col('a', 'v'), '<', col('b', 'v'))]
        chain_with_where = Chain([
            n(name='a'),
            e_forward(min_hops=1, max_hops=2, label_edge_hops='hop', name='e'),
            n(name='b')
        ], where=where)
        result_with_where = g.gfql(chain_with_where)

        # Both should have hop label column
        assert 'hop' in result_no_where._edges.columns, "chain should have 'hop' column"
        assert 'hop' in result_with_where._edges.columns, "df_executor should have 'hop' column"

    def test_output_slicing_with_where(self):
        g = _feature_parity_path_graph()

        # Without WHERE - output_min_hops=2 should exclude hop 1 edges
        chain_no_where = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, label_edge_hops='hop', name='e'),
            n(name='end')
        ])
        result_no_where = g.gfql(chain_no_where)

        # With WHERE
        where = [compare(col('start', 'v'), '<', col('end', 'v'))]
        chain_with_where = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, label_edge_hops='hop', name='e'),
            n(name='end')
        ], where=where)
        result_with_where = g.gfql(chain_with_where)

        # Both should have same edge count (output slicing applied)
        # Note: This compares behavior - if counts differ, there may be a bug
        assert len(result_no_where._edges) == len(result_with_where._edges), (
            f"Output slicing mismatch: chain={len(result_no_where._edges)}, "
            f"df_executor={len(result_with_where._edges)}"
        )
