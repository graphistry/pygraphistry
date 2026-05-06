"""Operator and bug pattern tests for df_executor."""

import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
    execute_same_path_chain,
)
from graphistry.compute.gfql.same_path_types import col, compare
from graphistry.gfql.ref.enumerator import OracleCaps, enumerate_chain
from tests.gfql.ref.patterns_case_data import (
    FIVE_WHYS_CASES,
    FIVE_WHYS_IDS,
    IMPOSSIBLE_CONSTRAINT_CASES,
    IMPOSSIBLE_CONSTRAINT_IDS,
    LONGER_PATH_CASES,
    LONGER_PATH_IDS,
    MIXED_DIRECTION_CASES,
    MIXED_DIRECTION_IDS,
    MIN_HOPS_CASES,
    MIN_HOPS_IDS,
    MULTIPLE_PATH_LENGTH_CASES,
    MULTIPLE_PATH_LENGTH_IDS,
    PREDICATE_PARITY_CASES,
    PREDICATE_PARITY_IDS,
    PREDICATE_RESULT_CASES,
    PREDICATE_RESULT_IDS,
    UNDIRECTED_BUG_PATTERN_CASES,
    UNDIRECTED_BUG_PATTERN_IDS,
)
from tests.gfql.ref.conftest import (
    _assert_parity,
    assert_node_membership,
    make_cg_graph,
    make_cg_graph_from_rows,
    run_chain_checked,
)


def _chain_single_hop():
    return [n(name="start"), e_forward(), n(name="end")]


def _where_start_end_v(op: str):
    return [compare(col("start", "v"), op, col("end", "v"))]


def _chain_multihop_from_a(op, **kwargs):
    return [
        n({"id": "a"}, name="start"),
        op(**kwargs),
        n(name="end"),
    ]


def _node_edge_sets(result):
    nodes = set(result._nodes["id"]) if result._nodes is not None else set()
    edges = set(map(tuple, result._edges[["src", "dst"]].itertuples(index=False, name=None)))
    return nodes, edges


def _execute_sets(graph, chain, where, env=None, monkeypatch=None):
    if env and monkeypatch is not None:
        for key, value in env.items():
            monkeypatch.setenv(key, value)
    result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
    return _node_edge_sets(result)


def _set_env(monkeypatch, env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)


def _set_vector_env(monkeypatch):
    _set_env(
        monkeypatch,
        {
            "GRAPHISTRY_NON_ADJ_WHERE_STRATEGY": "vector",
            "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS": "2",
            "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX": "10",
        },
    )


def _assert_mode_matches_baseline(
    graph,
    chain,
    where,
    monkeypatch,
    env,
    expected_nodes=None,
    expected_edges=None,
):
    baseline_nodes, baseline_edges = _execute_sets(graph, chain, where)
    mode_nodes, mode_edges = _execute_sets(graph, chain, where, env=env, monkeypatch=monkeypatch)
    if expected_nodes is not None:
        assert baseline_nodes == expected_nodes
    if expected_edges is not None:
        assert baseline_edges == expected_edges
    assert mode_nodes == baseline_nodes
    assert mode_edges == baseline_edges


def _two_hop_chain(start_filter=None, end_filter=None):
    return [
        n(start_filter, name="start") if start_filter is not None else n(name="start"),
        e_forward(),
        n(name="mid"),
        e_forward(),
        n(end_filter, name="end") if end_filter is not None else n(name="end"),
    ]


def _value_mode_graph():
    return make_cg_graph_from_rows(
        [
            {"id": "a", "v": 1},
            {"id": "b", "v": 1},
            {"id": "c", "v": 1},
            {"id": "d", "v": 1},
            {"id": "m1", "v": 0},
            {"id": "m2", "v": 0},
        ],
        [
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
        ],
    )


def _multi_eq_graph():
    return make_cg_graph_from_rows(
        [
            {"id": "a", "group": 1, "v_mod10": 1},
            {"id": "b", "group": 2, "v_mod10": 1},
            {"id": "c", "group": 1, "v_mod10": 1},
            {"id": "d", "group": 2, "v_mod10": 2},
            {"id": "m1", "group": 0, "v_mod10": 0},
            {"id": "m2", "group": 0, "v_mod10": 0},
        ],
        [
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ],
    )


def _multi_clause_graph():
    return make_cg_graph_from_rows(
        [
            {"id": "a", "v": 1, "v_mod10": 1},
            {"id": "b", "v": 2, "v_mod10": 2},
            {"id": "c", "v": 3, "v_mod10": 1},
            {"id": "d", "v": 1, "v_mod10": 1},
            {"id": "m1", "v": 0, "v_mod10": 0},
            {"id": "m2", "v": 0, "v_mod10": 0},
        ],
        [
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ],
    )


class TestP1OperatorsSingleHop:

    @pytest.fixture
    def basic_graph(self):
        return make_cg_graph_from_rows(
            [{"id": "a", "v": 5}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 1}],
            [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "a", "dst": "d"}, {"src": "c", "dst": "d"}],
        )

    @pytest.mark.parametrize(
        "op, expected_in, expected_out",
        [
            ("==", {"a", "b"}, set()),
            ("!=", {"c", "d"}, set()),
            ("<", {"c"}, set()),
            (">", {"d"}, set()),
            ("<=", {"b", "c"}, set()),
            (">=", {"b", "d"}, set()),
        ],
    )
    def test_single_hop_operators(self, basic_graph, op, expected_in, expected_out):
        chain = _chain_single_hop()
        where = _where_start_end_v(op)
        result = run_chain_checked(basic_graph, chain, where)
        result_ids = set(result._nodes["id"])
        for node_id in expected_in:
            assert node_id in result_ids
        for node_id in expected_out:
            assert node_id not in result_ids


# --- P2 tests: longer paths (4+ nodes)


class TestP2LongerPaths:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        LONGER_PATH_CASES,
        ids=LONGER_PATH_IDS,
    )
    def test_longer_path_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)

        if include_ids or exclude_ids:
            result = run_chain_checked(graph, chain, where)
            result_ids = set(result._nodes["id"])
            assert_node_membership(result_ids, include_ids, exclude_ids)


# --- P1 tests: operators × multihop systematic


class TestP1OperatorsMultihop:

    @pytest.fixture
    def multihop_graph(self):
        return make_cg_graph_from_rows(
            [{"id": "a", "v": 5}, {"id": "b", "v": 3}, {"id": "c", "v": 5}, {"id": "d", "v": 10}, {"id": "e", "v": 1}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "b", "dst": "d"}, {"src": "b", "dst": "e"}],
        )

    @pytest.mark.parametrize("op", ["==", "!=", "<", ">", "<=", ">="])
    def test_multihop_operators(self, multihop_graph, op):
        chain = _chain_multihop_from_a(e_forward, min_hops=1, max_hops=2)
        where = _where_start_end_v(op)
        _assert_parity(multihop_graph, chain, where)


# --- P1 tests: undirected + multihop


class TestP1UndirectedMultihop:

    @pytest.mark.parametrize(
        "edges",
        [
            [
                {"src": "a", "dst": "b"},
                {"src": "b", "dst": "c"},
            ],
            [
                {"src": "b", "dst": "a"},
                {"src": "c", "dst": "b"},
            ],
        ],
        ids=["stored-forward", "stored-reverse"],
    )
    def test_undirected_multihop_orientations(self, edges):
        graph = make_cg_graph_from_rows(
            [
                {"id": "a", "v": 1},
                {"id": "b", "v": 5},
                {"id": "c", "v": 10},
            ],
            edges,
        )

        chain = _chain_multihop_from_a(e_undirected, min_hops=1, max_hops=2)
        where = _where_start_end_v("<")

        _assert_parity(graph, chain, where)


# --- P1 tests: mixed direction chains


class TestP1MixedDirectionChains:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        MIXED_DIRECTION_CASES,
        ids=MIXED_DIRECTION_IDS,
    )
    def test_mixed_direction_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)


# --- P2 tests: edge cases and boundary conditions


class TestP2EdgeCases:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, start_filter, attr, op",
        [
            (
                [{"id": "a", "v": 5}],
                [{"src": "a", "dst": "a"}],
                None,
                "v",
                "==",
            ),
            (
                [{"id": "a1", "v": 1}, {"id": "a2", "v": 5}, {"id": "b1", "v": 10}, {"id": "b2", "v": 15}],
                [{"src": "a1", "dst": "a2"}, {"src": "b1", "dst": "b2"}],
                None,
                "v",
                "<",
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}, {"id": "d", "v": 4}],
                [
                    {"src": "a", "dst": "b"},
                    {"src": "a", "dst": "c"},
                    {"src": "a", "dst": "d"},
                    {"src": "b", "dst": "c"},
                    {"src": "b", "dst": "d"},
                    {"src": "c", "dst": "d"},
                ],
                {"id": "a"},
                "v",
                "<",
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": None}, {"id": "c", "v": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                {"id": "a"},
                "v",
                "<",
            ),
            (
                [{"id": "a", "name": "alice"}, {"id": "b", "name": "bob"}, {"id": "c", "name": "charlie"}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                {"id": "a"},
                "name",
                "<",
            ),
        ],
        ids=["single-node", "disconnected", "dense", "null-compare", "string-compare"],
    )
    def test_edge_case_parity(self, node_rows, edge_rows, start_filter, attr, op):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        if start_filter is None:
            chain = [n(name="start"), e_forward(), n(name="end")]
        else:
            chain = [n(start_filter, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")]
        where = [compare(col("start", attr), op, col("end", attr))]
        _assert_parity(graph, chain, where)

    def test_multiple_where_all_operators(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1, "w": 10}, {"id": "b", "v": 5, "w": 5}, {"id": "c", "v": 10, "w": 1}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
        )

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


# --- P3 tests: bug pattern coverage


class TestBugPatternMultihopBackprop:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        [
            (
                [
                    {"id": "a", "v": 1},
                    {"id": "b", "v": 2},
                    {"id": "c", "v": 3},
                    {"id": "d", "v": 4},
                    {"id": "e", "v": 5},
                    {"id": "f", "v": 6},
                    {"id": "g", "v": 7},
                ],
                [
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                    {"src": "c", "dst": "d"},
                    {"src": "d", "dst": "e"},
                    {"src": "e", "dst": "f"},
                    {"src": "f", "dst": "g"},
                ],
                [
                    n({"id": "a"}, name="start"),
                    e_forward(min_hops=1, max_hops=2),
                    n(name="mid1"),
                    e_forward(min_hops=1, max_hops=2),
                    n(name="mid2"),
                    e_forward(min_hops=1, max_hops=2),
                    n(name="end"),
                ],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}, {"id": "d", "v": 4}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "d"}],
                [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=3), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 2}, {"id": "c", "v": 3}, {"id": "d", "v": 4}],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "b", "dst": "d"}, {"src": "c", "dst": "d"}],
                [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "v"), "<", col("end", "v"))],
            ),
        ],
        ids=[
            "three_consecutive_multihop_edges",
            "multihop_with_output_slicing_and_where",
            "multihop_diamond_graph",
        ],
    )
    def test_multihop_backprop_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)


class TestBugPatternMergeSuffix:

    @pytest.mark.parametrize(
        "op, node_rows",
        [
            (
                "==",
                [
                    {"id": "a", "v": 5},
                    {"id": "b", "v": 3},
                    {"id": "c", "v": 5},
                    {"id": "d", "v": 7},
                ],
            ),
            (
                "<",
                [
                    {"id": "a", "v": 5},
                    {"id": "b", "v": 3},
                    {"id": "c", "v": 10},
                    {"id": "d", "v": 1},
                ],
            ),
            (
                "<=",
                [
                    {"id": "a", "v": 5},
                    {"id": "b", "v": 3},
                    {"id": "c", "v": 5},
                    {"id": "d", "v": 10},
                ],
            ),
            (
                ">",
                [
                    {"id": "a", "v": 5},
                    {"id": "b", "v": 3},
                    {"id": "c", "v": 1},
                    {"id": "d", "v": 10},
                ],
            ),
            (
                ">=",
                [
                    {"id": "a", "v": 5},
                    {"id": "b", "v": 3},
                    {"id": "c", "v": 5},
                    {"id": "d", "v": 1},
                ],
            ),
        ],
        ids=["eq", "lt", "lte", "gt", "gte"],
    )
    def test_same_column_ops(self, op, node_rows):
        graph = make_cg_graph_from_rows(
            node_rows,
            [
                {"src": "a", "dst": "b"},
                {"src": "b", "dst": "c"},
                {"src": "b", "dst": "d"},
            ],
        )

        chain = _chain_multihop_from_a(e_forward, min_hops=1, max_hops=2)
        where = [compare(col("start", "v"), op, col("end", "v"))]
        _assert_parity(graph, chain, where)


class TestBugPatternUndirected:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        UNDIRECTED_BUG_PATTERN_CASES,
        ids=UNDIRECTED_BUG_PATTERN_IDS,
    )
    def test_undirected_bug_pattern_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)


class TestImpossibleConstraints:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where",
        IMPOSSIBLE_CONSTRAINT_CASES,
        ids=IMPOSSIBLE_CONSTRAINT_IDS,
    )
    def test_impossible_constraints_matrix(self, node_rows, edge_rows, chain, where):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)

    def test_empty_graph_with_constraints(self):
        graph = make_cg_graph(pd.DataFrame({"id": [], "v": []}), pd.DataFrame({"src": [], "dst": []}))

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_no_edges_with_constraints(self):
        graph = make_cg_graph(pd.DataFrame([{"id": "a", "v": 1}, {"id": "b", "v": 10}]), pd.DataFrame({"src": [], "dst": []}))

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestFiveWhysAmplification:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        FIVE_WHYS_CASES,
        ids=FIVE_WHYS_IDS,
    )
    def test_five_whys_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _assert_parity(graph, chain, where)
        if include_ids or exclude_ids:
            result_ids, _ = _execute_sets(graph, chain, where)
            assert_node_membership(result_ids, include_ids, exclude_ids)


class TestMinHopsEdgeFiltering:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids, expected_edge_count",
        MIN_HOPS_CASES,
        ids=MIN_HOPS_IDS,
    )
    def test_min_hops_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids, expected_edge_count):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert_node_membership(result_ids, include_ids, exclude_ids)
        if expected_edge_count is not None:
            edge_count = len(result._edges) if result._edges is not None else 0
            assert edge_count == expected_edge_count


class TestMultiplePathLengths:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        MULTIPLE_PATH_LENGTH_CASES,
        ids=MULTIPLE_PATH_LENGTH_IDS,
    )
    def test_multiple_path_lengths_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert_node_membership(result_ids, include_ids, exclude_ids)


class TestPredicateTypes:

    @pytest.mark.parametrize(
        "node_rows, where",
        PREDICATE_PARITY_CASES,
        ids=PREDICATE_PARITY_IDS,
    )
    def test_predicate_parity_matrix(self, node_rows, where):
        graph = make_cg_graph_from_rows(node_rows, [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}])
        chain = [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")]
        _assert_parity(graph, chain, where)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids",
        PREDICATE_RESULT_CASES,
        ids=PREDICATE_RESULT_IDS,
    )
    def test_predicate_result_matrix(self, node_rows, edge_rows, chain, where, include_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert_node_membership(result_ids, include_ids)

    def test_neq_with_nulls(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1}, {"id": "b", "v": None}, {"id": "c", "v": 1}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
        )

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

        _assert_parity(graph, chain, where)

class TestNonAdjacentValueMode:
    @pytest.mark.parametrize(
        "graph_builder, chain, where, env, expected_nodes, expected_edges",
        [
            (
                _value_mode_graph,
                _two_hop_chain(start_filter={"v": 1}, end_filter={"v": 1}),
                [compare(col("start", "v"), "==", col("end", "v"))],
                {"GRAPHISTRY_NON_ADJ_WHERE_MODE": "value", "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX": "10"},
                {"a", "m1", "c"},
                {("a", "m1"), ("m1", "c")},
            ),
            (
                _multi_eq_graph,
                _two_hop_chain(),
                [
                    compare(col("start", "group"), "==", col("end", "group")),
                    compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
                ],
                {
                    "GRAPHISTRY_NON_ADJ_WHERE_STRATEGY": "vector",
                    "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS": "2",
                    "GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX": "10",
                },
                {"a", "m1", "c"},
                {("a", "m1"), ("m1", "c")},
            ),
        ],
        ids=["value_mode_matches_baseline", "multi_eq_vector_mode_matches_expected"],
    )
    def test_mode_expected_matches_baseline(
        self, monkeypatch, graph_builder, chain, where, env, expected_nodes, expected_edges
    ):
        _assert_mode_matches_baseline(
            graph_builder(),
            chain,
            where,
            monkeypatch,
            env,
            expected_nodes=expected_nodes,
            expected_edges=expected_edges,
        )

    def test_multi_eq_vector_mode_parity(self, monkeypatch):
        graph = _multi_eq_graph()
        chain = _two_hop_chain()
        where = [
            compare(col("start", "group"), "==", col("end", "group")),
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
        ]

        _set_vector_env(monkeypatch)
        _assert_parity(graph, chain, where)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, env, expected_nodes, expected_edges",
        [
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 1}, {"id": "c", "v": 1}, {"id": "d", "v": 1}, {"id": "m1", "v": 0}, {"id": "m2", "v": 0}],
                [{"src": "a", "dst": "m1"}, {"src": "m1", "dst": "c"}, {"src": "b", "dst": "m2"}],
                [n({"v": 1}, name="start"), e_forward(), n(name="mid"), e_forward(), n({"v": 1}, name="end")],
                [compare(col("start", "v"), "==", col("end", "v"))],
                {"GRAPHISTRY_NON_ADJ_WHERE_MODE": "auto", "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX": "10"},
                {"a", "m1", "c"},
                {("a", "m1"), ("m1", "c")},
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 1}, {"id": "c", "v": 1}, {"id": "d", "v": 2}, {"id": "m1", "v": 0}, {"id": "m2", "v": 0}],
                [{"src": "a", "dst": "m1"}, {"src": "m1", "dst": "c"}, {"src": "b", "dst": "m2"}, {"src": "m2", "dst": "d"}],
                [n({"v": 1}, name="start"), e_forward(), n(name="mid"), e_forward(), n(name="end")],
                [compare(col("start", "v"), "!=", col("end", "v"))],
                {
                    "GRAPHISTRY_NON_ADJ_WHERE_MODE": "value",
                    "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX": "10",
                    "GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS": "!=",
                },
                {"b", "m2", "d"},
                {("b", "m2"), ("m2", "d")},
            ),
        ],
        ids=[
            "auto_mode_matches_baseline",
            "value_mode_neq_matches_baseline",
        ],
    )
    def test_mode_matches_baseline_matrix(
        self, monkeypatch, node_rows, edge_rows, chain, where, env, expected_nodes, expected_edges
    ):
        _assert_mode_matches_baseline(
            make_cg_graph_from_rows(node_rows, edge_rows),
            chain,
            where,
            monkeypatch,
            env,
            expected_nodes=expected_nodes,
            expected_edges=expected_edges,
        )


class TestNonAdjacentBoundsAndOrdering:
    @pytest.mark.parametrize(
        "where, env",
        [
            (
                [compare(col("start", "v"), "<", col("end", "v"))],
                {"GRAPHISTRY_NON_ADJ_WHERE_BOUNDS": "1"},
            ),
            (
                [compare(col("start", "v"), "<", col("end", "v")), compare(col("start", "group"), "==", col("end", "group"))],
                {"GRAPHISTRY_NON_ADJ_WHERE_ORDER": "selectivity"},
            ),
        ],
        ids=[
            "bounds_matches_baseline",
            "ordering_matches_baseline",
        ],
    )
    def test_bounds_ordering_matches_baseline(self, monkeypatch, where, env):
        _assert_mode_matches_baseline(
            make_cg_graph_from_rows(
                [
                    {"id": "a", "v": 1, "group": 1},
                    {"id": "b", "v": 5, "group": 2},
                    {"id": "c", "v": 3, "group": 1},
                    {"id": "d", "v": 2, "group": 2},
                    {"id": "m1", "v": 0, "group": 0},
                    {"id": "m2", "v": 0, "group": 0},
                ],
                [
                    {"src": "a", "dst": "m1"},
                    {"src": "m1", "dst": "c"},
                    {"src": "b", "dst": "m2"},
                    {"src": "m2", "dst": "d"},
                ],
            ),
            _two_hop_chain(),
            where,
            monkeypatch,
            env,
            expected_nodes={"a", "m1", "c"},
            expected_edges={("a", "m1"), ("m1", "c")},
        )


class TestNonAdjacentMultiClause:
    def test_multi_clause_matches_expected(self):
        graph = _multi_clause_graph()
        chain = _two_hop_chain()
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        result_nodes, result_edges = _execute_sets(graph, chain, where)
        assert result_nodes == {"a", "m1", "c"}
        assert result_edges == {("a", "m1"), ("m1", "c")}

    @pytest.mark.parametrize(
        "env",
        [
            {"GRAPHISTRY_NON_ADJ_WHERE_MODE": "auto", "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX": "1"},
            {
                "GRAPHISTRY_NON_ADJ_WHERE_MODE": "auto",
                "GRAPHISTRY_NON_ADJ_WHERE_INEQ_AGG": "1",
                "GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX": "1",
            },
        ],
        ids=[
            "multi_clause_auto_guard_parity",
            "multi_clause_ineq_agg_parity",
        ],
    )
    def test_multi_clause_env_parity(self, monkeypatch, env):
        graph = _multi_clause_graph()
        chain = _two_hop_chain()
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]
        _assert_mode_matches_baseline(graph, chain, where, monkeypatch, env)

    def test_multi_eq_value_mode_matches_expected(self, monkeypatch):
        _assert_mode_matches_baseline(
            _multi_eq_graph(),
            _two_hop_chain(),
            [
                compare(col("start", "group"), "==", col("end", "group")),
                compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            ],
            monkeypatch,
            {"GRAPHISTRY_NON_ADJ_WHERE_MODE": "value", "GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX": "10"},
            expected_nodes={"a", "m1", "c"},
            expected_edges={("a", "m1"), ("m1", "c")},
        )



class TestEdgeWhereSemijoinParity:

    @pytest.fixture
    def edge_value_graph(self):
        return make_cg_graph_from_rows(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}, {"id": "d"}],
            [{"src": "a", "dst": "b", "w": 5}, {"src": "a", "dst": "b", "w": 1}, {"src": "b", "dst": "c", "w": 3}, {"src": "b", "dst": "c", "w": 10}, {"src": "b", "dst": "d", "w": 7}],
        )

    @pytest.mark.parametrize("op", [">", "!="], ids=["gt", "neq"])
    def test_edge_where_semijoin_parity(self, edge_value_graph, monkeypatch, op):
        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "w"), op, col("e2", "w"))]

        baseline = execute_same_path_chain(edge_value_graph, chain, where, Engine.PANDAS)

        monkeypatch.setenv("GRAPHISTRY_EDGE_WHERE_SEMIJOIN", "1")
        semijoin = execute_same_path_chain(edge_value_graph, chain, where, Engine.PANDAS)

        baseline_edges = set(
            map(tuple, baseline._edges[["src", "dst", "w"]].itertuples(index=False, name=None))
        )
        semijoin_edges = set(
            map(tuple, semijoin._edges[["src", "dst", "w"]].itertuples(index=False, name=None))
        )
        assert baseline_edges == semijoin_edges

    def test_edge_where_null_semijoin_parity(self, monkeypatch):
        graph = make_cg_graph_from_rows(
            [{"id": "a"}, {"id": "b"}, {"id": "c"}],
            [{"src": "a", "dst": "b", "w": None}, {"src": "a", "dst": "b", "w": 2}, {"src": "b", "dst": "c", "w": None}, {"src": "b", "dst": "c", "w": 1}],
        )

        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "w"), ">", col("e2", "w"))]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)

        monkeypatch.setenv("GRAPHISTRY_EDGE_WHERE_SEMIJOIN", "1")
        semijoin = execute_same_path_chain(graph, chain, where, Engine.PANDAS)

        baseline_edges = set(
            map(tuple, baseline._edges[["src", "dst", "w"]].itertuples(index=False, name=None))
        )
        semijoin_edges = set(
            map(tuple, semijoin._edges[["src", "dst", "w"]].itertuples(index=False, name=None))
        )
        def _normalize(edges):
            return {
                tuple("<nan>" if pd.isna(value) else value for value in edge)
                for edge in edges
            }

        assert _normalize(baseline_edges) == _normalize(semijoin_edges)

    def test_vector_strategy_mixed_ops_parity(self, monkeypatch):
        graph = make_cg_graph_from_rows(
            [
                {"id": "a", "v": 1, "v_mod10": 1},
                {"id": "b", "v": 2, "v_mod10": 1},
                {"id": "c", "v": 3, "v_mod10": 1},
                {"id": "d", "v": 1, "v_mod10": 2},
                {"id": "m1", "v": 0, "v_mod10": 0},
                {"id": "m2", "v": 0, "v_mod10": 0},
            ],
            [{"src": "a", "dst": "m1"}, {"src": "m1", "dst": "c"}, {"src": "b", "dst": "m2"}, {"src": "m2", "dst": "d"}],
        )

        chain = _two_hop_chain()
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        _set_vector_env(monkeypatch)
        _assert_parity(graph, chain, where)
