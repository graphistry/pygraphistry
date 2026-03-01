"""5-whys amplification and WHERE clause tests for df_executor."""

import pytest

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected
from graphistry.compute.gfql.df_executor import execute_same_path_chain
from graphistry.compute.gfql.same_path_types import col, compare
from tests.gfql.ref.amplify_case_data import (
    HOP_LABEL_PATTERNS_CASES,
    HOP_LABEL_PATTERNS_IDS,
    SENSITIVE_PHENOMENA_CASES,
    SENSITIVE_PHENOMENA_IDS,
    YANNAKAKIS_PRINCIPLE_CASES,
    YANNAKAKIS_PRINCIPLE_IDS,
)
from tests.gfql.ref.conftest import (
    _assert_parity,
    assert_node_membership,
    make_cg_graph_from_rows,
    run_chain_with_parity,
)


class TestYannakakisPrinciple:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids, required_edges, excluded_edges, exact_nodes, edge_count",
        YANNAKAKIS_PRINCIPLE_CASES,
        ids=YANNAKAKIS_PRINCIPLE_IDS,
    )
    def test_yannakakis_principle_matrix(
        self,
        node_rows,
        edge_rows,
        chain,
        where,
        include_ids,
        exclude_ids,
        required_edges,
        excluded_edges,
        exact_nodes,
        edge_count,
    ):
        _, result_nodes, result_edges = run_chain_with_parity(
            make_cg_graph_from_rows(node_rows, edge_rows), chain, where
        )
        if exact_nodes is not None:
            assert result_nodes == exact_nodes
        else:
            assert_node_membership(result_nodes, include_ids, exclude_ids)
        for edge in required_edges:
            assert edge in result_edges
        for edge in excluded_edges:
            assert edge not in result_edges
        if edge_count is not None:
            assert len(result_edges) == edge_count


class TestHopLabelingPatterns:
    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, exact_nodes, include_ids, exclude_ids",
        HOP_LABEL_PATTERNS_CASES,
        ids=HOP_LABEL_PATTERNS_IDS,
    )
    def test_hop_label_patterns_matrix(
        self, node_rows, edge_rows, chain, where, exact_nodes, include_ids, exclude_ids
    ):
        _, result_nodes, _ = run_chain_with_parity(make_cg_graph_from_rows(node_rows, edge_rows), chain, where)
        if exact_nodes is not None:
            assert result_nodes == exact_nodes
        else:
            assert_node_membership(result_nodes, include_ids, exclude_ids)

    def test_edge_hop_labels_consistent(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
        )

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


class TestSensitivePhenomena:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids, required_edges",
        SENSITIVE_PHENOMENA_CASES,
        ids=SENSITIVE_PHENOMENA_IDS,
    )
    def test_sensitive_phenomena_matrix(
        self, node_rows, edge_rows, chain, where, include_ids, exclude_ids, required_edges
    ):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _, result_nodes, result_edges = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)
        for edge in required_edges:
            assert edge in result_edges

    # --- Filter Cascades ---

    def test_filter_eliminates_all_at_step(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1, "type": "normal"}, {"id": "b", "v": 5, "type": "normal"}, {"id": "c", "v": 10, "type": "normal"}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
        )

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
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
        )

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

    def test_diamond_shared_edges(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 6}, {"id": "d", "v": 10}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "d"}, {"src": "a", "dst": "c"}, {"src": "c", "dst": "d"}],
        )

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

class TestNodeEdgeMatchFilters:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, include_ids, exclude_ids",
        [
            (
                [
                    {"id": "a", "v": 1, "type": "source"},
                    {"id": "b", "v": 10, "type": "target"},
                    {"id": "c", "v": 20, "type": "other"},
                ],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=1), n(name="end")],
                {"b"},
                {"c"},
            ),
            (
                [
                    {"id": "a", "v": 1, "type": "good"},
                    {"id": "b", "v": 5, "type": "bad"},
                    {"id": "c", "v": 10, "type": "target"},
                ],
                [{"src": "a", "dst": "c"}, {"src": "b", "dst": "c"}],
                [n(name="start"), e_forward(source_node_match={"type": "good"}, min_hops=1, max_hops=1), n({"id": "c"}, name="end")],
                {"a"},
                {"b"},
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 10}, {"id": "c", "v": 20}],
                [{"src": "a", "dst": "b", "type": "friend"}, {"src": "a", "dst": "c", "type": "enemy"}],
                [n({"id": "a"}, name="start"), e_forward(edge_match={"type": "friend"}, min_hops=1, max_hops=1), n(name="end")],
                {"b"},
                {"c"},
            ),
            (
                [
                    {"id": "a", "v": 1, "type": "source"},
                    {"id": "b", "v": 5, "type": "target"},
                    {"id": "c", "v": 10, "type": "target"},
                ],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=2), n(name="end")],
                {"b", "c"},
                set(),
            ),
            (
                [
                    {"id": "a", "v": 1, "role": "sender", "type": "node"},
                    {"id": "b", "v": 5, "role": "receiver", "type": "node"},
                    {"id": "c", "v": 10, "role": "none", "type": "target"},
                    {"id": "d", "v": 15, "role": "none", "type": "other"},
                ],
                [{"src": "a", "dst": "c"}, {"src": "b", "dst": "c"}, {"src": "a", "dst": "d"}],
                [
                    n(name="start"),
                    e_forward(source_node_match={"role": "sender"}, destination_node_match={"type": "target"}, min_hops=1, max_hops=1),
                    n(name="end"),
                ],
                {"a", "c"},
                {"b", "d"},
            ),
            (
                [{"id": "a", "v": 1}, {"id": "b", "v": 5}, {"id": "c", "v": 10}, {"id": "d", "v": 15}],
                [
                    {"src": "a", "dst": "b", "quality": "good"},
                    {"src": "b", "dst": "c", "quality": "good"},
                    {"src": "b", "dst": "d", "quality": "bad"},
                ],
                [n({"id": "a"}, name="start"), e_forward(edge_match={"quality": "good"}, min_hops=1, max_hops=2), n(name="end")],
                {"b", "c"},
                {"d"},
            ),
            (
                [
                    {"id": "a", "v": 1, "type": "source"},
                    {"id": "b", "v": 5, "type": "target"},
                    {"id": "c", "v": 10, "type": "target"},
                ],
                [{"src": "b", "dst": "a"}, {"src": "b", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_undirected(destination_node_match={"type": "target"}, min_hops=1, max_hops=2), n(name="end")],
                {"b", "c"},
                set(),
            ),
        ],
        ids=[
            "destination_node_match_single_hop",
            "source_node_match_single_hop",
            "edge_match_single_hop",
            "destination_node_match_multi_hop",
            "combined_source_and_dest_match",
            "edge_match_multi_hop",
            "undirected_with_destination_match",
        ],
    )
    def test_node_edge_match_filters_matrix(self, node_rows, edge_rows, chain, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _, result_nodes, _ = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)


class TestWhereClauseConjunction:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids",
        [
            (
                [
                    {"id": "a", "x": 10, "y": 1},
                    {"id": "b", "x": 5, "y": 5},
                    {"id": "c", "x": 5, "y": 10},
                    {"id": "d", "x": 5, "y": 0},
                    {"id": "e", "x": 15, "y": 10},
                ],
                [
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                    {"src": "b", "dst": "d"},
                    {"src": "b", "dst": "e"},
                ],
                [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [compare(col("start", "x"), ">", col("end", "x")), compare(col("start", "y"), "<", col("end", "y"))],
                {"c"},
                {"d", "e"},
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 1, "z": 10},
                    {"id": "b", "x": 5, "y": 5, "z": 5},
                    {"id": "c", "x": 5, "y": 10, "z": 5},
                    {"id": "d", "x": 5, "y": 10, "z": 15},
                    {"id": "e", "x": 9, "y": 10, "z": 5},
                ],
                [
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                    {"src": "b", "dst": "d"},
                    {"src": "b", "dst": "e"},
                ],
                [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
                [
                    compare(col("start", "x"), "==", col("end", "x")),
                    compare(col("start", "y"), "<", col("end", "y")),
                    compare(col("start", "z"), ">", col("end", "z")),
                ],
                {"c"},
                {"d", "e"},
            ),
            (
                [
                    {"id": "a", "x": 10, "y": 1},
                    {"id": "b", "x": 7, "y": 5},
                    {"id": "c", "x": 5, "y": 10},
                    {"id": "d", "x": 5, "y": 0},
                ],
                [
                    {"src": "a", "dst": "b"},
                    {"src": "b", "dst": "c"},
                    {"src": "b", "dst": "d"},
                ],
                [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
                [compare(col("start", "x"), ">", col("end", "x")), compare(col("start", "y"), "<", col("end", "y"))],
                {"c"},
                {"d"},
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 5},
                    {"id": "b", "x": 3, "y": 7},
                    {"id": "c", "x": 7, "y": 3},
                ],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
                [compare(col("start", "x"), ">", col("end", "x")), compare(col("start", "y"), "<", col("end", "y"))],
                {"b"},
                {"c"},
            ),
            (
                [
                    {"id": "a", "x": 10, "y": 1},
                    {"id": "b", "x": 7, "y": 5},
                    {"id": "c", "x": 5, "y": 10},
                ],
                [{"src": "b", "dst": "a"}, {"src": "c", "dst": "b"}],
                [n({"id": "a"}, name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
                [compare(col("start", "x"), ">", col("end", "x")), compare(col("start", "y"), "<", col("end", "y"))],
                {"c"},
                set(),
            ),
        ],
        ids=[
            "conjunction_two_clauses_same_columns",
            "conjunction_three_clauses",
            "conjunction_multihop_single_edge_step",
            "conjunction_with_impossible_combination",
            "conjunction_undirected_multihop",
        ],
    )
    def test_conjunction_matrix(self, node_rows, edge_rows, chain, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _, result_nodes, _ = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids, required_edges, excluded_edges, allow_seed_or_empty",
        [
            (
                [
                    {"id": "a", "x": 5, "y": 1},
                    {"id": "b1", "x": 5, "y": 5},
                    {"id": "b2", "x": 9, "y": 5},
                    {"id": "c1", "x": 5, "y": 10},
                    {"id": "c2", "x": 5, "y": 0},
                ],
                [
                    {"src": "a", "dst": "b1"},
                    {"src": "a", "dst": "b2"},
                    {"src": "b1", "dst": "c1"},
                    {"src": "b1", "dst": "c2"},
                    {"src": "b2", "dst": "c1"},
                ],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "==", col("b", "x")), compare(col("a", "y"), "<", col("c", "y"))],
                {"b1", "c1"},
                {"b2", "c2"},
                set(),
                set(),
                False,
            ),
            (
                [{"id": "a", "x": 5, "y": 5}, {"id": "b", "x": 10, "y": 10}, {"id": "c", "x": 3, "y": 3}],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}],
                [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
                [compare(col("start", "x"), ">", col("end", "x")), compare(col("start", "y"), "<", col("end", "y"))],
                set(),
                {"b", "c"},
                set(),
                set(),
                True,
            ),
            (
                [{"id": "a", "x": 5, "y": 1}, {"id": "b1", "x": 5, "y": 5}, {"id": "b2", "x": 9, "y": 5}, {"id": "c", "x": 5, "y": 10}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "b1", "dst": "c"}, {"src": "b2", "dst": "c"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "==", col("b", "x")), compare(col("a", "y"), "<", col("c", "y"))],
                {"c", "b1"},
                {"b2"},
                set(),
                {("a", "b2")},
                False,
            ),
        ],
        ids=[
            "conjunction_adjacent_and_nonadjacent",
            "conjunction_empty_result",
            "conjunction_diamond_multiple_paths",
        ],
    )
    def test_conjunction_remaining_matrix(
        self,
        node_rows,
        edge_rows,
        chain,
        where,
        include_ids,
        exclude_ids,
        required_edges,
        excluded_edges,
        allow_seed_or_empty,
    ):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        result, result_nodes, result_edges = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)

        if allow_seed_or_empty:
            assert "a" in result_nodes or len(result_nodes) == 0, "empty or seed-only result"

        for edge in required_edges:
            assert edge in result_edges
        for edge in excluded_edges:
            assert edge not in result_edges

class TestWhereClauseNegation:

    @pytest.mark.parametrize(
        "node_rows, edge_rows, where, include_ids, exclude_ids",
        [
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 5}, {"id": "c", "x": 10}],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}],
                [compare(col("start", "x"), "!=", col("end", "x"))],
                {"c"},
                {"b"},
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 10},
                    {"id": "b", "x": 5, "y": 10},
                    {"id": "c", "x": 10, "y": 10},
                    {"id": "d", "x": 10, "y": 20},
                ],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "a", "dst": "d"}],
                [compare(col("start", "x"), "!=", col("end", "x")), compare(col("start", "y"), "==", col("end", "y"))],
                {"c"},
                {"b", "d"},
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 10},
                    {"id": "b", "x": 5, "y": 5},
                    {"id": "c", "x": 10, "y": 5},
                    {"id": "d", "x": 10, "y": 15},
                ],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "a", "dst": "d"}],
                [compare(col("start", "x"), "!=", col("end", "x")), compare(col("start", "y"), ">", col("end", "y"))],
                {"c"},
                {"b", "d"},
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 10},
                    {"id": "b", "x": 5, "y": 20},
                    {"id": "c", "x": 10, "y": 10},
                    {"id": "d", "x": 10, "y": 20},
                ],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}, {"src": "a", "dst": "d"}],
                [compare(col("start", "x"), "!=", col("end", "x")), compare(col("start", "y"), "!=", col("end", "y"))],
                {"d"},
                {"b", "c"},
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 5}, {"id": "c", "x": 5}],
                [{"src": "a", "dst": "b"}, {"src": "a", "dst": "c"}],
                [compare(col("start", "x"), "!=", col("end", "x"))],
                set(),
                {"b", "c"},
            ),
        ],
        ids=[
            "negation_simple",
            "negation_with_equality",
            "negation_with_inequality",
            "double_negation",
            "negation_all_match_empty_result",
        ],
    )
    def test_negation_single_hop_matrix(self, node_rows, edge_rows, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        chain = [n({"id": "a"}, name="start"), e_forward(), n(name="end")]
        _, result_nodes, _ = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, include_ids, exclude_ids",
        [
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 7}, {"id": "c", "x": 5}, {"id": "d", "x": 10}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "b", "dst": "d"}],
                {"d"},
                {"c"},
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b1", "x": 7}, {"id": "b2", "x": 8}, {"id": "b3", "x": 9}, {"id": "c1", "x": 5}, {"id": "c2", "x": 10}, {"id": "c3", "x": 5}],
                [
                    {"src": "a", "dst": "b1"},
                    {"src": "a", "dst": "b2"},
                    {"src": "a", "dst": "b3"},
                    {"src": "b1", "dst": "c1"},
                    {"src": "b2", "dst": "c2"},
                    {"src": "b3", "dst": "c3"},
                ],
                {"b2", "c2"},
                {"b1", "b3", "c1", "c3"},
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 10}, {"id": "c", "x": 5}],
                [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}],
                set(),
                {"c"},
            ),
        ],
        ids=[
            "negation_multihop",
            "negation_multiple_ends_some_match",
            "negation_conflict_start_end_same_value",
        ],
    )
    def test_negation_two_hop_matrix(self, node_rows, edge_rows, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        chain = [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]
        _, result_nodes, _ = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)

    @pytest.mark.parametrize(
        "node_rows, edge_rows, chain, where, include_ids, exclude_ids, required_edges, excluded_edges",
        [
            (
                [{"id": "a", "x": 5}, {"id": "b1", "x": 5}, {"id": "b2", "x": 10}, {"id": "c", "x": 15}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "b1", "dst": "c"}, {"src": "b2", "dst": "c"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "!=", col("b", "x"))],
                {"b2", "c"},
                {"b1"},
                set(),
                set(),
            ),
            (
                [
                    {"id": "a", "x": 5, "y": 10},
                    {"id": "b1", "x": 5, "y": 7},
                    {"id": "b2", "x": 9, "y": 7},
                    {"id": "c1", "x": 5, "y": 10},
                    {"id": "c2", "x": 5, "y": 20},
                ],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "b1", "dst": "c1"}, {"src": "b1", "dst": "c2"}, {"src": "b2", "dst": "c2"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "==", col("b", "x")), compare(col("a", "y"), "!=", col("c", "y"))],
                {"b1", "c2"},
                {"b2", "c1"},
                set(),
                set(),
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b1", "x": 5}, {"id": "b2", "x": 10}, {"id": "c", "x": 5}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "b1", "dst": "c"}, {"src": "b2", "dst": "c"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "!=", col("b", "x"))],
                {"b2", "c"},
                {"b1"},
                {("a", "b2")},
                {("a", "b1")},
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b1", "x": 5}, {"id": "b2", "x": 5}, {"id": "c", "x": 10}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "b1", "dst": "c"}, {"src": "b2", "dst": "c"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "!=", col("b", "x"))],
                set(),
                {"c", "b1", "b2"},
                set(),
                set(),
            ),
            (
                [{"id": "a", "x": 5, "y": 10}, {"id": "b1", "x": 5, "y": 7}, {"id": "b2", "x": 10, "y": 7}, {"id": "b3", "x": 5, "y": 7}, {"id": "c", "x": 10, "y": 10}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "a", "dst": "b3"}, {"src": "b1", "dst": "c"}, {"src": "b2", "dst": "c"}, {"src": "b3", "dst": "c"}],
                [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")],
                [compare(col("a", "x"), "!=", col("b", "x")), compare(col("a", "y"), "==", col("c", "y"))],
                {"b2", "c"},
                {"b1", "b3"},
                set(),
                set(),
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b1", "x": 5}, {"id": "b2", "x": 10}, {"id": "c", "x": 15}],
                [{"src": "a", "dst": "b1"}, {"src": "a", "dst": "b2"}, {"src": "c", "dst": "b1"}, {"src": "c", "dst": "b2"}],
                [n({"id": "a"}, name="a"), e_undirected(name="e1"), n(name="b"), e_undirected(name="e2"), n(name="c")],
                [compare(col("a", "x"), "!=", col("b", "x"))],
                {"b2", "c"},
                {"b1"},
                set(),
                set(),
            ),
        ],
        ids=[
            "negation_adjacent_steps",
            "negation_nonadjacent_with_equality_adjacent",
            "negation_diamond_one_path_valid",
            "negation_diamond_both_paths_fail",
            "negation_convergent_paths_different_intermediates",
            "negation_undirected_diamond",
        ],
    )
    def test_negation_remaining_matrix(
        self,
        node_rows,
        edge_rows,
        chain,
        where,
        include_ids,
        exclude_ids,
        required_edges,
        excluded_edges,
    ):
        graph = make_cg_graph_from_rows(node_rows, edge_rows)
        _, result_nodes, result_edges = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)
        for edge in required_edges:
            assert edge in result_edges
        for edge in excluded_edges:
            assert edge not in result_edges

    def test_negation_cycle_same_node_different_hops(self):
        graph = make_cg_graph_from_rows(
            [{"id": "a", "x": 5}, {"id": "b", "x": 10}, {"id": "c", "x": 5}],
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "c", "dst": "a"}],
        )

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

    @pytest.mark.parametrize(
        "node_rows, where, include_ids, exclude_ids",
        [
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 10}, {"id": "c", "x": 10}, {"id": "d", "x": 5}],
                [compare(col("a", "x"), "!=", col("b", "x")), compare(col("b", "x"), "==", col("c", "x"))],
                {"b", "c"},
                {"d"},
            ),
            (
                [{"id": "a", "x": 5}, {"id": "b", "x": 10}, {"id": "c", "x": 5}, {"id": "d", "x": 10}],
                [compare(col("a", "x"), "!=", col("b", "x")), compare(col("b", "x"), "!=", col("c", "x"))],
                {"b", "c"},
                {"d"},
            ),
        ],
        ids=[
            "negation_with_equality_conflicting_requirements",
            "negation_transitive_chain",
        ],
    )
    def test_negation_two_edge_transitive_matrix(self, node_rows, where, include_ids, exclude_ids):
        graph = make_cg_graph_from_rows(
            node_rows,
            [{"src": "a", "dst": "b"}, {"src": "b", "dst": "c"}, {"src": "b", "dst": "d"}],
        )
        chain = [n({"id": "a"}, name="a"), e_forward(name="e1"), n(name="b"), e_forward(name="e2"), n(name="c")]
        _, result_nodes, _ = run_chain_with_parity(graph, chain, where)
        assert_node_membership(result_nodes, include_ids, exclude_ids)
