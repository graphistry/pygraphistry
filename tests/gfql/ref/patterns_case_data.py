"""Shared case data for GFQL pattern-heavy df_executor tests."""

import numpy as np
import pandas as pd

from graphistry.compute import n, e_forward, e_reverse, e_undirected
from graphistry.compute.gfql.same_path_types import col, compare


def _nodes_v(*rows):
    return [{"id": node_id, "v": value} for node_id, value in rows]


def _nodes_attrs(*rows):
    return [{"id": node_id, **attrs} for node_id, attrs in rows]


def _edges(*rows):
    return [{"src": src, "dst": dst} for src, dst in rows]


def _cmp(a_alias, a_col, op, b_alias, b_col):
    return compare(col(a_alias, a_col), op, col(b_alias, b_col))


IMPOSSIBLE_CONSTRAINT_CASES = [
    (
        _nodes_v(("a", 5), ("b", 10), ("c", 3)),
        _edges(("a", "b"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v"), _cmp("start", "v", ">", "end", "v")],
    ),
    (
        _nodes_v(("a", 5), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
        [_cmp("start", "v", "==", "end", "v"), _cmp("start", "v", "!=", "end", "v")],
    ),
    (
        _nodes_v(("a", 5), ("b", 10), ("c", 3)),
        _edges(("a", "b"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
        [_cmp("start", "v", "<=", "end", "v"), _cmp("start", "v", ">", "end", "v")],
    ),
    (
        _nodes_v(("a", 100), ("b", 50), ("c", 10)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid"), e_forward(), n({"id": "c"}, name="end")],
        [_cmp("start", "v", "<", "mid", "v")],
    ),
    (
        _nodes_v(("a", 100), ("b", 50), ("c", 25), ("d", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
    (
        _nodes_attrs(("a", {"v": 5, "w": 10}), ("b", {"v": 10, "w": 5}), ("c", {"v": 3, "w": 20})),
        _edges(("a", "b"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v"), _cmp("start", "w", "<", "end", "w")],
    ),
    (
        _nodes_v(("a", 1), ("b", 100), ("c", 50)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid"), e_forward(), n({"id": "c"}, name="end")],
        [_cmp("mid", "v", "<", "end", "v")],
    ),
    (
        _nodes_v(("a", 100), ("b", 50), ("c", 10)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid"), e_forward(), n({"id": "c"}, name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
]

IMPOSSIBLE_CONSTRAINT_IDS = [
    "contradictory_lt_gt_same_column",
    "contradictory_eq_neq_same_column",
    "contradictory_lte_gt_same_column",
    "no_paths_satisfy_predicate",
    "multihop_no_valid_endpoints",
    "contradictory_on_different_columns",
    "chain_with_impossible_intermediate",
    "non_adjacent_impossible_constraint",
]


FIVE_WHYS_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("x", 100), ("y", 200)),
        _edges(("b", "a"), ("c", "b"), ("x", "y")),
        [n({"id": "a"}, name="start"), e_reverse(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        {"x", "y"},
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("d", 15), ("e", 100), ("f", 200)),
        _edges(("b", "a"), ("c", "b"), ("d", "b"), ("f", "e")),
        [n({"id": "a"}, name="start"), e_reverse(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"c", "d"},
        {"e", "f"},
    ),
    (
        _nodes_v(("a", 1000), ("b", 1), ("c", 2), ("d", 3)),
        _edges(("a", "b"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 100), ("c", 2)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid"), e_forward(), n(name="end")],
        [_cmp("mid", "v", "<", "end", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 10), ("b", 20), ("c", 30), ("z", 5)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c"},
        {"z"},
    ),
    (
        _nodes_attrs(("a", {"v": 1, "w": 100}), ("b", {"v": None, "w": None}), ("c", {"v": 10, "w": 10})),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 10), ("c", 5), ("d", 15), ("e", 20)),
        _edges(("a", "b"), ("a", "c"), ("a", "d"), ("b", "e"), ("c", "e"), ("d", "e")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"e"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("d", 20)),
        _edges(("a", "b"), ("b", "c"), ("c", "d"), ("a", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c", "d"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("c", "b")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"c"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("d", 20)),
        _edges(("b", "a"), ("c", "b"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_undirected(), n(name="mid1"), e_reverse(), n(name="mid2"), e_undirected(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 100), ("b", 50), ("c", 25), ("d", 10)),
        _edges(("b", "a"), ("c", "b"), ("d", "c")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=1, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        set(),
    ),
]

FIVE_WHYS_IDS = [
    "reverse_multihop_with_unreachable_intermediate",
    "reverse_multihop_asymmetric_fanout",
    "aggressive_where_empties_mid_pass",
    "where_eliminates_all_intermediates",
    "non_adjacent_where_references_unreached_value",
    "non_adjacent_multihop_value_comparison",
    "diamond_convergent_multihop_where",
    "parallel_paths_different_lengths",
    "undirected_multihop_bidirectional_traversal",
    "undirected_reverse_mixed_chain",
    "undirected_multihop_with_aggressive_where",
]


MIN_HOPS_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"c"},
        set(),
        2,
    ),
    (
        _nodes_v(("a", 1), ("b", 2), ("c", 3), ("d", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=3, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"d"},
        set(),
        3,
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("d", 15)),
        _edges(("a", "b"), ("b", "c"), ("c", "d"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"c"},
        set(),
        None,
    ),
    (
        _nodes_v(("a", 10), ("b", 5), ("c", 1)),
        _edges(("b", "a"), ("c", "b")),
        [n({"id": "a"}, name="start"), e_reverse(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", ">", "end", "v")],
        {"c"},
        set(),
        None,
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("c", "b")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"c"},
        set(),
        None,
    ),
    (
        _nodes_v(("start", 0), ("mid1", 1), ("mid2", 2), ("end", 100)),
        _edges(("start", "mid1"), ("mid1", "mid2"), ("mid2", "end")),
        [n({"id": "start"}, name="s"), e_forward(min_hops=3, max_hops=3), n(name="e")],
        [_cmp("s", "v", "<", "e", "v")],
        {"end"},
        set(),
        3,
    ),
    (
        _nodes_v(("start", 0), ("a", 1), ("b", 2), ("end", 10), ("x", 100)),
        _edges(("start", "a"), ("a", "b"), ("b", "end"), ("start", "x")),
        [n({"id": "start"}, name="s"), e_forward(min_hops=3, max_hops=3), n(name="e")],
        [_cmp("s", "v", "<", "e", "v")],
        {"end"},
        {"x"},
        None,
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10), ("d", 15)),
        _edges(("a", "b"), ("c", "b"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid1"), e_reverse(), n(name="mid2"), e_forward(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"d"},
        set(),
        None,
    ),
]

MIN_HOPS_IDS = [
    "min_hops_2_linear_chain",
    "min_hops_3_long_chain",
    "min_hops_equals_max_hops_exact_path",
    "min_hops_reverse_chain",
    "min_hops_undirected_chain",
    "min_hops_sparse_critical_intermediate",
    "min_hops_with_branch_not_taken",
    "min_hops_mixed_directions",
]


MULTIPLE_PATH_LENGTH_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("b", "c"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 2), ("c", 3), ("d", 10)),
        _edges(("a", "d"), ("a", "b"), ("b", "d"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=2, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c", "d"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 2), ("c", 3), ("d", 10)),
        _edges(("a", "d"), ("a", "b"), ("b", "d"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=3, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c", "d"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "a")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=3, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<=", "end", "v")],
        {"a", "b", "c"},
        set(),
    ),
    (
        _nodes_v(("a", 1), ("x", 2), ("y", 3), ("z", 4), ("d", 10)),
        _edges(("a", "x"), ("x", "d"), ("a", "y"), ("y", "z"), ("z", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=3, max_hops=3), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"y", "z", "d"},
        {"x"},
    ),
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("a", "b"), ("b", "c"), ("a", "c")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        {"b", "c"},
        set(),
    ),
    (
        _nodes_v(("a", 10), ("b", 5), ("c", 1)),
        _edges(("b", "a"), ("c", "b"), ("c", "a")),
        [n({"id": "a"}, name="start"), e_reverse(min_hops=2, max_hops=2), n(name="end")],
        [_cmp("start", "v", ">", "end", "v")],
        {"b", "c"},
        set(),
    ),
]

MULTIPLE_PATH_LENGTH_IDS = [
    "diamond_with_shortcut",
    "triple_paths_different_lengths",
    "triple_paths_exact_min_hops_3",
    "cycle_multiple_path_lengths",
    "parallel_paths_with_min_hops_filter",
    "undirected_multiple_routes",
    "reverse_multiple_path_lengths",
]


LONGER_PATH_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 3), ("d", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "d")),
        [n(name="a"), e_forward(), n(name="b"), e_forward(), n(name="c"), e_forward(), n(name="d")],
        [_cmp("a", "v", "<", "d", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 3), ("c", 5), ("d", 7), ("e", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")),
        [n(name="a"), e_forward(), n(name="b"), e_forward(), n(name="c"), e_forward(), n(name="d"), e_forward(), n(name="e")],
        [_cmp("a", "v", "<", "c", "v"), _cmp("c", "v", "<", "e", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 3), ("c", 5), ("d", 7), ("e", 10)),
        _edges(("a", "b"), ("b", "c"), ("c", "d"), ("d", "e")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="mid"), e_forward(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
        set(),
        set(),
    ),
    (
        _nodes_v(("a", 1), ("b", 3), ("c", 5), ("d1", 10), ("d2", 0)),
        _edges(("a", "b"), ("b", "c"), ("c", "d1"), ("c", "d2")),
        [n(name="a"), e_forward(), n(name="b"), e_forward(), n(name="c"), e_forward(), n(name="d")],
        [_cmp("a", "v", "<", "d", "v")],
        {"d1"},
        {"d2"},
    ),
]

LONGER_PATH_IDS = [
    "four_node_chain",
    "five_node_chain_multiple_where",
    "long_chain_with_multihop",
    "long_chain_filters_partial_path",
]


MIXED_DIRECTION_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 3), ("d", 10)),
        _edges(("a", "b"), ("c", "b"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid1"), e_reverse(), n(name="mid2"), e_forward(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
    (
        _nodes_v(("a", 10), ("b", 5), ("c", 7), ("d", 1)),
        _edges(("b", "a"), ("b", "c"), ("d", "c")),
        [n({"id": "a"}, name="start"), e_reverse(), n(name="mid1"), e_forward(), n(name="mid2"), e_reverse(), n(name="end")],
        [_cmp("start", "v", ">", "end", "v")],
    ),
    (
        _nodes_v(("a", 1), ("b", 3), ("c", 5), ("d", 7), ("e", 10)),
        _edges(("a", "b"), ("b", "c"), ("d", "c"), ("e", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="mid"), e_reverse(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
]

MIXED_DIRECTION_IDS = [
    "forward_reverse_forward",
    "reverse_forward_reverse",
    "mixed_with_multihop",
]


UNDIRECTED_BUG_PATTERN_CASES = [
    (
        _nodes_v(("a", 1), ("b", 5), ("c", 10)),
        _edges(("b", "a"), ("c", "b")),
        [n({"id": "a"}, name="start"), e_undirected(), n(name="mid"), e_undirected(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
    (
        _nodes_attrs(("a", {"v": 1, "w": 10}), ("b", {"v": 5, "w": 5}), ("c", {"v": 10, "w": 1})),
        _edges(("b", "a"), ("c", "b")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v"), _cmp("start", "w", ">", "end", "w")],
    ),
    (
        _nodes_v(("a", 1), ("b", 2), ("c", 3), ("d", 4)),
        _edges(("a", "b"), ("c", "b"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(), n(name="mid"), e_undirected(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
    (
        _nodes_v(("a", 1), ("b", 2)),
        _edges(("a", "a"), ("a", "b")),
        [n({"id": "a"}, name="start"), e_undirected(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
    (
        _nodes_v(("a", 1), ("b", 2), ("c", 3), ("d", 4)),
        _edges(("b", "a"), ("b", "c"), ("d", "c")),
        [n({"id": "a"}, name="start"), e_undirected(), n(name="mid1"), e_reverse(), n(name="mid2"), e_undirected(), n(name="end")],
        [_cmp("start", "v", "<", "end", "v")],
    ),
]

UNDIRECTED_BUG_PATTERN_IDS = [
    "undirected_non_adjacent_where",
    "undirected_multiple_where",
    "mixed_directed_undirected_chain",
    "undirected_with_self_loop",
    "undirected_reverse_undirected_chain",
]


PREDICATE_PARITY_CASES = [
    (
        _nodes_attrs(("a", {"active": True}), ("b", {"active": False}), ("c", {"active": True})),
        [_cmp("start", "active", "==", "end", "active")],
    ),
    (
        _nodes_attrs(("a", {"active": False}), ("b", {"active": False}), ("c", {"active": True})),
        [_cmp("start", "active", "<", "end", "active")],
    ),
    (
        _nodes_attrs(
            ("a", {"ts": pd.Timestamp("2024-01-01")}),
            ("b", {"ts": pd.Timestamp("2024-06-01")}),
            ("c", {"ts": pd.Timestamp("2024-12-01")}),
        ),
        [_cmp("start", "ts", "<", "end", "ts")],
    ),
    (
        _nodes_attrs(("a", {"score": 1.5}), ("b", {"score": 2.7}), ("c", {"score": 1.5})),
        [_cmp("start", "score", "<=", "end", "score")],
    ),
    (
        _nodes_attrs(("a", {"v": 1.0}), ("b", {"v": np.nan}), ("c", {"v": 10.0})),
        [_cmp("start", "v", "<", "end", "v")],
    ),
]

PREDICATE_PARITY_IDS = [
    "boolean_comparison_eq",
    "boolean_comparison_lt",
    "datetime_comparison",
    "float_comparison_with_decimals",
    "nan_in_numeric_comparison",
]


PREDICATE_RESULT_CASES = [
    (
        _nodes_attrs(("a", {"name": "apple"}), ("b", {"name": "banana"}), ("c", {"name": "cherry"})),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "name", "<", "end", "name")],
        {"b", "c"},
    ),
    (
        _nodes_attrs(("a", {"tag": "important"}), ("b", {"tag": "normal"}), ("c", {"tag": "important"})),
        _edges(("a", "b"), ("b", "c")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=2), n(name="end")],
        [_cmp("start", "tag", "==", "end", "tag")],
        {"c"},
    ),
    (
        _nodes_attrs(
            ("a", {"created": pd.Timestamp("2024-01-01")}),
            ("b", {"created": pd.Timestamp("2024-03-01")}),
            ("c", {"created": pd.Timestamp("2024-06-01")}),
            ("d", {"created": pd.Timestamp("2024-09-01")}),
        ),
        _edges(("a", "b"), ("b", "c"), ("c", "d")),
        [n({"id": "a"}, name="start"), e_forward(min_hops=1, max_hops=3), n(name="end")],
        [_cmp("start", "created", "<", "end", "created")],
        {"b", "c", "d"},
    ),
]

PREDICATE_RESULT_IDS = [
    "string_lexicographic_comparison",
    "string_equality",
    "multihop_with_datetime_range",
]
