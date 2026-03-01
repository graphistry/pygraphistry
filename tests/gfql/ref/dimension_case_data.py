"""Shared case data for GFQL dimension coverage tests."""

from graphistry.compute import n, e_forward, e_reverse, e_undirected, is_in
from graphistry.compute.gfql.same_path_types import col, compare


def _nodes(*ids):
    return [{"id": node_id} for node_id in ids]


def _chain_forward_two_edges(end_alias: str = "end"):
    return [
        n({"id": "a"}, name="a"),
        e_forward(name="e1"),
        n(name="b"),
        e_forward(name="e2"),
        n(name=end_alias),
    ]


def _chain_reverse_two_edges():
    return [
        n({"id": "a"}, name="a"),
        e_reverse(name="e1"),
        n(name="b"),
        e_reverse(name="e2"),
        n(name="end"),
    ]


def _chain_undirected_two_edges():
    return [
        n({"id": "a"}, name="a"),
        e_undirected(name="e1"),
        n(name="b"),
        e_undirected(name="e2"),
        n(name="end"),
    ]


def _chain_forward_three_edges():
    return [
        n({"id": "a"}, name="a"),
        e_forward(name="e1"),
        n(name="b"),
        e_forward(name="e2"),
        n(name="c"),
        e_forward(name="e3"),
        n(name="d"),
    ]


EDGE_WHERE_MEMBERSHIP_CASES = [
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "b", "dst": "a", "etype": "follow"},
            {"src": "c", "dst": "b", "etype": "follow"},
            {"src": "d", "dst": "b", "etype": "block"},
        ],
        _chain_reverse_two_edges(),
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "friend"},
            {"src": "b", "dst": "c", "etype": "friend"},
            {"src": "b", "dst": "d", "etype": "enemy"},
        ],
        _chain_undirected_two_edges(),
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "follow"},
            {"src": "b", "dst": "c", "etype": "follow"},
            {"src": "b", "dst": "d", "etype": None},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c"),
        [
            {"src": "a", "dst": "b", "weight": 5},
            {"src": "b", "dst": "c", "weight": None},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "weight"), "!=", col("e2", "weight"))],
        set(),
        {"c"},
    ),
    (
        _nodes("a", "b", "c", "d", "e"),
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},
            {"src": "b", "dst": "d", "weight": 10},
            {"src": "b", "dst": "e", "weight": 15},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "weight"), ">", col("e2", "weight"))],
        {"c"},
        {"d", "e"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},
            {"src": "b", "dst": "d", "weight": 5},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "weight"), "<=", col("e2", "weight"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "friend"},
            {"src": "c", "dst": "b", "etype": "friend"},
            {"src": "d", "dst": "b", "etype": "enemy"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        [
            {"id": "a", "x": 1},
            {"id": "b", "x": 10},
            {"id": "c", "x": 20},
            {"id": "d", "x": 3},
        ],
        [
            {"src": "a", "dst": "b", "etype": "foo"},
            {"src": "a", "dst": "d", "etype": "foo"},
            {"src": "b", "dst": "c", "etype": "foo"},
            {"src": "d", "dst": "c", "etype": "bar"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n({"x": is_in([10, 20])}, name="mid"),
            e_forward(name="e2"),
            n(name="end"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c"),
        [
            {"src": "a", "dst": "b", "label": "alpha"},
            {"src": "b", "dst": "c", "label": "alpha"},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "label"), "==", col("e2", "label"))],
        {"c"},
        set(),
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "b", "dst": "c", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "x"},
        ],
        _chain_forward_three_edges(),
        [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ],
        {"d"},
        set(),
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "b", "dst": "c", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "y"},
        ],
        _chain_forward_three_edges(),
        [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ],
        set(),
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d", "e"),
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},
            {"src": "b", "dst": "d", "weight": 5},
            {"src": "d", "dst": "e", "weight": 10},
        ],
        _chain_forward_two_edges(),
        [compare(col("e1", "weight"), "==", col("e2", "weight"))],
        {"c"},
        {"d"},
    ),
    (
        [
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ],
        [
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},
        ],
        [n({"id": "a"}, name="start"), e_reverse(name="e"), n(name="end")],
        [compare(col("start", "threshold"), "==", col("e", "weight"))],
        {"b"},
        set(),
    ),
    (
        [
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ],
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},
        ],
        [n({"id": "a"}, name="start"), e_undirected(name="e"), n(name="end")],
        [compare(col("start", "threshold"), "==", col("e", "weight"))],
        {"b"},
        set(),
    ),
    (
        [{"id": "a", "x": 10}, {"id": "b", "y": 10}, {"id": "c", "y": 5}],
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "a", "dst": "c", "weight": 10},
        ],
        [n({"id": "a"}, name="a"), e_forward(name="e"), n(name="b")],
        [
            compare(col("a", "x"), "==", col("e", "weight")),
            compare(col("e", "weight"), "==", col("b", "y")),
        ],
        {"b"},
        {"c"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "call"},
            {"src": "c", "dst": "b", "etype": "call"},
            {"src": "d", "dst": "b", "etype": "callback"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "b", "dst": "a", "etype": "out"},
            {"src": "b", "dst": "c", "etype": "out"},
            {"src": "b", "dst": "d", "etype": "in"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "b", "dst": "a", "etype": "link"},
            {"src": "b", "dst": "c", "etype": "link"},
            {"src": "b", "dst": "d", "etype": "other"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"c"},
        {"d"},
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "x"},
            {"src": "b", "dst": "d", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "x"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"b", "c", "d"},
        set(),
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "y"},
            {"src": "b", "dst": "d", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "y"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"d"},
        set(),
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "x"},
            {"src": "a", "dst": "c", "etype": "y"},
            {"src": "b", "dst": "d", "etype": "x"},
            {"src": "c", "dst": "d", "etype": "x"},
        ],
        [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ],
        [compare(col("e1", "etype"), "==", col("e2", "etype"))],
        {"b", "d"},
        set(),
    ),
    (
        _nodes("a", "b", "c", "d"),
        [
            {"src": "a", "dst": "b", "etype": "friend"},
            {"src": "b", "dst": "c", "etype": "friend"},
            {"src": "d", "dst": "b", "etype": "enemy"},
        ],
        _chain_undirected_two_edges(),
        [compare(col("e1", "etype"), "!=", col("e2", "etype"))],
        {"d"},
        {"c"},
    ),
]

EDGE_WHERE_MEMBERSHIP_IDS = [
    "edge_where_reverse_direction",
    "edge_where_undirected_mixed_types",
    "edge_where_null_values_excluded",
    "edge_where_null_inequality",
    "edge_where_numeric_comparison",
    "edge_where_le_ge_operators",
    "edge_where_mixed_forward_reverse",
    "edge_where_with_node_filter",
    "edge_where_string_vs_numeric",
    "edge_where_three_edges_chain",
    "edge_where_three_edges_one_mismatch",
    "two_multihop_steps_edge_where",
    "node_to_edge_reverse",
    "node_to_edge_undirected",
    "three_way_mixed_columns",
    "forward_then_reverse_edge_where",
    "reverse_then_forward_edge_where",
    "undirected_then_forward_edge_where",
    "diamond_with_edge_where_all_match",
    "diamond_with_edge_where_partial_match",
    "diamond_with_edge_where_one_invalid",
    "undirected_edge_not_equal",
]

DIRECTIONAL_OPERATOR_CASES = [
    (
        _chain_reverse_two_edges(),
        "<",
        [
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},
            {"src": "d", "dst": "b", "weight": 15},
        ],
        "d",
        "c",
    ),
    (
        _chain_reverse_two_edges(),
        ">=",
        [
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},
            {"src": "d", "dst": "b", "weight": 15},
        ],
        "c",
        "d",
    ),
    (
        _chain_reverse_two_edges(),
        ">",
        [
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},
            {"src": "d", "dst": "b", "weight": 15},
        ],
        "c",
        "d",
    ),
    (
        _chain_reverse_two_edges(),
        "<=",
        [
            {"src": "b", "dst": "a", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},
            {"src": "d", "dst": "b", "weight": 5},
        ],
        "c",
        "d",
    ),
    (
        _chain_undirected_two_edges(),
        "<",
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},
            {"src": "b", "dst": "d", "weight": 15},
        ],
        "d",
        "c",
    ),
    (
        _chain_undirected_two_edges(),
        "<=",
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 10},
            {"src": "d", "dst": "b", "weight": 5},
        ],
        "c",
        "d",
    ),
    (
        _chain_undirected_two_edges(),
        ">",
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": 5},
            {"src": "d", "dst": "b", "weight": 15},
        ],
        "c",
        "d",
    ),
    (
        _chain_undirected_two_edges(),
        ">=",
        [
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 10},
            {"src": "b", "dst": "d", "weight": 15},
        ],
        "c",
        "d",
    ),
]

DIRECTIONAL_OPERATOR_IDS = [
    "reverse_lt",
    "reverse_ge",
    "reverse_gt",
    "reverse_le",
    "undirected_lt",
    "undirected_le",
    "undirected_gt",
    "undirected_ge",
]
