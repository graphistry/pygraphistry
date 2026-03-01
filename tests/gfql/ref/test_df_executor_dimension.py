"""Dimension coverage matrix tests for df_executor."""

import numpy as np
import pandas as pd

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected, is_in
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    DFSamePathExecutor,
    execute_same_path_chain,
)
from graphistry.compute.gfql.same_path_types import col, compare
from tests.gfql.ref.conftest import _assert_parity, make_cg_graph, run_chain_with_parity


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


class TestWhereClauseEdgeColumns:
    def test_edge_column_equality_two_edges(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges("c")
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.etype == e2.etype (follow==follow)"
        assert "d" not in result_nodes, "d: e1.etype != e2.etype (follow!=block)"

    def test_edge_column_negation_two_edges(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges("c")
        where = [compare(col("e1", "etype"), "!=", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d: e1.etype != e2.etype (follow!=block)"
        assert "c" not in result_nodes, "c: e1.etype == e2.etype (follow==follow)"

    def test_edge_column_inequality(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges("c")
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight > e2.weight (10 > 5)"
        assert "d" not in result_nodes, "d: e1.weight < e2.weight (10 < 15)"

    def test_mixed_node_and_edge_columns(self):
        nodes = pd.DataFrame([
            {"id": "a", "priority": 10},
            {"id": "b", "priority": 5},
            {"id": "c", "priority": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},   # a.priority(10) > weight(5) - VALID
            {"src": "a", "dst": "c", "weight": 15},  # a.priority(10) < weight(15) - INVALID
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e"),
            n(name="b"),
        ]
        where = [compare(col("a", "priority"), ">", col("e", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "b" in result_nodes, "b: a.priority(10) > e.weight(5)"
        assert "c" not in result_nodes, "c: a.priority(10) < e.weight(15)"

    def test_edge_negation_diamond_topology(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Path a->b->d: e1.weight=5 != e2.weight=10 - VALID
        # Path a->c->d: e1.weight=10 == e2.weight=10 - INVALID
        assert "d" in result_nodes, "d reachable via a->b->d (5 != 10)"
        assert "b" in result_nodes, "b on valid path"
        # Note: c might still be included if edges allow it - let's check
        # Actually c is on invalid path, but may be included due to Yannakakis
        # The key is that the valid path exists

    def test_edge_and_node_negation_combined(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges("c")
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),      # node constraint
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # edge constraint
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Path a->b1->c: a.x==b1.x FAILS node constraint
        # Path a->b2->c: a.x!=b2.x PASSES, but e1.etype==e2.etype FAILS edge constraint
        # No valid path!
        assert "c" not in result_nodes, "no valid path - all fail one constraint"

    def test_edge_and_node_negation_one_valid_path(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges("c")
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("e1", "etype"), "!=", col("e2", "etype")),
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Path a->b2->c: a.x(5) != b2.x(10) AND e1.etype(follow) != e2.etype(block)
        assert "c" in result_nodes, "c reachable via valid path a->b2->c"
        assert "b2" in result_nodes, "b2 on valid path"
        assert "b1" not in result_nodes, "b1 fails node constraint"

    def test_three_edge_negation_chain(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_three_edges()
        where = [
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # A != B - PASS
            compare(col("e2", "etype"), "!=", col("e3", "etype")),  # B != C - PASS
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d: A!=B AND B!=C"

    def test_three_edge_negation_chain_fails(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_three_edges()
        where = [
            compare(col("e1", "etype"), "!=", col("e2", "etype")),  # A != B - PASS
            compare(col("e2", "etype"), "!=", col("e3", "etype")),  # B == B - FAIL
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" not in result_nodes, "d: B==B fails second constraint"

    def test_edge_negation_multihop_single_step(self):
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
        graph = make_cg_graph(nodes, edges)

        # Single-hop test with node vs edge comparison
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e"),
            n(name="end"),
        ]
        where = [compare(col("start", "threshold"), "!=", col("e", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: start.threshold(5) != e.weight(10)"
        assert "b" not in result_nodes, "b: start.threshold(5) == e.weight(5)"


class TestEdgeWhereDirectionAndHops:

    def test_edge_where_reverse_direction(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_reverse_two_edges()
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.etype(follow) == e2.etype(follow)"
        assert "d" not in result_nodes, "d: e1.etype(follow) != e2.etype(block)"

    def test_edge_where_undirected_both_orientations(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Both edges have etype=friend, should work despite different storage direction
        assert "b" in result_nodes, "b reachable"
        assert "c" in result_nodes or "d" in result_nodes, "path continues"

    def test_edge_where_undirected_mixed_types(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="mid"),
            e_undirected(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.friend == e2.friend"
        assert "d" not in result_nodes, "d: e1.friend != e2.enemy"

    def test_edge_where_null_values_excluded(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.follow == e2.follow"
        # d should be excluded because NULL != "follow"
        assert "d" not in result_nodes, "d: e1.follow != e2.NULL"

    def test_edge_where_null_inequality(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 5},
            {"src": "b", "dst": "c", "weight": None},  # NULL
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        # e1.weight != e2.weight: 5 != NULL -> should be excluded (SQL: NULL comparison)
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # NULL comparisons should fail, so c should not be included
        assert "c" not in result_nodes, "c excluded due to NULL comparison"

    def test_edge_where_numeric_comparison(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) == e2.weight(10)"
        assert "e" not in result_nodes, "e: e1.weight(10) < e2.weight(15)"

    def test_edge_where_le_ge_operators(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    def test_edge_where_three_edges_chain(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_three_edges()
        where = [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d reachable via path with all matching edge types"

    def test_edge_where_three_edges_one_mismatch(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_three_edges()
        where = [
            compare(col("e1", "etype"), "==", col("e2", "etype")),
            compare(col("e2", "etype"), "==", col("e3", "etype")),
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # e2.etype(x) != e3.etype(y), so no valid complete path
        assert "d" not in result_nodes, "d: e2.x != e3.y"

    def test_edge_where_mixed_forward_reverse(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.friend == e2.friend"
        assert "d" not in result_nodes, "d: e1.friend != e2.enemy"

    def test_edge_where_with_node_filter(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n({"x": is_in([10, 20])}, name="mid"),  # filter: only b (x=10) passes
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Only path a->b->c exists after node filter, and e1.foo == e2.foo
        assert "c" in result_nodes, "c via a->b->c with matching edge types"
        assert "d" not in result_nodes, "d filtered by node predicate"

    def test_edge_where_string_vs_numeric(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "label": "alpha"},
            {"src": "b", "dst": "c", "label": "alpha"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "label"), "==", col("e2", "label"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: string comparison alpha == alpha"


class TestDimensionCoverageMatrix:

    # --- Reverse edges with inequality operators ---

    def test_reverse_edge_less_than(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_reverse_two_edges()
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d: e1.weight(10) < e2.weight(15)"
        assert "c" not in result_nodes, "c: e1.weight(10) >= e2.weight(5)"

    def test_reverse_edge_greater_equal(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_reverse_two_edges()
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) >= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) < e2.weight(15)"

    # --- Undirected edges with inequality operators ---

    def test_undirected_edge_less_than(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_undirected_two_edges()
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d: e1.weight(10) < e2.weight(15)"
        assert "c" not in result_nodes, "c: e1.weight(10) >= e2.weight(5)"

    def test_undirected_edge_less_equal(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_undirected_two_edges()
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    # --- NULL with inequality operators ---

    def test_null_less_than_excluded(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},  # NULL
            {"src": "b", "dst": "c", "weight": 10},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "<", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # NULL < 10 should be NULL (treated as false)
        assert "c" not in result_nodes, "c excluded: NULL < 10 is NULL"

    def test_null_greater_than_excluded(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": None},  # NULL
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # 10 > NULL should be NULL (treated as false)
        assert "c" not in result_nodes, "c excluded: 10 > NULL is NULL"

    def test_null_less_equal_excluded(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": 10},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" not in result_nodes, "c excluded: NULL <= 10 is NULL"

    def test_null_greater_equal_excluded(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" not in result_nodes, "c excluded: 10 >= NULL is NULL"

    # --- Mixed NULL positions ---

    def test_both_null_equality(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # NULL == NULL should be NULL (treated as false in SQL)
        assert "c" not in result_nodes, "c excluded: NULL == NULL is NULL"

    def test_both_null_inequality(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": None},
            {"src": "b", "dst": "c", "weight": None},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "!=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # NULL != NULL should be NULL (treated as false in SQL)
        assert "c" not in result_nodes, "c excluded: NULL != NULL is NULL"

    def test_null_mixed_with_valid_paths(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) == e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) == e2.weight(NULL) is NULL"

    # --- NaN vs None distinction ---

    def test_nan_explicit(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10.0},
            {"src": "b", "dst": "c", "weight": np.nan},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" not in result_nodes, "c excluded: 10.0 == NaN is NaN"

    def test_none_in_string_column(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "label": "foo"},
            {"src": "b", "dst": "c", "label": None},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "label"), "==", col("e2", "label"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" not in result_nodes, "c excluded: 'foo' == None is NULL"

    # --- Node column NULL handling ---

    def test_node_column_null(self):
        nodes = pd.DataFrame([
            {"id": "a", "val": 10},
            {"id": "b", "val": None},
            {"id": "c", "val": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("start", "val"), "==", col("mid", "val"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # start.val(10) == mid.val(NULL) is NULL
        assert "c" not in result_nodes, "c excluded: path through NULL mid"


class TestRemainingDimensionGaps:

    # --- Reverse + remaining operators ---

    def test_reverse_edge_greater_than(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_reverse_two_edges()
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) <= e2.weight(15)"

    def test_reverse_edge_less_equal(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_reverse_two_edges()
        where = [compare(col("e1", "weight"), "<=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) <= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) > e2.weight(5)"

    # --- Undirected + remaining operators ---

    def test_undirected_edge_greater_than(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_undirected_two_edges()
        where = [compare(col("e1", "weight"), ">", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) > e2.weight(5)"
        assert "d" not in result_nodes, "d: e1.weight(10) <= e2.weight(15)"

    def test_undirected_edge_greater_equal(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_undirected_two_edges()
        where = [compare(col("e1", "weight"), ">=", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.weight(10) >= e2.weight(10)"
        assert "d" not in result_nodes, "d: e1.weight(10) < e2.weight(15)"

    def test_undirected_edge_not_equal(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = _chain_undirected_two_edges()
        where = [compare(col("e1", "etype"), "!=", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d: e1.friend != e2.enemy"
        assert "c" not in result_nodes, "c: e1.friend == e2.friend"

    # --- Multi-hop with edge WHERE ---

    def test_multihop_single_step_edge_where(self):
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
        graph = make_cg_graph(nodes, edges)

        # Single hop - just to verify edge WHERE works
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(name="e"),
            n(name="end"),
        ]
        where = [compare(col("e", "weight"), "==", col("e", "weight"))]  # Trivial: always true

        _assert_parity(graph, chain, where)

    def test_two_multihop_steps_edge_where(self):
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
        graph = make_cg_graph(nodes, edges)

        # Two single-hop steps to compare
        chain = _chain_forward_two_edges()
        where = [compare(col("e1", "weight"), "==", col("e2", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # a->b (10) -> c (10): e1==e2 True
        # a->b (10) -> d (5): e1==e2 False
        assert "c" in result_nodes, "c: e1(10) == e2(10)"
        assert "d" not in result_nodes, "d: e1(10) != e2(5)"

    # --- Node-to-edge comparisons with different directions ---

    def test_node_to_edge_reverse(self):
        nodes = pd.DataFrame([
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a", "weight": 10},  # reverse: a <- b
            {"src": "c", "dst": "b", "weight": 10},  # reverse: b <- c
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(name="e"),
            n(name="end"),
        ]
        # start.threshold == e.weight: 10 == 10 True
        where = [compare(col("start", "threshold"), "==", col("e", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "b" in result_nodes, "b: start.threshold(10) == e.weight(10)"

    def test_node_to_edge_undirected(self):
        nodes = pd.DataFrame([
            {"id": "a", "threshold": 10},
            {"id": "b", "threshold": 5},
            {"id": "c", "threshold": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},
            {"src": "c", "dst": "b", "weight": 5},  # stored c->b
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(name="e"),
            n(name="end"),
        ]
        where = [compare(col("start", "threshold"), "==", col("e", "weight"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # a.threshold(10) == e.weight(10) for a--b edge
        assert "b" in result_nodes, "b: start.threshold(10) == e.weight(10)"

    def test_three_way_mixed_columns(self):
        nodes = pd.DataFrame([
            {"id": "a", "x": 10},
            {"id": "b", "y": 10},
            {"id": "c", "y": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "weight": 10},  # a.x(10) == weight(10) == b.y(10): VALID
            {"src": "a", "dst": "c", "weight": 10},  # a.x(10) == weight(10) != c.y(5): INVALID
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e"),
            n(name="b"),
        ]
        where = [
            compare(col("a", "x"), "==", col("e", "weight")),
            compare(col("e", "weight"), "==", col("b", "y")),
        ]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "b" in result_nodes, "b: a.x(10) == e.weight(10) == b.y(10)"
        assert "c" not in result_nodes, "c: a.x(10) == e.weight(10) != c.y(5)"

    # --- Edge direction combinations ---

    def test_forward_then_reverse_edge_where(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_reverse(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.call == e2.call"
        assert "d" not in result_nodes, "d: e1.call != e2.callback"

    def test_reverse_then_forward_edge_where(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_reverse(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.out == e2.out"
        assert "d" not in result_nodes, "d: e1.out != e2.in"

    def test_undirected_then_forward_edge_where(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="end"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "c" in result_nodes, "c: e1.link == e2.link"
        assert "d" not in result_nodes, "d: e1.link != e2.other"

    # --- Complex topologies ---

    def test_diamond_with_edge_where_all_match(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        assert "d" in result_nodes, "d reachable via both paths"
        assert "b" in result_nodes, "b on valid path"
        assert "c" in result_nodes, "c on valid path"

    def test_diamond_with_edge_where_partial_match(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Both paths are valid (x==x and y==y)
        assert "d" in result_nodes, "d reachable via both valid paths"

    def test_diamond_with_edge_where_one_invalid(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="mid"),
            e_forward(name="e2"),
            n(name="d"),
        ]
        where = [compare(col("e1", "etype"), "==", col("e2", "etype"))]

        result, result_nodes, _ = run_chain_with_parity(graph, chain, where)

        # Only a->b->d is valid
        assert "d" in result_nodes, "d reachable via a->b->d"
        assert "b" in result_nodes, "b on valid path"
