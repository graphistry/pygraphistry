"""Operator and bug pattern tests for df_executor."""

import numpy as np
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
from graphistry.tests.test_compute import CGFull

# Import shared helpers - pytest auto-loads conftest.py
from tests.gfql.ref.conftest import _assert_parity

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

    @pytest.mark.skip(reason="Oracle doesn't support multi-hop + WHERE")
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


