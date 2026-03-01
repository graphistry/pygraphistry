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
from tests.gfql.ref.conftest import _assert_parity, make_cg_graph, run_chain_checked


class TestP1OperatorsSingleHop:

    @pytest.fixture
    def basic_graph(self):
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
        return make_cg_graph(nodes, edges)

    def test_single_hop_eq(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "==", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # Only a->b satisfies 5 == 5
        assert "a" in set(result._nodes["id"])
        assert "b" in set(result._nodes["id"])

    def test_single_hop_neq(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "!=", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # a->c (5 != 10) and a->d (5 != 1) and c->d (10 != 1) satisfy
        result_ids = set(result._nodes["id"])
        assert "c" in result_ids, "c participates in valid paths"
        assert "d" in result_ids, "d participates in valid paths"

    def test_single_hop_lt(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # a->c (5 < 10) satisfies
        assert "c" in set(result._nodes["id"])

    def test_single_hop_gt(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), ">", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # a->d (5 > 1) and c->d (10 > 1) satisfy
        assert "d" in set(result._nodes["id"])

    def test_single_hop_lte(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # a->b (5 <= 5) and a->c (5 <= 10) satisfy
        result_ids = set(result._nodes["id"])
        assert "b" in result_ids
        assert "c" in result_ids

    def test_single_hop_gte(self, basic_graph):
        chain = [n(name="start"), e_forward(), n(name="end")]
        where = [compare(col("start", "v"), ">=", col("end", "v"))]
        result = run_chain_checked(basic_graph, chain, where)
        # a->b (5 >= 5) and a->d (5 >= 1) and c->d (10 >= 1) satisfy
        result_ids = set(result._nodes["id"])
        assert "b" in result_ids
        assert "d" in result_ids


# --- P2 tests: longer paths (4+ nodes)


class TestP2LongerPaths:

    def test_four_node_chain(self):
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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"])
        assert "d1" in result_ids, "d1 satisfies WHERE but excluded"
        assert "d2" not in result_ids, "d2 violates WHERE but included"


# --- P1 tests: operators × multihop systematic


class TestP1OperatorsMultihop:

    @pytest.fixture
    def multihop_graph(self):
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
        return make_cg_graph(nodes, edges)

    def test_multihop_eq(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_neq(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "!=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_lt(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_gt(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_lte(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)

    def test_multihop_gte(self, multihop_graph):
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">=", col("end", "v"))]
        _assert_parity(multihop_graph, chain, where)


# --- P1 tests: undirected + multihop


class TestP1UndirectedMultihop:

    def test_undirected_multihop_basic(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_multihop_bidirectional(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


# --- P1 tests: mixed direction chains


class TestP1MixedDirectionChains:

    def test_forward_reverse_forward(self):
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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="mid"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


# --- P2 tests: edge cases and boundary conditions


class TestP2EdgeCases:

    def test_single_node_graph(self):
        nodes = pd.DataFrame([{"id": "a", "v": 5}])
        edges = pd.DataFrame([{"src": "a", "dst": "a"}])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_disconnected_components(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_dense_graph(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_null_values_in_comparison(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": None},  # Null value
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_string_comparison(self):
        nodes = pd.DataFrame([
            {"id": "a", "name": "alice"},
            {"id": "b", "name": "bob"},
            {"id": "c", "name": "charlie"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "name"), "<", col("end", "name"))]

        _assert_parity(graph, chain, where)

    def test_multiple_where_all_operators(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 10},
            {"id": "b", "v": 5, "w": 5},
            {"id": "c", "v": 10, "w": 1},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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

    def test_three_consecutive_multihop_edges(self):
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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_multihop_diamond_graph(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestBugPatternMergeSuffix:

    def test_same_column_eq(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v == end.v: only c matches (v=5)
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_lt(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v < end.v: c matches (5 < 10), d doesn't (5 < 1 is false)
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_lte(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v <= end.v: c (5<=5) and d (5<=10) match
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_gt(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v > end.v: only c matches (5 > 1)
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_same_column_gte(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.v >= end.v: c (5>=5) and d (5>=1) match
        where = [compare(col("start", "v"), ">=", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestBugPatternUndirected:

    def test_undirected_non_adjacent_where(self):
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
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 10},
            {"id": "b", "v": 5, "w": 5},
            {"id": "c", "v": 10, "w": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_undirected_reverse_undirected_chain(self):
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
        graph = make_cg_graph(nodes, edges)

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

    def test_contradictory_lt_gt_same_column(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
            {"id": "c", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
            {"id": "c", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},  # Highest value
            {"id": "b", "v": 50},
            {"id": "c", "v": 10},   # Lowest value
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v - but a.v=100 is the highest, so impossible
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_contradictory_on_different_columns(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 5, "w": 10},
            {"id": "b", "v": 10, "w": 5},  # v is higher, w is lower
            {"id": "c", "v": 3, "w": 20},  # v is lower, w is higher
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # This would make mid.v > end.v impossible
            {"id": "c", "v": 50},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 100},  # Highest
            {"id": "b", "v": 50},
            {"id": "c", "v": 10},   # Lowest
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame({"id": [], "v": []})
        edges = pd.DataFrame({"src": [], "dst": []})
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_no_edges_with_constraints(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame({"src": [], "dst": []})
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestFiveWhysAmplification:

    # =========================================================================
    # Bug 1: Backward traversal join direction
    # Root cause: Direction semantics not tested at reachability level
    # =========================================================================

    def test_reverse_multihop_with_unreachable_intermediate(self):
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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),  # Exactly 2 hops
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
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
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # Intermediate - will be filtered (100 > 2)
            {"id": "c", "v": 2},    # End - would match if path existed
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "w": 100},
            {"id": "b", "v": None, "w": None},  # Intermediate, no v/w
            {"id": "c", "v": 10, "w": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},  # a->b exists
            {"src": "c", "dst": "b"},  # c->b exists (b<-c)
        ])
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v - but a.v=100 is highest, so no matches
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)


class TestMinHopsEdgeFiltering:

    def test_min_hops_2_linear_chain(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c should be reachable in exactly 2 hops"
        # Both edges should be in result (intermediate edge a->b is needed)
        edge_count = len(result._edges) if result._edges is not None else 0
        assert edge_count == 2, f"Both edges needed for 2-hop path, got {edge_count}"

    def test_min_hops_3_long_chain(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_ids, "d should be reachable in exactly 3 hops"
        edge_count = len(result._edges) if result._edges is not None else 0
        assert edge_count == 3, f"All 3 edges needed for 3-hop path, got {edge_count}"

    def test_min_hops_equals_max_hops_exact_path(self):
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
        graph = make_cg_graph(nodes, edges)

        # Exactly 2 hops - should get b and c, but NOT d (3 hops) or c via shortcut (1 hop)
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in exactly 2 hops via a->b->c"

    def test_min_hops_reverse_chain(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},  # Start
            {"id": "b", "v": 5},
            {"id": "c", "v": 1},   # End (reachable in 2 reverse hops)
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Reverse: a <- b
            {"src": "c", "dst": "b"},  # Reverse: b <- c
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in 2 reverse hops"

    def test_min_hops_undirected_chain(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids, "c reachable in 2 undirected hops"

    def test_min_hops_sparse_critical_intermediate(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "start"}, name="s"),
            e_forward(min_hops=3, max_hops=3),
            n(name="e"),
        ]
        where = [compare(col("s", "v"), "<", col("e", "v"))]

        result = run_chain_checked(graph, chain, where)
        assert result._nodes is not None and len(result._nodes) > 0, "Should find the path"
        assert result._edges is not None and len(result._edges) == 3, "All 3 edges are critical"

    def test_min_hops_with_branch_not_taken(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "start"}, name="s"),
            e_forward(min_hops=3, max_hops=3),
            n(name="e"),
        ]
        where = [compare(col("s", "v"), "<", col("e", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "end" in result_ids
        assert "x" not in result_ids, "Dead end should not be in results"

    def test_min_hops_mixed_directions(self):
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
        graph = make_cg_graph(nodes, edges)

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

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_ids, "Should find path a->b<-c->d"


class TestMultiplePathLengths:

    def test_diamond_with_shortcut(self):
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
        graph = make_cg_graph(nodes, edges)

        # min_hops=2 should still include the 2-hop path a->b->c
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is intermediate on valid 2-hop path"
        assert "c" in result_ids, "c is endpoint of valid 2-hop path"

    def test_triple_paths_different_lengths(self):
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
        graph = make_cg_graph(nodes, edges)

        # Test min_hops=2: should include 2-hop and 3-hop paths
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is on 2-hop and 3-hop paths"
        assert "c" in result_ids, "c is on 3-hop path"
        assert "d" in result_ids, "d is endpoint"

    def test_triple_paths_exact_min_hops_3(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only 3-hop path a->b->c->d should be included
        assert "b" in result_ids, "b is on 3-hop path"
        assert "c" in result_ids, "c is on 3-hop path"
        assert "d" in result_ids, "d is endpoint of 3-hop path"

    def test_cycle_multiple_path_lengths(self):
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
        graph = make_cg_graph(nodes, edges)

        # 3-hop path a->b->c->a exists
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        # start.v < end.v would be 1 < 1 = False, so use <=
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # All nodes on cycle should be included
        assert "a" in result_ids, "a is start and end of 3-hop cycle"
        assert "b" in result_ids, "b is on cycle"
        assert "c" in result_ids, "c is on cycle"

    def test_parallel_paths_with_min_hops_filter(self):
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
        graph = make_cg_graph(nodes, edges)

        # min_hops=3 should only include the y->z->d path
        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=3, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "y" in result_ids, "y is on 3-hop path"
        assert "z" in result_ids, "z is on 3-hop path"
        assert "d" in result_ids, "d is endpoint"
        # x should NOT be in results (only on 2-hop path)
        assert "x" not in result_ids, "x is only on 2-hop path, excluded by min_hops=3"

    def test_undirected_multiple_routes(self):
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
        graph = make_cg_graph(nodes, edges)

        # Undirected with min_hops=2
        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        # 2-hop path a-b-c should be found
        assert "b" in result_ids, "b is on 2-hop undirected path"
        assert "c" in result_ids, "c is endpoint of 2-hop path"

    def test_reverse_multiple_path_lengths(self):
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
        graph = make_cg_graph(nodes, edges)

        # Reverse with min_hops=2
        chain = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids, "b is on 2-hop reverse path"
        assert "c" in result_ids, "c is endpoint of 2-hop reverse path"


class TestPredicateTypes:

    def test_boolean_comparison_eq(self):
        nodes = pd.DataFrame([
            {"id": "a", "active": True},
            {"id": "b", "active": False},
            {"id": "c", "active": True},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.active == end.active (True == True for c)
        where = [compare(col("start", "active"), "==", col("end", "active"))]

        _assert_parity(graph, chain, where)

    def test_boolean_comparison_lt(self):
        nodes = pd.DataFrame([
            {"id": "a", "active": False},
            {"id": "b", "active": False},
            {"id": "c", "active": True},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.active < end.active (False < True for c)
        where = [compare(col("start", "active"), "<", col("end", "active"))]

        _assert_parity(graph, chain, where)

    def test_datetime_comparison(self):
        nodes = pd.DataFrame([
            {"id": "a", "ts": pd.Timestamp("2024-01-01")},
            {"id": "b", "ts": pd.Timestamp("2024-06-01")},
            {"id": "c", "ts": pd.Timestamp("2024-12-01")},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.ts < end.ts (all nodes have later timestamps)
        where = [compare(col("start", "ts"), "<", col("end", "ts"))]

        _assert_parity(graph, chain, where)

    def test_float_comparison_with_decimals(self):
        nodes = pd.DataFrame([
            {"id": "a", "score": 1.5},
            {"id": "b", "score": 2.7},
            {"id": "c", "score": 1.5},  # Same as a
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.score <= end.score
        where = [compare(col("start", "score"), "<=", col("end", "score"))]

        _assert_parity(graph, chain, where)

    def test_nan_in_numeric_comparison(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1.0},
            {"id": "b", "v": np.nan},  # NaN
            {"id": "c", "v": 10.0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Comparisons with NaN should be False
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

    def test_string_lexicographic_comparison(self):
        nodes = pd.DataFrame([
            {"id": "a", "name": "apple"},
            {"id": "b", "name": "banana"},
            {"id": "c", "name": "cherry"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # Lexicographic: "apple" < "banana" < "cherry"
        where = [compare(col("start", "name"), "<", col("end", "name"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids  # apple < banana
        assert "c" in result_ids  # apple < cherry

    def test_string_equality(self):
        nodes = pd.DataFrame([
            {"id": "a", "tag": "important"},
            {"id": "b", "tag": "normal"},
            {"id": "c", "tag": "important"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        # start.tag == end.tag (only c matches)
        where = [compare(col("start", "tag"), "==", col("end", "tag"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_ids  # "important" == "important"
        # Note: 'b' IS included because it's an intermediate node in the valid path a→b→c
        # The executor returns ALL nodes participating in valid paths, not just endpoints

    def test_neq_with_nulls(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": None},
            {"id": "c", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = make_cg_graph(nodes, edges)

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

    def test_multihop_with_datetime_range(self):
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
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=3),
            n(name="end"),
        ]
        # All nodes created after start
        where = [compare(col("start", "created"), "<", col("end", "created"))]

        result = run_chain_checked(graph, chain, where)
        result_ids = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_ids
        assert "c" in result_ids
        assert "d" in result_ids


class TestNonAdjacentValueMode:
    def test_value_mode_matches_baseline(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 1},
            {"id": "c", "v": 1},
            {"id": "d", "v": 1},
            {"id": "m1", "v": 0},
            {"id": "m2", "v": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"v": 1}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n({"v": 1}, name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "value")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX", "10")
        value_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        value_nodes = set(value_mode._nodes["id"])
        value_edges = set(map(tuple, value_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert value_nodes == baseline_nodes
        assert value_edges == baseline_edges

    def test_multi_eq_vector_mode_matches_expected(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "group": 1, "v_mod10": 1},
            {"id": "b", "group": 2, "v_mod10": 1},
            {"id": "c", "group": 1, "v_mod10": 1},
            {"id": "d", "group": 2, "v_mod10": 2},
            {"id": "m1", "group": 0, "v_mod10": 0},
            {"id": "m2", "group": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "group"), "==", col("end", "group")),
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
        ]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_STRATEGY", "vector")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS", "2")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX", "10")
        vector_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        vector_nodes = set(vector_mode._nodes["id"])
        vector_edges = set(map(tuple, vector_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert vector_nodes == baseline_nodes
        assert vector_edges == baseline_edges

    def test_multi_eq_vector_mode_parity(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "group": 1, "v_mod10": 1},
            {"id": "b", "group": 2, "v_mod10": 1},
            {"id": "c", "group": 1, "v_mod10": 1},
            {"id": "d", "group": 2, "v_mod10": 2},
            {"id": "m1", "group": 0, "v_mod10": 0},
            {"id": "m2", "group": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "group"), "==", col("end", "group")),
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
        ]

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_STRATEGY", "vector")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS", "2")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX", "10")
        _assert_parity(graph, chain, where)

    def test_auto_mode_matches_baseline(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 1},
            {"id": "c", "v": 1},
            {"id": "d", "v": 1},
            {"id": "m1", "v": 0},
            {"id": "m2", "v": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"v": 1}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n({"v": 1}, name="end"),
        ]
        where = [compare(col("start", "v"), "==", col("end", "v"))]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX", "10")
        auto_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        auto_nodes = set(auto_mode._nodes["id"])
        auto_edges = set(map(tuple, auto_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert auto_nodes == baseline_nodes
        assert auto_edges == baseline_edges

    def test_value_mode_neq_matches_baseline(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 1},
            {"id": "c", "v": 1},
            {"id": "d", "v": 2},
            {"id": "m1", "v": 0},
            {"id": "m2", "v": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n({"v": 1}, name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "!=", col("end", "v"))]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "value")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX", "10")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VALUE_OPS", "!=")
        value_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        value_nodes = set(value_mode._nodes["id"])
        value_edges = set(map(tuple, value_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"b", "m2", "d"}
        assert baseline_edges == {("b", "m2"), ("m2", "d")}
        assert value_nodes == baseline_nodes
        assert value_edges == baseline_edges


class TestNonAdjacentBoundsAndOrdering:
    def test_bounds_matches_baseline(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "group": 1},
            {"id": "b", "v": 5, "group": 2},
            {"id": "c", "v": 3, "group": 1},
            {"id": "d", "v": 2, "group": 2},
            {"id": "m1", "v": 0, "group": 0},
            {"id": "m2", "v": 0, "group": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_BOUNDS", "1")
        bounds_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        bounds_nodes = set(bounds_mode._nodes["id"])
        bounds_edges = set(map(tuple, bounds_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert bounds_nodes == baseline_nodes
        assert bounds_edges == baseline_edges

    def test_ordering_matches_baseline(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "group": 1},
            {"id": "b", "v": 5, "group": 2},
            {"id": "c", "v": 3, "group": 1},
            {"id": "d", "v": 2, "group": 2},
            {"id": "m1", "v": 0, "group": 0},
            {"id": "m2", "v": 0, "group": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "group"), "==", col("end", "group")),
        ]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_ORDER", "selectivity")
        ordered = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        ordered_nodes = set(ordered._nodes["id"])
        ordered_edges = set(map(tuple, ordered._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert ordered_nodes == baseline_nodes
        assert ordered_edges == baseline_edges


class TestNonAdjacentMultiClause:
    def test_multi_clause_matches_expected(self):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "v_mod10": 1},
            {"id": "b", "v": 2, "v_mod10": 2},
            {"id": "c", "v": 3, "v_mod10": 1},
            {"id": "d", "v": 1, "v_mod10": 1},
            {"id": "m1", "v": 0, "v_mod10": 0},
            {"id": "m2", "v": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"])
        result_edges = set(map(tuple, result._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert result_nodes == {"a", "m1", "c"}
        assert result_edges == {("a", "m1"), ("m1", "c")}

    def test_multi_clause_auto_guard_parity(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "v_mod10": 1},
            {"id": "b", "v": 2, "v_mod10": 2},
            {"id": "c", "v": 3, "v_mod10": 1},
            {"id": "d", "v": 1, "v_mod10": 1},
            {"id": "m1", "v": 0, "v_mod10": 0},
            {"id": "m2", "v": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX", "1")
        guarded = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        guarded_nodes = set(guarded._nodes["id"])
        guarded_edges = set(map(tuple, guarded._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert guarded_nodes == baseline_nodes
        assert guarded_edges == baseline_edges

    def test_multi_clause_ineq_agg_parity(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "v_mod10": 1},
            {"id": "b", "v": 2, "v_mod10": 2},
            {"id": "c", "v": 3, "v_mod10": 1},
            {"id": "d", "v": 1, "v_mod10": 1},
            {"id": "m1", "v": 0, "v_mod10": 0},
            {"id": "m2", "v": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "auto")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_INEQ_AGG", "1")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_DOMAIN_SEMIJOIN_PAIR_MAX", "1")
        agg_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        agg_nodes = set(agg_mode._nodes["id"])
        agg_edges = set(map(tuple, agg_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert agg_nodes == baseline_nodes
        assert agg_edges == baseline_edges

    def test_multi_eq_value_mode_matches_expected(self, monkeypatch):
        nodes = pd.DataFrame([
            {"id": "a", "group": 1, "v_mod10": 1},
            {"id": "b", "group": 2, "v_mod10": 1},
            {"id": "c", "group": 1, "v_mod10": 1},
            {"id": "d", "group": 2, "v_mod10": 2},
            {"id": "m1", "group": 0, "v_mod10": 0},
            {"id": "m2", "group": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "group"), "==", col("end", "group")),
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
        ]

        baseline = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        baseline_nodes = set(baseline._nodes["id"])
        baseline_edges = set(map(tuple, baseline._edges[["src", "dst"]].itertuples(index=False, name=None)))

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_MODE", "value")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VALUE_CARD_MAX", "10")
        value_mode = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        value_nodes = set(value_mode._nodes["id"])
        value_edges = set(map(tuple, value_mode._edges[["src", "dst"]].itertuples(index=False, name=None)))

        assert baseline_nodes == {"a", "m1", "c"}
        assert baseline_edges == {("a", "m1"), ("m1", "c")}
        assert value_nodes == baseline_nodes
        assert value_edges == baseline_edges



class TestEdgeWhereSemijoinParity:

    @pytest.fixture
    def edge_value_graph(self):
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
            {"id": "d"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "w": 5},
            {"src": "a", "dst": "b", "w": 1},
            {"src": "b", "dst": "c", "w": 3},
            {"src": "b", "dst": "c", "w": 10},
            {"src": "b", "dst": "d", "w": 7},
        ])
        return make_cg_graph(nodes, edges)

    def test_edge_where_gt_semijoin_parity(self, edge_value_graph, monkeypatch):
        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "w"), ">", col("e2", "w"))]

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

    def test_edge_where_neq_semijoin_parity(self, edge_value_graph, monkeypatch):
        chain = [
            n(name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("e1", "w"), "!=", col("e2", "w"))]

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
        nodes = pd.DataFrame([
            {"id": "a"},
            {"id": "b"},
            {"id": "c"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "w": None},
            {"src": "a", "dst": "b", "w": 2},
            {"src": "b", "dst": "c", "w": None},
            {"src": "b", "dst": "c", "w": 1},
        ])
        graph = make_cg_graph(nodes, edges)

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
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "v_mod10": 1},
            {"id": "b", "v": 2, "v_mod10": 1},
            {"id": "c", "v": 3, "v_mod10": 1},
            {"id": "d", "v": 1, "v_mod10": 2},
            {"id": "m1", "v": 0, "v_mod10": 0},
            {"id": "m2", "v": 0, "v_mod10": 0},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "m1"},
            {"src": "m1", "dst": "c"},
            {"src": "b", "dst": "m2"},
            {"src": "m2", "dst": "d"},
        ])
        graph = make_cg_graph(nodes, edges)

        chain = [
            n(name="start"),
            e_forward(),
            n(name="mid"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "v_mod10"), "==", col("end", "v_mod10")),
            compare(col("start", "v"), "<", col("end", "v")),
        ]

        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_STRATEGY", "vector")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_MAX_HOPS", "2")
        monkeypatch.setenv("GRAPHISTRY_NON_ADJ_WHERE_VECTOR_LABEL_MAX", "10")
        _assert_parity(graph, chain, where)
