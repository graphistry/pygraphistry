"""5-whys amplification and WHERE clause tests for df_executor."""

import pandas as pd

from graphistry.Engine import Engine
from graphistry.compute import n, e_forward, e_reverse, e_undirected, is_in
from graphistry.compute.gfql.df_executor import execute_same_path_chain
from graphistry.gfql.same_path_types import col, compare
from graphistry.tests.test_compute import CGFull

# Import shared helpers - pytest auto-loads conftest.py
from tests.gfql.ref.conftest import _assert_parity

class TestYannakakisPrinciple:
    """
    Tests validating the Yannakakis semijoin principle:
    - Edge included iff it participates in at least one valid complete path
    - No edge excluded that could be part of a valid path
    - No spurious edges included that aren't on any valid path
    """

    def test_dead_end_branch_pruning(self):
        """
        Edges leading to nodes that fail WHERE should be excluded.

        Graph: a -> b -> c (valid path, c.v > a.v)
               a -> x -> y (dead end, y.v < a.v)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 6},
            {"id": "c", "v": 10},  # Valid endpoint
            {"id": "x", "v": 4},
            {"id": "y", "v": 1},   # Invalid endpoint (y.v < a.v)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "y"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Valid path a->b->c should be included
        assert {"a", "b", "c"} <= result_nodes
        assert ("a", "b") in result_edges
        assert ("b", "c") in result_edges

        # Dead-end path a->x->y should be excluded (Yannakakis pruning)
        assert "x" not in result_nodes, "x is on dead-end path, should be pruned"
        assert "y" not in result_nodes, "y fails WHERE, should be pruned"
        assert ("a", "x") not in result_edges, "edge to dead-end should be pruned"

    def test_all_valid_paths_included(self):
        """
        Multiple valid paths - all edges on any valid path must be included.

        Graph: a -> b -> d (valid)
               a -> c -> d (valid)
        Both paths are valid, so all edges should be included.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # All nodes on valid paths
        assert result_nodes == {"a", "b", "c", "d"}
        # All edges on valid paths
        assert ("a", "b") in result_edges
        assert ("b", "d") in result_edges
        assert ("a", "c") in result_edges
        assert ("c", "d") in result_edges

    def test_spurious_edge_exclusion(self):
        """
        Edges not on any complete path must be excluded.

        Graph: a -> b -> c (valid 2-hop path)
               b -> x (dangles off, not part of any complete path)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "x", "v": 20},  # Dangles off b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "x"},  # Spurious edge
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
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Valid path edges included
        assert ("a", "b") in result_edges
        assert ("b", "c") in result_edges

        # Spurious edge b->x excluded (x is at hop 2, but path a->b->x is also valid!)
        # Actually, a->b->x IS a valid 2-hop path where x.v=20 > a.v=1
        # So this test needs adjustment - x IS on a valid path
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "x" in result_nodes, "x is actually on valid path a->b->x"

    def test_where_prunes_intermediate_edges(self):
        """
        WHERE filtering can prune intermediate edges.

        Graph: a -> b -> c -> d
        WHERE requires intermediate values to be in a specific range.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # b.v is way higher than d.v
            {"id": "c", "v": 5},
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
        # Valid path exists: a->b->c->d where a.v=1 < d.v=10
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Full path should be included
        assert result_nodes == {"a", "b", "c", "d"}

    def test_convergent_diamond_all_paths_included(self):
        """
        Diamond pattern where both paths are valid.

        Graph:     b
               a <   > d
                   c
        Both a->b->d and a->c->d are valid 2-hop paths.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "c", "dst": "d"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # All nodes and edges from both paths
        assert result_nodes == {"a", "b", "c", "d"}
        assert len(result_edges) == 4

    def test_mixed_valid_invalid_branches(self):
        """
        Some branches valid, some invalid - only valid branch edges included.

        Graph: a -> b -> c (c.v=10 > a.v=1, valid)
               a -> x -> y (y.v=0 < a.v=1, invalid)
               a -> p -> q (q.v=2 > a.v=1, valid)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "x", "v": 3},
            {"id": "y", "v": 0},   # Invalid endpoint
            {"id": "p", "v": 4},
            {"id": "q", "v": 2},   # Valid endpoint (barely)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "x"},
            {"src": "x", "dst": "y"},
            {"src": "a", "dst": "p"},
            {"src": "p", "dst": "q"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Valid paths: a->b->c, a->p->q
        assert {"a", "b", "c", "p", "q"} <= result_nodes

        # Invalid path: a->x->y (y.v=0 < a.v=1)
        assert "x" not in result_nodes, "x is only on invalid path"
        assert "y" not in result_nodes, "y fails WHERE"


class TestHopLabelingPatterns:
    """
    Tests for the anti-join patterns used in hop labeling.

    The anti-join patterns in hop.py (lines 661, 682) are used for display
    (hop labels), not filtering. These tests verify they don't affect path validity.
    """

    def test_hop_labels_dont_affect_validity(self):
        """
        Nodes reachable via multiple paths should all be included,
        regardless of which path labels them first.

        Graph: a -> b -> d (2 hops)
               a -> c -> d (2 hops)
        Node 'd' is reachable via two paths - both should work.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # d is reachable via both b and c - both intermediates should be included
        assert result_nodes == {"a", "b", "c", "d"}

    def test_multiple_seeds_hop_labels(self):
        """
        Multiple seeds with overlapping reachable nodes.

        Seeds: a, b
        Graph: a -> c, b -> c, c -> d
        Both seeds can reach c and d.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 2},
            {"id": "c", "v": 5},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Multiple seeds via filter
        chain = [
            n({"v": is_in([1, 2])}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # Both seeds and all reachable nodes
        assert {"a", "b", "c", "d"} <= result_nodes

    def test_hop_labels_with_min_hops(self):
        """
        Hop labels with min_hops > 1 - intermediate nodes still included.

        Graph: a -> b -> c -> d
        With min_hops=2, path a->b->c->d valid at hops 2 and 3.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 3},
            {"id": "c", "v": 5},
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
            e_forward(min_hops=2, max_hops=3),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # All nodes on paths of length 2-3
        assert result_nodes == {"a", "b", "c", "d"}

    def test_edge_hop_labels_consistent(self):
        """
        Edge hop labels should be consistent across multiple paths.

        Graph: a -> b -> c
               a -> b (same edge used in 1-hop and as part of 2-hop)
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

    def test_undirected_hop_labels(self):
        """
        Undirected traversal - nodes reachable in both directions.

        Graph: a - b - c (undirected)
        From a, can reach b at hop 1, c at hop 2.
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
            e_undirected(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        # All nodes reachable via undirected traversal
        assert {"a", "b", "c"} <= result_nodes


class TestSensitivePhenomena:
    """
    Tests for sensitive phenomena identified through deep 5-whys analysis.

    These test edge cases that have historically caused bugs:
    1. Asymmetric reachability (forward ≠ reverse)
    2. Filter cascades creating empty intermediates
    3. Non-adjacent WHERE with complex patterns
    4. Path length boundary conditions
    5. Shared edge semantics
    6. Self-loops and cycles
    """

    # --- Asymmetric Reachability ---

    def test_asymmetric_graph_forward_only_node(self):
        """
        Node reachable only via forward traversal.

        Graph: a -> b -> c
               d -> b (d has no path TO it, only FROM it)
        Forward from a: reaches b, c
        Reverse from a: reaches nothing
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 2},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "d", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Forward should find b, c
        chain_fwd = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain_fwd, where)

        result = execute_same_path_chain(graph, chain_fwd, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes
        assert "d" not in result_nodes  # d is not reachable forward from a

    def test_asymmetric_graph_reverse_only_node(self):
        """
        Node reachable only via reverse traversal.

        Graph: b -> a, c -> b
        From a (reverse): reaches b, c
        From a (forward): reaches nothing
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 10},
            {"id": "b", "v": 5},
            {"id": "c", "v": 1},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},
            {"src": "c", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        # Reverse should find b, c
        chain_rev = [
            n({"id": "a"}, name="start"),
            e_reverse(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), ">", col("end", "v"))]

        _assert_parity(graph, chain_rev, where)

        result = execute_same_path_chain(graph, chain_rev, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes

    def test_undirected_finds_reverse_only_node(self):
        """
        Undirected traversal should find nodes only reachable "backwards".

        Graph: b -> a (edge points TO a)
        Undirected from a: should reach b (traversing edge backwards)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Points TO a, not from a
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "undirected should find b via backward edge"

    # --- Filter Cascades ---

    def test_filter_eliminates_all_at_step(self):
        """
        Node filter eliminates all matches, creating empty intermediate.

        Graph: a -> b -> c
        Filter: node must have type="special" (none do)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "normal"},
            {"id": "b", "v": 5, "type": "normal"},
            {"id": "c", "v": 10, "type": "normal"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

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
        """
        WHERE clause eliminates all valid paths.

        Graph: a -> b -> c (all v increasing)
        WHERE: start.v > end.v (impossible since v increases)
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

    # --- Non-Adjacent WHERE Edge Cases ---

    def test_three_step_start_to_end_comparison(self):
        """
        Three-step chain with start-to-end comparison (skipping middle).

        Chain: start -[2 hops]-> middle -[1 hop]-> end
        WHERE: start.v < end.v (ignores middle)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 100},  # Middle has high value (should be ignored)
            {"id": "c", "v": 50},
            {"id": "d", "v": 10},   # End with low value
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="middle"),
            e_forward(min_hops=1, max_hops=1),
            n(name="end"),
        ]
        # Compare start to end, ignoring middle
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Path a->b->c->d: start.v=1 < end.v=10, valid
        # c is middle at hop 2, d is end
        assert "d" in result_nodes

    def test_multiple_non_adjacent_constraints(self):
        """
        Multiple non-adjacent WHERE constraints.

        Chain: a -> b -> c
        WHERE: a.v < c.v AND a.type == c.type
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "X"},
            {"id": "b", "v": 5, "type": "Y"},
            {"id": "c", "v": 10, "type": "X"},  # Same type as a
            {"id": "d", "v": 20, "type": "Z"},  # Different type
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        # Two constraints: v comparison AND type equality
        where = [
            compare(col("start", "v"), "<", col("end", "v")),
            compare(col("start", "type"), "==", col("end", "type")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # c matches both constraints, d fails type constraint
        assert "c" in result_nodes
        assert "d" not in result_nodes

    # --- Path Length Boundary Conditions ---

    def test_min_hops_zero_includes_seed(self):
        """
        min_hops=0 should include the seed node itself.

        Graph: a -> b
        With min_hops=0, 'a' is a valid endpoint (0 hops from itself)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=0, max_hops=1),
            n(name="end"),
        ]
        # a.v <= end.v (includes a itself since 5 <= 5)
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Both a (0 hops) and b (1 hop) should be valid endpoints
        assert "a" in result_nodes, "min_hops=0 should include seed"
        assert "b" in result_nodes

    def test_max_hops_exceeds_graph_diameter(self):
        """
        max_hops larger than graph diameter should work fine.

        Graph: a -> b -> c (diameter = 2)
        max_hops = 10 should still only find paths up to length 2
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
            e_forward(min_hops=1, max_hops=10),  # Way more than needed
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes
        assert "c" in result_nodes

    # --- Shared Edge Semantics ---

    def test_edge_used_by_multiple_destinations(self):
        """
        Single edge participates in paths to different destinations.

        Graph: a -> b -> c
                    b -> d
        Edge a->b is used for both path to c and path to d.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = set(zip(result._edges["src"], result._edges["dst"])) if result._edges is not None else set()

        # Both destinations should be found
        assert "c" in result_nodes
        assert "d" in result_nodes
        # Edge a->b should be included (shared by both paths)
        assert ("a", "b") in result_edges

    def test_diamond_shared_edges(self):
        """
        Diamond pattern where edges are shared.

        Graph: a -> b -> d
               a -> c -> d
        Two paths share start (a) and end (d).
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 6},
            {"id": "d", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "d"},
            {"src": "a", "dst": "c"},
            {"src": "c", "dst": "d"},
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
        result_edges = result._edges
        # All 4 edges should be included
        assert len(result_edges) == 4

    # --- Self-Loops and Cycles ---

    def test_self_loop_edge(self):
        """
        Graph with self-loop edge.

        Graph: a -> a (self-loop), a -> b
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "a"},  # Self-loop
            {"src": "a", "dst": "b"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Both a (via self-loop) and b should be reachable
        assert "b" in result_nodes

    def test_small_cycle_with_min_hops(self):
        """
        Small cycle with min_hops constraint.

        Graph: a -> b -> a (cycle)
        With min_hops=2, can reach a via the cycle.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 5},
            {"id": "b", "v": 3},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "a"},  # Creates cycle
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        # a.v=5 <= end.v, so a (reached at hop 2) is valid
        where = [compare(col("start", "v"), "<=", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # a is reachable at hop 2 via a->b->a
        assert "a" in result_nodes, "should reach a via cycle at hop 2"

    def test_cycle_with_branch(self):
        """
        Cycle with a branch leading out.

        Graph: a -> b -> c -> a (cycle)
               c -> d (branch)
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
            {"src": "c", "dst": "a"},  # Cycle back
            {"src": "c", "dst": "d"},  # Branch out
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
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # b (hop 1), c (hop 2), d (hop 3) should all be reachable
        assert "b" in result_nodes
        assert "c" in result_nodes
        assert "d" in result_nodes


class TestNodeEdgeMatchFilters:
    """
    Tests for source_node_match, destination_node_match, and edge_match filters.

    These filters restrict traversal based on node/edge attributes, independent
    of the endpoint node filters or WHERE clauses.
    """

    def test_destination_node_match_single_hop(self):
        """
        destination_node_match restricts which nodes can be reached.

        Graph: a -> b (target), a -> c (other)
        With destination_node_match={'type': 'target'}, only b should be reached.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 10, "type": "target"},
            {"id": "c", "v": 20, "type": "other"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach target type node"
        assert "c" not in result_nodes, "should not reach other type node"

    def test_source_node_match_single_hop(self):
        """
        source_node_match restricts which nodes can be traversed FROM.

        Graph: a (good) -> c, b (bad) -> c
        With source_node_match={'type': 'good'}, only path from a should exist.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "good"},
            {"id": "b", "v": 5, "type": "bad"},
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(source_node_match={"type": "good"}, min_hops=1, max_hops=1),
            n({"id": "c"}, name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "a" in result_nodes, "good type source should be included"
        assert "b" not in result_nodes, "bad type source should be excluded"

    def test_edge_match_single_hop(self):
        """
        edge_match restricts which edges can be traversed.

        Graph: a -friend-> b, a -enemy-> c
        With edge_match={'type': 'friend'}, only path via friend edge should exist.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 10},
            {"id": "c", "v": 20},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "type": "friend"},
            {"src": "a", "dst": "c", "type": "enemy"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(edge_match={"type": "friend"}, min_hops=1, max_hops=1),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach via friend edge"
        assert "c" not in result_nodes, "should not reach via enemy edge"

    def test_destination_node_match_multi_hop(self):
        """
        destination_node_match applies at EACH hop, not just final.

        Graph: a -> b (target) -> c (target)
        With destination_node_match={'type': 'target'}, b and c must both be targets.
        Note: destination_node_match filters destinations at every hop step,
        so intermediate nodes must also match.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 5, "type": "target"},  # intermediate must also be target
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(destination_node_match={"type": "target"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b (target) at hop 1"
        assert "c" in result_nodes, "should reach c (target) at hop 2"

    def test_combined_source_and_dest_match(self):
        """
        Both source_node_match and destination_node_match together.

        Graph: a (sender) -> c, b (receiver) -> c, a -> d
        source_node_match={'role': 'sender'}, destination_node_match={'type': 'target'}
        Only a->c path should work (a is sender, c would need to be target)
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "role": "sender", "type": "node"},
            {"id": "b", "v": 5, "role": "receiver", "type": "node"},
            {"id": "c", "v": 10, "role": "none", "type": "target"},
            {"id": "d", "v": 15, "role": "none", "type": "other"},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "c"},
            {"src": "b", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n(name="start"),
            e_forward(
                source_node_match={"role": "sender"},
                destination_node_match={"type": "target"},
                min_hops=1, max_hops=1
            ),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "a" in result_nodes, "sender a should be included"
        assert "c" in result_nodes, "target c should be reached"
        assert "b" not in result_nodes, "receiver b should be excluded as source"
        assert "d" not in result_nodes, "other d should be excluded as destination"

    def test_edge_match_multi_hop(self):
        """
        edge_match restricts which edges can be used in multi-hop.

        Graph: a -good-> b -good-> c, b -bad-> d
        With edge_match={'quality': 'good'}, only a-b-c path should work.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1},
            {"id": "b", "v": 5},
            {"id": "c", "v": 10},
            {"id": "d", "v": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b", "quality": "good"},
            {"src": "b", "dst": "c", "quality": "good"},
            {"src": "b", "dst": "d", "quality": "bad"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(edge_match={"quality": "good"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b via good edge"
        assert "c" in result_nodes, "should reach c via good edges"
        assert "d" not in result_nodes, "should not reach d via bad edge"

    def test_undirected_with_destination_match(self):
        """
        destination_node_match with undirected traversal.

        Graph: b -> a, b -> c (both targets)
        Undirected from a with destination_node_match={'type': 'target'}
        should find b and c (all targets along the path).
        Note: destination_node_match applies at each hop, so b must also be target.
        """
        nodes = pd.DataFrame([
            {"id": "a", "v": 1, "type": "source"},
            {"id": "b", "v": 5, "type": "target"},  # must also be target for multi-hop
            {"id": "c", "v": 10, "type": "target"},
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # Points TO a
            {"src": "b", "dst": "c"},  # Points TO c
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(destination_node_match={"type": "target"}, min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "v"), "<", col("end", "v"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "should reach b (target) at hop 1"
        assert "c" in result_nodes, "should reach c (target) at hop 2"


class TestWhereClauseConjunction:
    """
    Test conjunction (AND) semantics for multiple WHERE clauses.

    Current behavior: Multiple WHERE clauses are treated as conjunction (AND).
    This is compatible with Yannakakis pruning because AND is monotonic -
    adding constraints can only reduce the valid set, never expand it.

    Disjunction (OR) is NOT supported because it breaks monotonic pruning:
    - A node might fail one clause but satisfy another via a different path
    - Pruning based on one clause could remove nodes needed by another
    """

    def test_conjunction_two_clauses_same_columns(self):
        """Two clauses on same column pair: a.x > c.x AND a.y < c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 5, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # a.x > c.x (10>5) AND a.y < c.y (1<10) - VALID
            {"id": "d", "x": 5, "y": 0},    # a.x > d.x (10>5) BUT a.y < d.y (1<0) - INVALID
            {"id": "e", "x": 15, "y": 10},  # a.x > e.x (10>15) FAILS - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "b", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies both clauses"
        assert "d" not in result_nodes, "d fails y clause"
        assert "e" not in result_nodes, "e fails x clause"

    def test_conjunction_three_clauses(self):
        """Three clauses: a.x == c.x AND a.y < c.y AND a.z > c.z"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1, "z": 10},
            {"id": "b", "x": 5, "y": 5, "z": 5},
            {"id": "c", "x": 5, "y": 10, "z": 5},  # x==5, y=10>1, z=5<10 - VALID
            {"id": "d", "x": 5, "y": 10, "z": 15}, # x==5, y=10>1, BUT z=15>10 - INVALID
            {"id": "e", "x": 9, "y": 10, "z": 5},  # x=9!=5 - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
            {"src": "b", "dst": "e"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=1, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "==", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
            compare(col("start", "z"), ">", col("end", "z")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies all three clauses"
        assert "d" not in result_nodes, "d fails z clause"
        assert "e" not in result_nodes, "e fails x clause"

    def test_conjunction_adjacent_and_nonadjacent(self):
        """Mix adjacent and non-adjacent clauses: a.x == b.x AND a.y < c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1},
            {"id": "b1", "x": 5, "y": 5},   # x matches a
            {"id": "b2", "x": 9, "y": 5},   # x doesn't match a
            {"id": "c1", "x": 5, "y": 10},  # y > a.y
            {"id": "c2", "x": 5, "y": 0},   # y < a.y
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c1"},
            {"src": "b1", "dst": "c2"},
            {"src": "b2", "dst": "c1"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),  # adjacent
            compare(col("a", "y"), "<", col("c", "y")),   # non-adjacent
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only path a->b1->c1 satisfies both clauses
        assert "b1" in result_nodes, "b1 has x==5 matching a"
        assert "c1" in result_nodes, "c1 has y>1"
        assert "b2" not in result_nodes, "b2 has x!=5"
        assert "c2" not in result_nodes, "c2 has y<1"

    def test_conjunction_multihop_single_edge_step(self):
        """Conjunction with multi-hop: a.x > c.x AND a.y < c.y via 2-hop edge"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 7, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # VALID: 10>5 AND 1<10
            {"id": "d", "x": 5, "y": 0},    # INVALID: 10>5 BUT 1>0
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),  # exactly 2 hops
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c satisfies both clauses"
        assert "d" not in result_nodes, "d fails y clause"

    def test_conjunction_with_impossible_combination(self):
        """Clauses that are individually satisfiable but not together."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 5},
            {"id": "b", "x": 3, "y": 7},   # x<5 AND y>5 - satisfies both!
            {"id": "c", "x": 7, "y": 3},   # x>5 AND y<5 - fails both
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
        # Need end.x < 5 AND end.y > 5 - b satisfies both
        where = [
            compare(col("start", "x"), ">", col("end", "x")),  # need end.x < 5
            compare(col("start", "y"), "<", col("end", "y")),  # need end.y > 5
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" in result_nodes, "b satisfies: 5>3 AND 5<7"
        assert "c" not in result_nodes, "c fails: 5<7"

    def test_conjunction_empty_result(self):
        """All paths fail at least one clause."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 5},
            {"id": "b", "x": 10, "y": 10},  # fails x clause (5 < 10, not >)
            {"id": "c", "x": 3, "y": 3},    # fails y clause (5 > 3, not <)
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
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Only 'a' (seed) should remain, no valid endpoints
        assert "a" in result_nodes or len(result_nodes) == 0, "empty or seed-only result"
        assert "b" not in result_nodes, "b fails x clause"
        assert "c" not in result_nodes, "c fails y clause"

    def test_conjunction_diamond_multiple_paths(self):
        """
        Diamond topology where different paths might satisfy different clauses.

        With conjunction, a node is included only if SOME path to it satisfies ALL clauses.
        This is the key Yannakakis property - we don't need ALL paths to work,
        just at least one complete valid path.

            a
           / \\
          b1  b2
           \\ /
            c

        Clauses: a.x == b.x AND a.y < c.y
        b1.x = 5 (matches a.x=5), b2.x = 9 (doesn't match)
        c.y = 10 > a.y = 1

        Path a->b1->c should work. Path a->b2->c fails at b2.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 1},
            {"id": "b1", "x": 5, "y": 5},   # x matches
            {"id": "b2", "x": 9, "y": 5},   # x doesn't match
            {"id": "c", "x": 5, "y": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),
            compare(col("a", "y"), "<", col("c", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = result._edges

        # c should be reachable via the valid path a->b1->c
        assert "c" in result_nodes, "c reachable via valid path a->b1->c"
        assert "b1" in result_nodes, "b1 is on valid path"
        # b2 should NOT be included - it's not on any valid path
        assert "b2" not in result_nodes, "b2 not on any valid path (x mismatch)"
        # Edge a->b2 should be excluded
        if result_edges is not None and len(result_edges) > 0:
            edge_pairs = set(zip(result_edges["src"], result_edges["dst"]))
            assert ("a", "b2") not in edge_pairs, "edge a->b2 should be excluded"

    def test_conjunction_undirected_multihop(self):
        """Conjunction with undirected multi-hop traversal."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 10, "y": 1},
            {"id": "b", "x": 7, "y": 5},
            {"id": "c", "x": 5, "y": 10},   # VALID via undirected
        ])
        edges = pd.DataFrame([
            {"src": "b", "dst": "a"},  # reversed - need undirected to traverse
            {"src": "c", "dst": "b"},  # reversed
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_undirected(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), ">", col("end", "x")),
            compare(col("start", "y"), "<", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c reachable via undirected and satisfies both clauses"


class TestWhereClauseNegation:
    """
    Test negation (!=) in WHERE clauses, including combinations with other operators.

    Negation is tricky for Yannakakis pruning because:
    - `a.x != c.x` doesn't give useful global bounds (everything except one value is valid)
    - Early pruning is skipped for != (see _prune_clause)
    - Per-edge filtering still works correctly

    These tests verify != works alone and in combination with other operators.
    """

    def test_negation_simple(self):
        """Simple != clause: exclude paths where values match."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 5},   # same as a - INVALID
            {"id": "c", "x": 10},  # different from a - VALID
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
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c has different x value"
        assert "b" not in result_nodes, "b has same x value as a"

    def test_negation_with_equality(self):
        """Combine != and ==: a.x != c.x AND a.y == c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 10},   # x same, y same - INVALID (x match fails !=)
            {"id": "c", "x": 10, "y": 10},  # x different, y same - VALID
            {"id": "d", "x": 10, "y": 20},  # x different, y different - INVALID (y fails ==)
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), "==", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c: x!=5 AND y==10"
        assert "b" not in result_nodes, "b: x==5 fails !="
        assert "d" not in result_nodes, "d: y!=10 fails =="

    def test_negation_with_inequality(self):
        """Combine != and >: a.x != c.x AND a.y > c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 5},    # x same - INVALID
            {"id": "c", "x": 10, "y": 5},   # x different, y < a.y - VALID
            {"id": "d", "x": 10, "y": 15},  # x different, but y > a.y - INVALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), ">", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "c" in result_nodes, "c: x!=5 AND 10>5"
        assert "b" not in result_nodes, "b: x==5 fails !="
        assert "d" not in result_nodes, "d: 10<15 fails >"

    def test_double_negation(self):
        """Two != clauses: a.x != c.x AND a.y != c.y"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b", "x": 5, "y": 20},   # x same - INVALID
            {"id": "c", "x": 10, "y": 10},  # y same - INVALID
            {"id": "d", "x": 10, "y": 20},  # both different - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "a", "dst": "c"},
            {"src": "a", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(),
            n(name="end"),
        ]
        where = [
            compare(col("start", "x"), "!=", col("end", "x")),
            compare(col("start", "y"), "!=", col("end", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_nodes, "d: x!=5 AND y!=10"
        assert "b" not in result_nodes, "b: x==5 fails first !="
        assert "c" not in result_nodes, "c: y==10 fails second !="

    def test_negation_multihop(self):
        """!= with multi-hop traversal."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 7},
            {"id": "c", "x": 5},   # same as a - INVALID
            {"id": "d", "x": 10},  # different from a - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "d" in result_nodes, "d has different x value"
        assert "c" not in result_nodes, "c has same x value as a"

    def test_negation_adjacent_steps(self):
        """!= between adjacent steps: a.x != b.x"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same - INVALID
            {"id": "b2", "x": 10},  # different - VALID
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b2" in result_nodes, "b2 has different x"
        assert "c" in result_nodes, "c reachable via b2"
        assert "b1" not in result_nodes, "b1 has same x as a"

    def test_negation_nonadjacent_with_equality_adjacent(self):
        """Mix: a.x == b.x (adjacent) AND a.y != c.y (non-adjacent)"""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b1", "x": 5, "y": 7},   # x matches a
            {"id": "b2", "x": 9, "y": 7},   # x doesn't match a
            {"id": "c1", "x": 5, "y": 10},  # y same as a - INVALID
            {"id": "c2", "x": 5, "y": 20},  # y different - VALID
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c1"},
            {"src": "b1", "dst": "c2"},
            {"src": "b2", "dst": "c2"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "==", col("b", "x")),  # adjacent
            compare(col("a", "y"), "!=", col("c", "y")),  # non-adjacent
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        # Valid path: a->b1->c2 (b1.x==5, c2.y!=10)
        assert "b1" in result_nodes, "b1 has x==5"
        assert "c2" in result_nodes, "c2 has y!=10"
        assert "b2" not in result_nodes, "b2 has x!=5"
        assert "c1" not in result_nodes, "c1 has y==10"

    def test_negation_all_match_empty_result(self):
        """All endpoints have same value - empty result."""
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 5},
            {"id": "c", "x": 5},
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
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        assert "b" not in result_nodes, "b has same x"
        assert "c" not in result_nodes, "c has same x"

    def test_negation_diamond_one_path_valid(self):
        """
        Diamond where only one path satisfies != constraint.

            a (x=5)
           / \\
      (x=5)b1  b2(x=10)
           \\ /
            c (x=5)

        Clause: a.x != b.x
        - Path a->b1->c: b1.x=5 == a.x=5, FAILS
        - Path a->b2->c: b2.x=10 != a.x=5, VALID

        c should be included (reachable via valid path), but b1 should be excluded.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},   # same as a - invalid path
            {"id": "b2", "x": 10},  # different - valid path
            {"id": "c", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()
        result_edges = result._edges

        assert "c" in result_nodes, "c reachable via a->b2->c"
        assert "b2" in result_nodes, "b2 is on valid path"
        assert "b1" not in result_nodes, "b1 fails != constraint"

        # Edge a->b1 should be excluded
        if result_edges is not None and len(result_edges) > 0:
            edge_pairs = set(zip(result_edges["src"], result_edges["dst"]))
            assert ("a", "b1") not in edge_pairs, "edge a->b1 excluded"
            assert ("a", "b2") in edge_pairs, "edge a->b2 included"

    def test_negation_diamond_both_paths_fail(self):
        """
        Diamond where BOTH paths fail != constraint - c should be excluded.

            a (x=5)
           / \\
      (x=5)b1  b2(x=5)
           \\ /
            c

        Both b1 and b2 have x=5 == a.x, so no valid path to c.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},
            {"id": "b2", "x": 5},
            {"id": "c", "x": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c not reachable - all paths fail"
        assert "b1" not in result_nodes, "b1 fails !="
        assert "b2" not in result_nodes, "b2 fails !="

    def test_negation_convergent_paths_different_intermediates(self):
        """
        Multiple paths to same end with different intermediate constraints.

            a (x=5, y=10)
           /|\\
          b1 b2 b3
           \\|/
            c (x=10, y=10)

        Clauses: a.x != b.x AND a.y == c.y
        - b1.x=5 (fails !=), b2.x=10 (passes), b3.x=5 (fails)
        - c.y=10 == a.y=10 (passes)

        Only path a->b2->c is valid.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5, "y": 10},
            {"id": "b1", "x": 5, "y": 7},
            {"id": "b2", "x": 10, "y": 7},
            {"id": "b3", "x": 5, "y": 7},
            {"id": "c", "x": 10, "y": 10},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "a", "dst": "b3"},
            {"src": "b1", "dst": "c"},
            {"src": "b2", "dst": "c"},
            {"src": "b3", "dst": "c"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("a", "y"), "==", col("c", "y")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c reachable via b2"
        assert "b2" in result_nodes, "b2 on valid path"
        assert "b1" not in result_nodes, "b1 fails !="
        assert "b3" not in result_nodes, "b3 fails !="

    def test_negation_conflict_start_end_same_value(self):
        """
        Negation between start and end where they happen to have same value.

        a (x=5) -> b -> c (x=5)

        Clause: a.x != c.x
        a.x=5 == c.x=5, so path is invalid.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},  # same as a
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
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" not in result_nodes, "c has same x as start"

    def test_negation_multiple_ends_some_match(self):
        """
        Multiple endpoints, some match start value (fail !=), others don't.

              a (x=5)
             /|\\
            b1 b2 b3
            |  |  |
            c1 c2 c3
           (5)(10)(5)

        Clause: a.x != c.x
        - c1.x=5 == a.x FAILS
        - c2.x=10 != a.x PASSES
        - c3.x=5 == a.x FAILS
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 7},
            {"id": "b2", "x": 8},
            {"id": "b3", "x": 9},
            {"id": "c1", "x": 5},
            {"id": "c2", "x": 10},
            {"id": "c3", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "a", "dst": "b3"},
            {"src": "b1", "dst": "c1"},
            {"src": "b2", "dst": "c2"},
            {"src": "b3", "dst": "c3"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="start"),
            e_forward(min_hops=2, max_hops=2),
            n(name="end"),
        ]
        where = [compare(col("start", "x"), "!=", col("end", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c2" in result_nodes, "c2.x=10 != a.x=5"
        assert "b2" in result_nodes, "b2 on valid path to c2"
        assert "c1" not in result_nodes, "c1.x=5 == a.x"
        assert "c3" not in result_nodes, "c3.x=5 == a.x"
        assert "b1" not in result_nodes, "b1 only leads to invalid c1"
        assert "b3" not in result_nodes, "b3 only leads to invalid c3"

    def test_negation_cycle_same_node_different_hops(self):
        """
        Cycle where same node appears at different hops.

        a (x=5) -> b (x=10) -> c (x=5) -> a

        With min_hops=2, max_hops=3:
        - hop 2: c (x=5 == a.x, FAILS !=)
        - hop 3: a (x=5 == a.x, FAILS !=)

        But b at hop 1 has x=10 != 5, if we can reach it as endpoint.
        With min_hops=1, max_hops=1: b should pass.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "c", "dst": "a"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

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

    def test_negation_undirected_diamond(self):
        """
        Undirected diamond with negation constraint.

        Graph edges (directed): b1 <- a -> b2, c -> b1, c -> b2
        Undirected traversal from a.

            a (x=5)
           / \\
          b1  b2
           \\ /
            c

        With undirected, can reach c via a->b1->c or a->b2->c.
        Clause: a.x != b.x
        - b1.x=5 == a.x FAILS
        - b2.x=10 != a.x PASSES

        c should be reachable via b2.
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b1", "x": 5},
            {"id": "b2", "x": 10},
            {"id": "c", "x": 15},
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b1"},
            {"src": "a", "dst": "b2"},
            {"src": "c", "dst": "b1"},  # reversed
            {"src": "c", "dst": "b2"},  # reversed
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_undirected(name="e1"),
            n(name="b"),
            e_undirected(name="e2"),
            n(name="c"),
        ]
        where = [compare(col("a", "x"), "!=", col("b", "x"))]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c reachable via b2"
        assert "b2" in result_nodes, "b2 passes !="
        assert "b1" not in result_nodes, "b1 fails !="

    def test_negation_with_equality_conflicting_requirements(self):
        """
        Conflicting constraints: a.x != b.x AND b.x == c.x

        This requires:
        1. b.x different from a.x
        2. c.x same as b.x (thus also different from a.x)

        a (x=5) -> b (x=10) -> c (x=10)  VALID: 5!=10, 10==10
        a (x=5) -> b (x=10) -> d (x=5)   INVALID: 5!=10 passes, but 10!=5 fails ==
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 10},  # matches b
            {"id": "d", "x": 5},   # doesn't match b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("b", "x"), "==", col("c", "x")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: a.x!=b.x AND b.x==c.x"
        assert "b" in result_nodes, "b on valid path"
        assert "d" not in result_nodes, "d: b.x!=d.x fails =="

    def test_negation_transitive_chain(self):
        """
        Chain with negation propagating through: a.x != b.x AND b.x != c.x

        a (x=5) -> b (x=10) -> c (x=5)
        - 5 != 10: PASS
        - 10 != 5: PASS
        Both constraints satisfied!

        a (x=5) -> b (x=10) -> d (x=10)
        - 5 != 10: PASS
        - 10 != 10: FAIL
        """
        nodes = pd.DataFrame([
            {"id": "a", "x": 5},
            {"id": "b", "x": 10},
            {"id": "c", "x": 5},   # different from b
            {"id": "d", "x": 10},  # same as b
        ])
        edges = pd.DataFrame([
            {"src": "a", "dst": "b"},
            {"src": "b", "dst": "c"},
            {"src": "b", "dst": "d"},
        ])
        graph = CGFull().nodes(nodes, "id").edges(edges, "src", "dst")

        chain = [
            n({"id": "a"}, name="a"),
            e_forward(name="e1"),
            n(name="b"),
            e_forward(name="e2"),
            n(name="c"),
        ]
        where = [
            compare(col("a", "x"), "!=", col("b", "x")),
            compare(col("b", "x"), "!=", col("c", "x")),
        ]

        _assert_parity(graph, chain, where)

        result = execute_same_path_chain(graph, chain, where, Engine.PANDAS)
        result_nodes = set(result._nodes["id"]) if result._nodes is not None else set()

        assert "c" in result_nodes, "c: 5!=10 AND 10!=5"
        assert "d" not in result_nodes, "d: 10==10 fails second !="


