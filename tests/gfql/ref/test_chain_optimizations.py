"""
Tests for chain.py optimizations.

This module tests the backward pass optimization and combine_steps optimization
to ensure correctness across various edge cases.

The backward pass optimization (commit 12d89596) skips the full hop() call for
simple single-hop edges and uses vectorized merge filtering instead.

The combine_steps optimization filters edges by valid endpoints instead of
re-running the forward op.
"""

import pandas as pd
import pytest
from typing import Set

from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, ASTEdge
from graphistry.compute.chain import Chain, _is_simple_single_hop

# Import test fixtures
from tests.gfql.ref.conftest import CGFull


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def linear_graph():
    """Linear graph: a -> b -> c -> d"""
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'type': ['start', 'mid', 'mid', 'end'],
        'value': [0, 1, 2, 3]
    })
    edges = pd.DataFrame({
        'src': ['a', 'b', 'c'],
        'dst': ['b', 'c', 'd'],
        'eid': [0, 1, 2],
        'weight': [1.0, 2.0, 3.0]
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


@pytest.fixture
def branching_graph():
    """Branching graph: a -> b, a -> c, b -> d, c -> d"""
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'type': ['root', 'left', 'right', 'sink'],
        'value': [0, 1, 2, 3]
    })
    edges = pd.DataFrame({
        'src': ['a', 'a', 'b', 'c'],
        'dst': ['b', 'c', 'd', 'd'],
        'eid': [0, 1, 2, 3],
        'branch': ['left', 'right', 'left', 'right']
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


@pytest.fixture
def cyclic_graph():
    """Cyclic graph: a -> b -> c -> a"""
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c'],
        'value': [0, 1, 2]
    })
    edges = pd.DataFrame({
        'src': ['a', 'b', 'c'],
        'dst': ['b', 'c', 'a'],
        'eid': [0, 1, 2]
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


@pytest.fixture
def disconnected_graph():
    """Disconnected graph: (a -> b) and (c -> d) with no connection"""
    nodes = pd.DataFrame({
        'id': ['a', 'b', 'c', 'd'],
        'component': [1, 1, 2, 2]
    })
    edges = pd.DataFrame({
        'src': ['a', 'c'],
        'dst': ['b', 'd'],
        'eid': [0, 1]
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


@pytest.fixture
def self_loop_graph():
    """Graph with self-loop: a -> a, a -> b"""
    nodes = pd.DataFrame({
        'id': ['a', 'b'],
        'value': [0, 1]
    })
    edges = pd.DataFrame({
        'src': ['a', 'a'],
        'dst': ['a', 'b'],
        'eid': [0, 1]
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


@pytest.fixture
def parallel_edges_graph():
    """Graph with parallel edges: a -> b (twice)"""
    nodes = pd.DataFrame({
        'id': ['a', 'b'],
        'value': [0, 1]
    })
    edges = pd.DataFrame({
        'src': ['a', 'a'],
        'dst': ['b', 'b'],
        'eid': [0, 1],
        'label': ['first', 'second']
    })
    return CGFull().nodes(nodes, 'id').edges(edges, 'src', 'dst', edge='eid')


# =============================================================================
# TestBackwardPassOptimization
# =============================================================================


class TestOptimizationEligibility:
    """Test that _is_simple_single_hop correctly identifies eligible edges."""

    def test_single_hop_default_is_eligible(self):
        """Default e_forward() is eligible for optimization."""
        op = e_forward()
        assert _is_simple_single_hop(op) is True

    def test_single_hop_explicit_is_eligible(self):
        """e_forward(hops=1) is eligible."""
        op = e_forward(hops=1)
        assert _is_simple_single_hop(op) is True

    def test_single_hop_min_max_is_eligible(self):
        """e_forward(min_hops=1, max_hops=1) is eligible."""
        op = e_forward(min_hops=1, max_hops=1)
        assert _is_simple_single_hop(op) is True

    def test_multihop_range_not_eligible(self):
        """e_forward(min_hops=1, max_hops=3) is NOT eligible."""
        op = e_forward(min_hops=1, max_hops=3)
        assert _is_simple_single_hop(op) is False

    def test_multihop_fixed_not_eligible(self):
        """e_forward(hops=2) is NOT eligible."""
        op = e_forward(hops=2)
        assert _is_simple_single_hop(op) is False

    def test_node_hop_labels_not_eligible(self):
        """e_forward(label_node_hops='hop') is NOT eligible."""
        op = e_forward(label_node_hops='hop')
        assert _is_simple_single_hop(op) is False

    def test_edge_hop_labels_not_eligible(self):
        """e_forward(label_edge_hops='hop') is NOT eligible."""
        op = e_forward(label_edge_hops='hop')
        assert _is_simple_single_hop(op) is False

    def test_seed_labels_not_eligible(self):
        """e_forward(label_seeds=True) is NOT eligible."""
        op = e_forward(label_seeds=True)
        assert _is_simple_single_hop(op) is False

    def test_output_slice_not_eligible(self):
        """e_forward(output_min_hops=1) is NOT eligible."""
        op = e_forward(output_min_hops=1)
        assert _is_simple_single_hop(op) is False

    def test_reverse_is_eligible(self):
        """e_reverse() is eligible."""
        op = e_reverse()
        assert _is_simple_single_hop(op) is True

    def test_undirected_is_eligible(self):
        """e_undirected() is eligible."""
        op = e_undirected()
        assert _is_simple_single_hop(op) is True


class TestDirectionSemantics:
    """Test that backward pass returns correct nodes for each direction."""

    def test_forward_edge_returns_src_nodes(self, linear_graph):
        """Forward edge backward pass should return src-side nodes."""
        # Query: a -> (forward) -> any
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return nodes a and b
        node_ids = set(result._nodes['id'].tolist())
        assert 'a' in node_ids  # start node
        assert 'b' in node_ids  # reached node

    def test_reverse_edge_returns_dst_nodes(self, linear_graph):
        """Reverse edge backward pass should return dst-side nodes."""
        # Query: d -> (reverse) -> any  (traverses against edge direction)
        chain = Chain([n({'id': 'd'}, name='start'), e_reverse(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return nodes d and c
        node_ids = set(result._nodes['id'].tolist())
        assert 'd' in node_ids  # start node
        assert 'c' in node_ids  # reached node (via reverse traversal)

    def test_undirected_edge_returns_both_endpoints(self, linear_graph):
        """Undirected edge should allow traversal in both directions."""
        # Query: b -> (undirected) -> any
        chain = Chain([n({'id': 'b'}, name='start'), e_undirected(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return b, a (backward), and c (forward)
        node_ids = set(result._nodes['id'].tolist())
        assert 'b' in node_ids
        assert 'a' in node_ids  # can reach via undirected
        assert 'c' in node_ids  # can reach via undirected

    def test_forward_filters_by_wavefront(self, branching_graph):
        """Forward should filter by valid dst wavefront."""
        # Query: a -> forward -> d only (not b or c)
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e'),
            n({'id': 'd'}, name='end')
        ])
        result = branching_graph.gfql(chain)

        # No edges since a doesn't directly connect to d
        assert len(result._edges) == 0

    def test_reverse_filters_by_wavefront(self, branching_graph):
        """Reverse should filter by valid src wavefront."""
        # Query: d -> reverse -> a only
        chain = Chain([
            n({'id': 'd'}, name='start'),
            e_reverse(name='e'),
            n({'id': 'a'}, name='end')
        ])
        result = branching_graph.gfql(chain)

        # No edges since d doesn't directly connect to a in reverse
        assert len(result._edges) == 0


class TestEdgeCases:
    """Test edge cases that could break the optimization."""

    def test_empty_forward_result(self, linear_graph):
        """Empty forward result should produce empty backward result."""
        # Query: nonexistent node -> forward -> any
        chain = Chain([n({'id': 'nonexistent'}), e_forward(), n()])
        result = linear_graph.gfql(chain)

        assert len(result._nodes) == 0
        assert len(result._edges) == 0

    def test_disconnected_components(self, disconnected_graph):
        """Should only traverse within connected component."""
        # Query from component 1
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = disconnected_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' not in node_ids  # different component
        assert 'd' not in node_ids  # different component

    def test_self_loop_edges(self, self_loop_graph):
        """Self-loop edges should be handled correctly."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = self_loop_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        edge_ids = set(result._edges['eid'].tolist())

        # Should include self-loop edge and both endpoints
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 0 in edge_ids  # self-loop
        assert 1 in edge_ids  # a -> b

    def test_parallel_edges(self, parallel_edges_graph):
        """Parallel edges should all be included."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = parallel_edges_graph.gfql(chain)

        edge_ids = set(result._edges['eid'].tolist())
        assert 0 in edge_ids
        assert 1 in edge_ids  # both parallel edges

    def test_cycle_traversal(self, cyclic_graph):
        """Cycles should be handled without infinite loops."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = cyclic_graph.gfql(chain)

        # Single hop from a should reach b
        node_ids = set(result._nodes['id'].tolist())
        assert 'a' in node_ids
        assert 'b' in node_ids


class TestResultCorrectness:
    """Test that optimized backward pass produces same results as original."""

    def test_tags_preserved_correctly(self, linear_graph):
        """Named aliases should produce correct boolean tags."""
        chain = Chain([
            n({'type': 'start'}, name='src'),
            e_forward(name='edge'),
            n(name='dst')
        ])
        result = linear_graph.gfql(chain)

        # Check node tags
        assert 'src' in result._nodes.columns
        src_tagged = result._nodes[result._nodes['src'] == True]['id'].tolist()
        assert src_tagged == ['a']

        # Check edge tags
        assert 'edge' in result._edges.columns
        edge_tagged = result._edges[result._edges['edge'] == True]['eid'].tolist()
        assert edge_tagged == [0]

    def test_attributes_preserved(self, linear_graph):
        """Node and edge attributes should be preserved."""
        chain = Chain([n(), e_forward(), n()])
        result = linear_graph.gfql(chain)

        # Node attributes
        assert 'type' in result._nodes.columns
        assert 'value' in result._nodes.columns

        # Edge attributes
        assert 'weight' in result._edges.columns

    def test_two_hop_chain_correctness(self, linear_graph):
        """Two-hop chain should produce correct results."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e1'),
            n(name='mid'),
            e_forward(name='e2'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        edge_ids = set(result._edges['eid'].tolist())

        assert node_ids == {'a', 'b', 'c'}
        assert edge_ids == {0, 1}

    def test_mixed_direction_chain(self, linear_graph):
        """Chain with mixed directions should work correctly."""
        # Start at b, go forward to c, then reverse to b
        # This tests that direction logic is correct for each step
        chain = Chain([
            n({'id': 'b'}, name='n1'),
            e_forward(name='e1'),
            n(name='n2'),
            e_reverse(name='e2'),
            n(name='n3')
        ])
        result = linear_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        # b -> c (forward), then c -> b (reverse of b->c edge)
        assert 'b' in node_ids
        assert 'c' in node_ids


# =============================================================================
# TestCombineStepsOptimization
# =============================================================================


class TestSingleHopOptimization:
    """Test that single-hop edges use endpoint filtering optimization."""

    def test_forward_filters_by_endpoints(self, linear_graph):
        """Forward edge should filter by src/dst endpoints correctly."""
        chain = Chain([n(), e_forward(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present
        assert len(result._edges) == 3

    def test_reverse_filters_by_endpoints(self, linear_graph):
        """Reverse edge should filter by endpoints correctly."""
        chain = Chain([n(), e_reverse(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present (just traversed in reverse)
        assert len(result._edges) == 3

    def test_undirected_filters_by_endpoints(self, linear_graph):
        """Undirected edge should filter by both endpoints."""
        chain = Chain([n(), e_undirected(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present
        assert len(result._edges) == 3


class TestHopLabelPreservation:
    """Test that hop labels are preserved correctly."""

    def test_node_hop_labels_preserved(self, linear_graph):
        """Node hop labels should be computed correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=2, label_node_hops='hop'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        assert 'hop' in result._nodes.columns

    def test_edge_hop_labels_preserved(self, linear_graph):
        """Edge hop labels should be computed correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=2, label_edge_hops='hop'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        assert 'hop' in result._edges.columns


class TestMultiStepChains:
    """Test multi-step chains with various configurations."""

    def test_three_hop_chain(self, linear_graph):
        """Three-hop chain should work correctly."""
        chain = Chain([
            n({'id': 'a'}, name='n1'),
            e_forward(name='e1'),
            n(name='n2'),
            e_forward(name='e2'),
            n(name='n3'),
            e_forward(name='e3'),
            n(name='n4')
        ])
        result = linear_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        assert node_ids == {'a', 'b', 'c', 'd'}

    def test_alternating_directions(self, linear_graph):
        """Alternating forward/reverse should work."""
        chain = Chain([
            n({'id': 'b'}, name='start'),
            e_forward(name='e1'),
            n(name='mid'),
            e_reverse(name='e2'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        # b -> c (forward), c -> b (reverse of b->c)
        node_ids = set(result._nodes['id'].tolist())
        assert 'b' in node_ids
        assert 'c' in node_ids


# =============================================================================
# TestChainDFExecutorParity
# =============================================================================


class TestBasicParity:
    """Test that chain produces same results with and without WHERE."""

    def test_same_nodes_with_and_without_where(self, linear_graph):
        """Node sets should match between chain and df_executor paths."""
        from graphistry.compute.gfql.same_path_types import col, compare

        ops = [n(name='a'), e_forward(name='e'), n(name='b')]

        # Without WHERE (uses chain.py)
        chain_no_where = Chain(ops)
        result_no_where = linear_graph.gfql(chain_no_where)

        # With trivial WHERE that doesn't filter (uses df_executor)
        # a.value <= b.value is always true since values increase
        where = [compare(col('a', 'value'), '<=', col('b', 'value'))]
        chain_with_where = Chain(ops, where=where)
        result_with_where = linear_graph.gfql(chain_with_where)

        nodes_no_where = set(result_no_where._nodes['id'].tolist())
        nodes_with_where = set(result_with_where._nodes['id'].tolist())

        assert nodes_no_where == nodes_with_where

    def test_same_edges_with_and_without_where(self, linear_graph):
        """Edge sets should match between chain and df_executor paths."""
        from graphistry.compute.gfql.same_path_types import col, compare

        ops = [n(name='a'), e_forward(name='e'), n(name='b')]

        chain_no_where = Chain(ops)
        result_no_where = linear_graph.gfql(chain_no_where)

        # a.value <= b.value is always true since values increase
        where = [compare(col('a', 'value'), '<=', col('b', 'value'))]
        chain_with_where = Chain(ops, where=where)
        result_with_where = linear_graph.gfql(chain_with_where)

        edges_no_where = set(result_no_where._edges['eid'].tolist())
        edges_with_where = set(result_with_where._edges['eid'].tolist())

        assert edges_no_where == edges_with_where


class TestComplexPatterns:
    """Test complex graph patterns."""

    def test_diamond_pattern(self, branching_graph):
        """Diamond pattern (a -> b,c -> d) should work correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e1'),
            n(name='mid'),
            e_forward(name='e2'),
            n({'id': 'd'}, name='end')
        ])
        result = branching_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        assert node_ids == {'a', 'b', 'c', 'd'}

        edge_ids = set(result._edges['eid'].tolist())
        assert edge_ids == {0, 1, 2, 3}  # all 4 edges

    def test_filtered_mid_node(self, branching_graph):
        """Filtering mid-node should reduce paths."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e1'),
            n({'type': 'left'}, name='mid'),  # only left branch
            e_forward(name='e2'),
            n(name='end')
        ])
        result = branching_graph.gfql(chain)

        node_ids = set(result._nodes['id'].tolist())
        assert 'a' in node_ids
        assert 'b' in node_ids  # left branch
        assert 'c' not in node_ids  # right branch filtered
        assert 'd' in node_ids


class TestWHEREVariants:
    """Test various WHERE clause configurations."""

    def test_adjacent_node_where(self, linear_graph):
        """WHERE on adjacent nodes should filter correctly."""
        from graphistry.compute.gfql.same_path_types import col, compare

        ops = [n(name='a'), e_forward(name='e'), n(name='b')]
        # Filter: a.value < b.value (always true for linear graph)
        where = [compare(col('a', 'value'), '<', col('b', 'value'))]

        chain = Chain(ops, where=where)
        result = linear_graph.gfql(chain)

        # All edges should pass since values increase
        assert len(result._edges) == 3

    def test_adjacent_node_where_filters(self, linear_graph):
        """WHERE should actually filter when condition fails."""
        from graphistry.compute.gfql.same_path_types import col, compare

        ops = [n(name='a'), e_forward(name='e'), n(name='b')]
        # Filter: a.value > b.value (never true for linear graph)
        where = [compare(col('a', 'value'), '>', col('b', 'value'))]

        chain = Chain(ops, where=where)
        result = linear_graph.gfql(chain)

        # No edges should pass
        assert len(result._edges) == 0
