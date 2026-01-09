"""
Tests for chain.py optimizations.

This module tests the backward pass optimization and combine_steps optimization
to ensure correctness across various edge cases.

The backward pass optimization (commit 12d89596) skips the full hop() call for
simple single-hop edges and uses vectorized merge filtering instead.

The combine_steps optimization filters edges by valid endpoints instead of
re-running the forward op.

###############################################################################
# IMPORTANT: NO XFAIL ALLOWED IN THIS FILE
#
# If a test fails, FIX THE BUG IN THE CODE. Do not use pytest.mark.xfail.
# Do not weaken assertions. Do not skip tests. Fix the actual implementation.
#
# This rule exists because AI assistants have repeatedly tried to mark failing
# tests as xfail instead of fixing the underlying bugs. This is not acceptable.
###############################################################################
"""

import pandas as pd
import pytest
from typing import Set

from graphistry.compute.ast import n, e_forward, e_reverse, e_undirected, ASTEdge
from graphistry.compute.chain import Chain

# Import test fixtures and cuDF parity helpers
from tests.gfql.ref.conftest import CGFull, maybe_cudf, to_list, to_set


# =============================================================================
# Test Fixtures (parametrized by engine_mode for pandas/cuDF parity testing)
# =============================================================================


def _make_linear_graph():
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


def _make_branching_graph():
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


def _make_cyclic_graph():
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


def _make_disconnected_graph():
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


def _make_self_loop_graph():
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


def _make_parallel_edges_graph():
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


@pytest.fixture
def linear_graph(engine_mode):
    """Linear graph: a -> b -> c -> d (parametrized by engine_mode)"""
    return maybe_cudf(_make_linear_graph(), engine_mode)


@pytest.fixture
def branching_graph(engine_mode):
    """Branching graph: a -> b, a -> c, b -> d, c -> d (parametrized by engine_mode)"""
    return maybe_cudf(_make_branching_graph(), engine_mode)


@pytest.fixture
def cyclic_graph(engine_mode):
    """Cyclic graph: a -> b -> c -> a (parametrized by engine_mode)"""
    return maybe_cudf(_make_cyclic_graph(), engine_mode)


@pytest.fixture
def disconnected_graph(engine_mode):
    """Disconnected graph: (a -> b) and (c -> d) with no connection (parametrized by engine_mode)"""
    return maybe_cudf(_make_disconnected_graph(), engine_mode)


@pytest.fixture
def self_loop_graph(engine_mode):
    """Graph with self-loop: a -> a, a -> b (parametrized by engine_mode)"""
    return maybe_cudf(_make_self_loop_graph(), engine_mode)


@pytest.fixture
def parallel_edges_graph(engine_mode):
    """Graph with parallel edges: a -> b (twice) (parametrized by engine_mode)"""
    return maybe_cudf(_make_parallel_edges_graph(), engine_mode)


# =============================================================================
# TestBackwardPassOptimization
# =============================================================================


class TestOptimizationEligibility:
    """Test that is_simple_single_hop correctly identifies eligible edges."""

    def test_single_hop_default_is_eligible(self):
        """Default e_forward() is eligible for optimization."""
        op = e_forward()
        assert op.is_simple_single_hop() is True

    def test_single_hop_explicit_is_eligible(self):
        """e_forward(hops=1) is eligible."""
        op = e_forward(hops=1)
        assert op.is_simple_single_hop() is True

    def test_single_hop_min_max_is_eligible(self):
        """e_forward(min_hops=1, max_hops=1) is eligible."""
        op = e_forward(min_hops=1, max_hops=1)
        assert op.is_simple_single_hop() is True

    def test_multihop_range_not_eligible(self):
        """e_forward(min_hops=1, max_hops=3) is NOT eligible."""
        op = e_forward(min_hops=1, max_hops=3)
        assert op.is_simple_single_hop() is False

    def test_multihop_fixed_not_eligible(self):
        """e_forward(hops=2) is NOT eligible."""
        op = e_forward(hops=2)
        assert op.is_simple_single_hop() is False

    def test_node_hop_labels_not_eligible(self):
        """e_forward(label_node_hops='hop') is NOT eligible."""
        op = e_forward(label_node_hops='hop')
        assert op.is_simple_single_hop() is False

    def test_edge_hop_labels_not_eligible(self):
        """e_forward(label_edge_hops='hop') is NOT eligible."""
        op = e_forward(label_edge_hops='hop')
        assert op.is_simple_single_hop() is False

    def test_seed_labels_not_eligible(self):
        """e_forward(label_seeds=True) is NOT eligible."""
        op = e_forward(label_seeds=True)
        assert op.is_simple_single_hop() is False

    def test_output_slice_not_eligible(self):
        """e_forward(output_min_hops=1) is NOT eligible."""
        op = e_forward(output_min_hops=1)
        assert op.is_simple_single_hop() is False

    def test_to_fixed_point_not_eligible(self):
        """e_forward(to_fixed_point=True) is NOT eligible (unbounded traversal)."""
        op = e_forward(to_fixed_point=True)
        assert op.is_simple_single_hop() is False

    def test_reverse_is_eligible(self):
        """e_reverse() is eligible."""
        op = e_reverse()
        assert op.is_simple_single_hop() is True

    def test_undirected_is_eligible(self):
        """e_undirected() is eligible."""
        op = e_undirected()
        assert op.is_simple_single_hop() is True


class TestDirectionSemantics:
    """Test that backward pass returns correct nodes for each direction."""

    def test_forward_edge_returns_src_nodes(self, linear_graph):
        """Forward edge backward pass should return src-side nodes."""
        # Query: a -> (forward) -> any
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return nodes a and b
        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids  # start node
        assert 'b' in node_ids  # reached node

    def test_reverse_edge_returns_dst_nodes(self, linear_graph):
        """Reverse edge backward pass should return dst-side nodes."""
        # Query: d -> (reverse) -> any  (traverses against edge direction)
        chain = Chain([n({'id': 'd'}, name='start'), e_reverse(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return nodes d and c
        node_ids = to_set(result._nodes['id'])
        assert 'd' in node_ids  # start node
        assert 'c' in node_ids  # reached node (via reverse traversal)

    def test_undirected_edge_returns_both_endpoints(self, linear_graph):
        """Undirected edge should allow traversal in both directions."""
        # Query: b -> (undirected) -> any
        chain = Chain([n({'id': 'b'}, name='start'), e_undirected(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        # Should return b, a (backward), and c (forward)
        node_ids = to_set(result._nodes['id'])
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

        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' not in node_ids  # different component
        assert 'd' not in node_ids  # different component

    def test_self_loop_edges(self, self_loop_graph):
        """Self-loop edges should be handled correctly."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = self_loop_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

        # Should include self-loop edge and both endpoints
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 0 in edge_ids  # self-loop
        assert 1 in edge_ids  # a -> b

    def test_parallel_edges(self, parallel_edges_graph):
        """Parallel edges should all be included."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = parallel_edges_graph.gfql(chain)

        edge_ids = to_set(result._edges['eid'])
        assert 0 in edge_ids
        assert 1 in edge_ids  # both parallel edges

    def test_cycle_traversal(self, cyclic_graph):
        """Cycles should be handled without infinite loops."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = cyclic_graph.gfql(chain)

        # Single hop from a should reach b
        node_ids = to_set(result._nodes['id'])
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
        src_tagged = to_list(result._nodes[result._nodes['src'] == True]['id'])
        assert src_tagged == ['a']

        # Check edge tags
        assert 'edge' in result._edges.columns
        edge_tagged = to_list(result._edges[result._edges['edge'] == True]['eid'])
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

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

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

        node_ids = to_set(result._nodes['id'])
        # b -> c (forward), then c -> b (reverse of b->c edge)
        assert 'b' in node_ids
        assert 'c' in node_ids


# =============================================================================
# TestFastPathBackwardPass
# =============================================================================
# These tests specifically exercise the fast path optimization in the backward
# pass that uses vectorized merge filtering instead of calling hop().
# Fast path is triggered when: op.is_simple_single_hop() returns True
# (i.e., hops=1, no labels, no output slicing)


class TestFastPathBackwardPassTopology:
    """Test fast path backward pass across different graph topologies."""

    def test_fast_path_linear_graph_forward(self, linear_graph):
        """Fast path on linear graph with forward edge."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

        assert node_ids == {'a', 'b'}
        assert edge_ids == {0}

    def test_fast_path_linear_graph_reverse(self, linear_graph):
        """Fast path on linear graph with reverse edge."""
        chain = Chain([n({'id': 'd'}, name='start'), e_reverse(name='e'), n(name='end')])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

        assert node_ids == {'c', 'd'}
        assert edge_ids == {2}  # c->d edge

    def test_fast_path_branching_graph(self, branching_graph):
        """Fast path on branching graph (diamond pattern)."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = branching_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # a -> b and a -> c
        assert node_ids == {'a', 'b', 'c'}
        assert len(result._edges) == 2

    def test_fast_path_cyclic_graph(self, cyclic_graph):
        """Fast path on cyclic graph."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = cyclic_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b'}
        assert len(result._edges) == 1

    def test_fast_path_disconnected_graph(self, disconnected_graph):
        """Fast path stays within connected component."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = disconnected_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b'}
        assert 'c' not in node_ids
        assert 'd' not in node_ids

    def test_fast_path_self_loop(self, self_loop_graph):
        """Fast path handles self-loop edges."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = self_loop_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 0 in edge_ids  # self-loop a->a
        assert 1 in edge_ids  # a->b

    def test_fast_path_parallel_edges(self, parallel_edges_graph):
        """Fast path handles parallel edges correctly."""
        chain = Chain([n({'id': 'a'}, name='start'), e_forward(name='e'), n(name='end')])
        result = parallel_edges_graph.gfql(chain)

        edge_ids = to_set(result._edges['eid'])
        # Both parallel edges should be included
        assert edge_ids == {0, 1}


class TestFastPathBackwardPassFiltering:
    """Test that fast path filters correctly based on node constraints."""

    def test_fast_path_filtered_end_node(self, linear_graph):
        """Fast path with filtered end node."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e'),
            n({'id': 'b'}, name='end')  # Only match b
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b'}
        assert len(result._edges) == 1

    def test_fast_path_no_matching_end(self, linear_graph):
        """Fast path when end node filter matches nothing reachable."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e'),
            n({'id': 'd'}, name='end')  # d is not reachable in 1 hop from a
        ])
        result = linear_graph.gfql(chain)

        assert len(result._edges) == 0

    def test_fast_path_type_filter(self, linear_graph):
        """Fast path with type-based node filter."""
        chain = Chain([
            n({'type': 'start'}, name='src'),
            e_forward(name='e'),
            n({'type': 'mid'}, name='dst')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids
        assert 'b' in node_ids  # b has type='mid'
        assert len(result._edges) == 1


class TestFastPathBackwardPassMultiStep:
    """Test fast path in multi-step chains (n->e->n->e->n)."""

    def test_fast_path_two_step_chain(self, linear_graph):
        """Two-step chain exercises fast path twice."""
        chain = Chain([
            n({'id': 'a'}, name='n1'),
            e_forward(name='e1'),
            n(name='n2'),
            e_forward(name='e2'),
            n(name='n3')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        edge_ids = to_set(result._edges['eid'])

        assert node_ids == {'a', 'b', 'c'}
        assert edge_ids == {0, 1}

    def test_fast_path_three_step_chain(self, linear_graph):
        """Three-step chain exercises fast path three times."""
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

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b', 'c', 'd'}
        assert len(result._edges) == 3

    def test_fast_path_mixed_directions_chain(self, linear_graph):
        """Chain with mixed forward/reverse directions."""
        chain = Chain([
            n({'id': 'b'}, name='n1'),
            e_forward(name='e1'),  # b -> c
            n(name='n2'),
            e_reverse(name='e2'),  # c <- b (follows b->c backward)
            n(name='n3')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'b' in node_ids
        assert 'c' in node_ids

    def test_fast_path_undirected_chain(self, linear_graph):
        """Chain with undirected edges.

        Without Cypher edge uniqueness:
        - Step 1: from b, undirected reaches a (via e0) and c (via e1)
        - Step 2: from {a,c}:
          - from a: undirected reaches b (via e0)
          - from c: undirected reaches b (via e1) and d (via e2)
        - All reachable nodes: {a, b, c, d}

        NOTE: Cypher DIFFERENT_RELATIONSHIPS uniqueness (edges can't repeat in path)
        is not currently implemented. With edge uniqueness, only {b,c,d} would be valid.
        See: https://neo4j.com/docs/cypher-manual/4.3/introduction/uniqueness/
        """
        chain = Chain([
            n({'id': 'b'}, name='n1'),
            e_undirected(name='e1'),
            n(name='n2'),
            e_undirected(name='e2'),
            n(name='n3')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # Without edge uniqueness: all reachable nodes
        assert node_ids == {'a', 'b', 'c', 'd'}, f"Expected {{a,b,c,d}}, got {node_ids}"


class TestFastPathBackwardPassTags:
    """Test that fast path preserves tags correctly."""

    def test_fast_path_node_tags_correct(self, linear_graph):
        """Fast path sets node tags correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        assert 'start' in result._nodes.columns
        assert 'end' in result._nodes.columns

        # Check specific tags
        start_nodes = to_list(result._nodes[result._nodes['start'] == True]['id'])
        end_nodes = to_list(result._nodes[result._nodes['end'] == True]['id'])

        assert start_nodes == ['a']
        assert 'b' in end_nodes

    def test_fast_path_edge_tags_correct(self, linear_graph):
        """Fast path sets edge tags correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='my_edge'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        assert 'my_edge' in result._edges.columns
        tagged_edges = to_list(result._edges[result._edges['my_edge'] == True]['eid'])
        assert 0 in tagged_edges  # The a->b edge

    def test_fast_path_multi_step_tags(self, linear_graph):
        """Tags correct across multi-step fast path chain."""
        chain = Chain([
            n({'id': 'a'}, name='first'),
            e_forward(name='edge1'),
            n(name='middle'),
            e_forward(name='edge2'),
            n(name='last')
        ])
        result = linear_graph.gfql(chain)

        # Check node tags
        first_nodes = to_list(result._nodes[result._nodes['first'] == True]['id'])
        middle_nodes = to_list(result._nodes[result._nodes['middle'] == True]['id'])
        last_nodes = to_list(result._nodes[result._nodes['last'] == True]['id'])

        assert first_nodes == ['a']
        assert 'b' in middle_nodes
        assert 'c' in last_nodes

        # Check edge tags
        edge1_tagged = to_list(result._edges[result._edges['edge1'] == True]['eid'])
        edge2_tagged = to_list(result._edges[result._edges['edge2'] == True]['eid'])

        assert 0 in edge1_tagged  # a->b
        assert 1 in edge2_tagged  # b->c


# =============================================================================
# TestFastPathCombineSteps
# =============================================================================
# These tests specifically exercise the fast path in combine_steps that uses
# endpoint filtering instead of re-running the forward op.
# Fast path is triggered when has_multihop=False (all edges are single-hop)


class TestFastPathCombineStepsBasic:
    """Basic tests for combine_steps fast path."""

    def test_fast_path_forward_filters_by_endpoints(self, linear_graph):
        """Forward edge should filter by src/dst endpoints correctly."""
        chain = Chain([n(), e_forward(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present
        assert len(result._edges) == 3

    def test_fast_path_reverse_filters_by_endpoints(self, linear_graph):
        """Reverse edge should filter by endpoints correctly."""
        chain = Chain([n(), e_reverse(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present (just traversed in reverse)
        assert len(result._edges) == 3

    def test_fast_path_undirected_filters_by_endpoints(self, linear_graph):
        """Undirected edge should filter by both endpoints."""
        chain = Chain([n(), e_undirected(), n()])
        result = linear_graph.gfql(chain)

        # All edges should be present
        assert len(result._edges) == 3


class TestFastPathCombineStepsFiltering:
    """Test fast path combine_steps with various filtering scenarios."""

    def test_fast_path_node_filter_reduces_edges(self, branching_graph):
        """Node filter in middle should reduce edges via endpoint filtering."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e1'),
            n({'type': 'left'}, name='mid'),  # Only left branch (b)
            e_forward(name='e2'),
            n(name='end')
        ])
        result = branching_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' not in node_ids  # Right branch filtered
        assert 'd' in node_ids

    def test_fast_path_sink_filter(self, branching_graph):
        """Filter to specific sink node."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e1'),
            n(name='mid'),
            e_forward(name='e2'),
            n({'id': 'd'}, name='end')  # Only reach d
        ])
        result = branching_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b', 'c', 'd'}

    def test_fast_path_unreachable_filter(self, linear_graph):
        """Filter that makes target unreachable produces empty result."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(name='e'),
            n({'id': 'd'}, name='end')  # d is 3 hops away, not 1
        ])
        result = linear_graph.gfql(chain)

        assert len(result._edges) == 0


class TestFastPathCombineStepsEdgeAttributes:
    """Test that fast path preserves edge attributes correctly."""

    def test_fast_path_preserves_edge_weight(self, linear_graph):
        """Edge attributes like weight should be preserved."""
        chain = Chain([n(), e_forward(), n()])
        result = linear_graph.gfql(chain)

        assert 'weight' in result._edges.columns
        weights = to_list(result._edges['weight'])
        assert 1.0 in weights
        assert 2.0 in weights
        assert 3.0 in weights

    def test_fast_path_preserves_custom_attributes(self, branching_graph):
        """Custom edge attributes (like 'branch') should be preserved."""
        chain = Chain([n(), e_forward(), n()])
        result = branching_graph.gfql(chain)

        assert 'branch' in result._edges.columns
        branches = to_set(result._edges['branch'])
        assert 'left' in branches
        assert 'right' in branches


# =============================================================================
# TestCombineStepsOptimization (Original - kept for backwards compatibility)
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

        node_ids = to_set(result._nodes['id'])
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
        node_ids = to_set(result._nodes['id'])
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

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b', 'c', 'd'}

        edge_ids = to_set(result._edges['eid'])
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

        node_ids = to_set(result._nodes['id'])
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


# =============================================================================
# TestSlowPathVariants
# =============================================================================
# These tests use multi-hop or labels to force the slow path (non-optimized).
# They mirror fast-path tests to ensure both paths produce correct results.


class TestSlowPathBackwardPass:
    """
    Test backward pass with multi-hop edges (slow path).

    These tests force the slow path by using min_hops/max_hops > 1 or labels,
    which disables the is_simple_single_hop() optimization.
    """

    def test_multihop_forward_reaches_correct_nodes(self, linear_graph):
        """Multi-hop forward should reach nodes at all hop distances."""
        # a -> b -> c (1-2 hops from a)
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=2, name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids  # start
        assert 'b' in node_ids  # 1 hop
        assert 'c' in node_ids  # 2 hops
        # d is 3 hops away, so shouldn't be included
        assert 'd' not in node_ids

    def test_multihop_reverse_reaches_correct_nodes(self, linear_graph):
        """Multi-hop reverse should traverse against edge direction."""
        # d <- c <- b (1-2 hops from d in reverse)
        chain = Chain([
            n({'id': 'd'}, name='start'),
            e_reverse(min_hops=1, max_hops=2, name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'd' in node_ids  # start
        assert 'c' in node_ids  # 1 hop reverse
        assert 'b' in node_ids  # 2 hops reverse
        # a is 3 hops away in reverse
        assert 'a' not in node_ids

    def test_labeled_edges_preserve_hop_info(self, linear_graph):
        """Edge with labels should preserve hop information."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, label_edge_hops='hop', name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        # Check hop column exists and has correct values
        assert 'hop' in result._edges.columns
        hops = to_list(result._edges['hop'])
        assert 1 in hops
        assert 2 in hops
        assert 3 in hops

    def test_labeled_nodes_preserve_hop_info(self, linear_graph):
        """Nodes with labels should preserve hop information.

        Note: By default label_seeds=False, so seed node 'a' has hop=NA.
        Use label_seeds=True to get hop=0 for seed nodes.
        """
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, label_node_hops='hop', name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        assert 'hop' in result._nodes.columns
        # Non-seed nodes should have hop values 1, 2, 3
        hop_df = result._nodes[['id', 'hop']].dropna(subset=['hop'])
        hop_values = to_set(hop_df['hop'])
        assert 1 in hop_values or 2 in hop_values or 3 in hop_values, "Should have hop labels for reachable nodes"

    def test_disconnected_multihop(self, disconnected_graph):
        """Multi-hop should stay within connected component."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=5, name='e'),  # Try to reach far
            n(name='end')
        ])
        result = disconnected_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' not in node_ids  # Different component
        assert 'd' not in node_ids


class TestSlowPathCombineSteps:
    """
    Test combine_steps with multi-hop edges (slow path).

    These tests force has_multihop=True which uses the full hop() call
    instead of endpoint filtering.
    """

    def test_multihop_then_single_hop(self, linear_graph):
        """Chain with multi-hop followed by single-hop."""
        chain = Chain([
            n({'id': 'a'}, name='n1'),
            e_forward(min_hops=1, max_hops=2, name='e1'),  # Slow path
            n(name='n2'),
            e_forward(name='e2'),  # Would be fast but chain has multihop
            n(name='n3')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # a -> b,c (1-2 hops) -> c,d (1 more hop)
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' in node_ids
        assert 'd' in node_ids

    def test_alternating_directions_multihop(self, linear_graph):
        """Alternating directions with multi-hop."""
        chain = Chain([
            n({'id': 'b'}, name='start'),
            e_forward(min_hops=1, max_hops=2, name='e1'),
            n(name='mid'),
            e_reverse(name='e2'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        # b -> c,d then reverse
        node_ids = to_set(result._nodes['id'])
        assert 'b' in node_ids
        assert 'c' in node_ids
        assert 'd' in node_ids

    def test_diamond_pattern_multihop(self, branching_graph):
        """Diamond pattern with multi-hop edge."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=2, name='e'),  # Can reach d in 2 hops
            n(name='end')
        ])
        result = branching_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        assert node_ids == {'a', 'b', 'c', 'd'}


class TestSlowPathEdgeCases:
    """Edge cases that exercise the slow path."""

    def test_empty_result_multihop(self, linear_graph):
        """Empty result with multi-hop should produce empty backward result."""
        chain = Chain([
            n({'id': 'nonexistent'}),
            e_forward(min_hops=1, max_hops=3),
            n()
        ])
        result = linear_graph.gfql(chain)

        assert len(result._nodes) == 0
        assert len(result._edges) == 0

    def test_self_loop_multihop(self, self_loop_graph):
        """Self-loop with multi-hop should handle correctly."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=2, name='e'),
            n(name='end')
        ])
        result = self_loop_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # Can reach a via self-loop and b via a->b
        assert 'a' in node_ids
        assert 'b' in node_ids

    def test_cycle_multihop(self, cyclic_graph):
        """Cycle with multi-hop should not infinite loop."""
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=5, name='e'),  # High max to test cycle handling
            n(name='end')
        ])
        result = cyclic_graph.gfql(chain)

        # Should complete without infinite loop and reach all nodes
        node_ids = to_set(result._nodes['id'])
        assert 'a' in node_ids
        assert 'b' in node_ids
        assert 'c' in node_ids


class TestSlowPathParity:
    """
    Verify slow path produces same results as fast path for equivalent queries.
    """

    def test_single_hop_vs_explicit_range(self, linear_graph):
        """e_forward() should equal e_forward(min_hops=1, max_hops=1)."""
        # Fast path
        chain_fast = Chain([n(), e_forward(), n()])
        result_fast = linear_graph.gfql(chain_fast)

        # Slow path (explicit min/max triggers different code)
        chain_slow = Chain([n(), e_forward(min_hops=1, max_hops=1), n()])
        result_slow = linear_graph.gfql(chain_slow)

        # Results should be identical
        fast_nodes = to_set(result_fast._nodes['id'])
        slow_nodes = to_set(result_slow._nodes['id'])
        assert fast_nodes == slow_nodes

        fast_edges = to_set(result_fast._edges['eid'])
        slow_edges = to_set(result_slow._edges['eid'])
        assert fast_edges == slow_edges

    def test_direction_semantics_preserved_multihop(self, linear_graph):
        """Direction semantics should be same for single and multi-hop."""
        # Fast path forward
        chain_fast = Chain([n({'id': 'a'}), e_forward(), n()])
        result_fast = linear_graph.gfql(chain_fast)

        # Slow path forward (range that includes only 1 hop)
        chain_slow = Chain([n({'id': 'a'}), e_forward(min_hops=1, max_hops=1), n()])
        result_slow = linear_graph.gfql(chain_slow)

        fast_nodes = to_set(result_fast._nodes['id'])
        slow_nodes = to_set(result_slow._nodes['id'])
        assert fast_nodes == slow_nodes

    def test_reverse_direction_parity(self, linear_graph):
        """Reverse direction should work same in fast and slow paths."""
        # Fast path reverse
        chain_fast = Chain([n({'id': 'd'}), e_reverse(), n()])
        result_fast = linear_graph.gfql(chain_fast)

        # Slow path reverse
        chain_slow = Chain([n({'id': 'd'}), e_reverse(min_hops=1, max_hops=1), n()])
        result_slow = linear_graph.gfql(chain_slow)

        fast_nodes = to_set(result_fast._nodes['id'])
        slow_nodes = to_set(result_slow._nodes['id'])
        assert fast_nodes == slow_nodes

    def test_undirected_parity(self, linear_graph):
        """Undirected should work same in fast and slow paths."""
        # Fast path undirected
        chain_fast = Chain([n({'id': 'b'}), e_undirected(), n()])
        result_fast = linear_graph.gfql(chain_fast)

        # Slow path undirected
        chain_slow = Chain([n({'id': 'b'}), e_undirected(min_hops=1, max_hops=1), n()])
        result_slow = linear_graph.gfql(chain_slow)

        fast_nodes = to_set(result_fast._nodes['id'])
        slow_nodes = to_set(result_slow._nodes['id'])
        assert fast_nodes == slow_nodes


class TestOutputSlicing:
    """
    Test output_min_hops and output_max_hops (forces slow path).

    ###########################################################################
    # IMPORTANT: NO XFAIL ALLOWED IN THESE TESTS
    #
    # If a test fails, FIX THE BUG IN chain.py or hop.py.
    # Do NOT use pytest.mark.xfail. Do NOT weaken assertions.
    # AI assistants: you must fix the actual code, not mark tests as expected failures.
    ###########################################################################
    """

    def test_output_min_hops_filters_early_hops(self, linear_graph):
        """output_min_hops filters edges by hop number, keeping all their endpoints.

        With output_min_hops=2:
        - Edges at hop 2+ are kept: b->c (hop 2), c->d (hop 3)
        - All nodes on these edges are included: {b, c, d}
        - Seed 'a' is NOT included because it's not on any output edge

        Expected: {b, c, d} - all endpoints of edges at hop 2+
        """
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # All endpoints of hop 2+ edges should be included
        assert 'b' in node_ids, "b (source of hop 2 edge) should be in result"
        assert 'c' in node_ids, "c (hop 2 destination, hop 3 source) should be in result"
        assert 'd' in node_ids, "d (hop 3 destination) should be in result"
        # Seed 'a' is NOT on any output edge
        assert 'a' not in node_ids, "a is not on any hop 2+ edge"

    def test_output_max_hops_filters_late_hops(self, linear_graph):
        """output_max_hops filters edges by hop number, keeping all their endpoints.

        With output_max_hops=2:
        - Edges at hop 1-2 are kept: a->b (hop 1), b->c (hop 2)
        - All nodes on these edges are included: {a, b, c}

        Expected: {a, b, c} - all endpoints of edges at hop <=2
        """
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, output_max_hops=2, name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # All endpoints of hop <=2 edges should be included
        assert 'a' in node_ids, "a (source of hop 1 edge) should be in result"
        assert 'b' in node_ids, "b (hop 1 dest, hop 2 source) should be in result"
        assert 'c' in node_ids, "c (hop 2 destination) should be in result"
        # d is only on hop 3 edge
        assert 'd' not in node_ids, "d (only on hop 3 edge) should be filtered"

    def test_output_slice_both_bounds(self, linear_graph):
        """Both output_min_hops and output_max_hops together.

        With output_min_hops=2, output_max_hops=2:
        - Only edge at exactly hop 2 is kept: b->c
        - All nodes on this edge are included: {b, c}

        Expected: {b, c} - endpoints of hop=2 edge only
        """
        chain = Chain([
            n({'id': 'a'}, name='start'),
            e_forward(min_hops=1, max_hops=3, output_min_hops=2, output_max_hops=2, name='e'),
            n(name='end')
        ])
        result = linear_graph.gfql(chain)

        node_ids = to_set(result._nodes['id'])
        # Only endpoints of hop 2 edge
        assert 'b' in node_ids, "b (source of hop 2 edge) should be in result"
        assert 'c' in node_ids, "c (destination of hop 2 edge) should be in result"
        # Filtered out - not on hop 2 edge
        assert 'a' not in node_ids, "a is not on hop 2 edge"
        assert 'd' not in node_ids, "d is not on hop 2 edge"
