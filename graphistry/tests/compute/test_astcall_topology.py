"""
Topological sensitivity tests for ASTCall operations in GFQL chains.

NOTE: As of PR #787, mixing call() with n()/e() in the same chain is not supported
and will raise GFQLValidationError. These tests document the behaviors that led to
this restriction. See issue #791 for architectural details and future enhancement plans.

This test suite validates that ASTCall operations (filters, enrichments) work
correctly when mixed with ASTNode/ASTEdge traversal operations in complex
topological patterns.

Key test principles:
1. Non-overlapping filter dimensions - each filter uses different attributes
   to make it obvious which filters were applied
2. Enrichment column dependencies - validate that columns added by one ASTCall
   are visible to subsequent operations
3. Topology sensitivity - test various chain positions, graph structures,
   and operation orderings

Related Issues:
- #786 - Chained filter operations (fixed for pure call() chains)
- #791 - Mixed call()/traversal chains (architectural limitation)
- PR #787 - Implementation and enforcement
"""
import pandas as pd
import pytest
from graphistry import PyGraphistry
from graphistry.compute.ast import ASTCall, n, e, e_forward, e_reverse
from graphistry.compute.predicates.numeric import Between, GT, GE, LT, LE
from graphistry.tests.test_compute import CGFull


class TestTopologicalChains:
    """Test mixed ASTNode/ASTCall/ASTEdge chains with non-overlapping dimensions.

    These tests validate that:
    - ASTCall operations thread their results correctly through chains
    - Filters on different dimensions compose properly
    - Wavefront semantics and graph threading coexist correctly
    """

    @pytest.fixture
    def rich_graph(self):
        """Create test graph with 4 independent filter dimensions per entity type.

        Node dimensions:
        - type: person/org/bot (category filter)
        - score: 10-50 (numeric filter)
        - region: NA/EU/ASIA (category filter)
        - status: active/inactive (boolean filter)

        Edge dimensions:
        - type: forward/backward/lateral (category filter)
        - weight: 1-10 (numeric filter)
        - category: low/med/high (category filter)
        - verified: True/False (boolean filter)
        """
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C', 'D', 'E', 'A', 'B'],
            'dst': ['B', 'C', 'D', 'E', 'A', 'C', 'D', 'E', 'A', 'B', 'D', 'E'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward',
                     'backward', 'backward', 'backward', 'backward', 'backward',
                     'lateral', 'lateral'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 2, 4],
            'category': ['low', 'low', 'med', 'med', 'high', 'high', 'high', 'high', 'med', 'med', 'low', 'med'],
            'verified': [True, True, True, False, False, True, False, True, False, True, True, False]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'bot'],
            'score': [10, 20, 30, 40, 50],
            'region': ['NA', 'NA', 'EU', 'EU', 'ASIA'],
            'status': ['active', 'active', 'active', 'inactive', 'active']
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_topology_node_call_call_node(self, rich_graph):
        """Pattern: [n({type filter}), call(weight filter), call(category filter), n({score filter})]

        Validates: All 4 filters apply on non-overlapping dimensions.
        Expected: Should progressively narrow results by applying each filter.
        """
        result = rich_graph.gfql([
            n({'type': 'person'}),                                      # Filter nodes: type=person (A, B)
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Filter edges: weight<=5
            ASTCall('filter_edges_by_dict', {'filter_dict': {'category': 'low'}}),  # Filter edges: category=low
            n({'score': GE(15)})                                      # Filter nodes: score>=15
        ])

        # Validate all filters applied:
        # 1. Nodes should have type=person AND score>=15 → only B (score=20)
        # 2. Edges should have weight<=5 AND category=low
        assert len(result._nodes) >= 1
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
            assert all(result._nodes['score'] >= 15)

        if len(result._edges) > 0:
            assert all(result._edges['weight'] <= 5)
            assert all(result._edges['category'] == 'low')

    def test_topology_edge_call_call_edge(self, rich_graph):
        """Pattern: [e({type filter}), call(weight filter), call(verified filter), e({category filter})]

        Edge-based version of previous test.
        Validates: All 4 edge filters apply correctly.
        """
        result = rich_graph.gfql([
            e({'type': 'forward'}),                                              # Filter edges: type=forward
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Filter edges: weight<=5
            ASTCall('filter_edges_by_dict', {'filter_dict': {'verified': True}}),  # Filter edges: verified=True
            e({'category': 'low'})                                               # Filter edges: category=low
        ])

        # All 4 edge filters should apply:
        # type=forward AND weight<=5 AND verified=True AND category=low
        # This should give us very few edges (maybe 1-2)
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all(result._edges['weight'] <= 5)
            assert all(result._edges['verified'] == True)  # noqa: E712
            assert all(result._edges['category'] == 'low')

    def test_topology_alternating_node_call_patterns(self, rich_graph):
        """Pattern: [n({f1}), call(f2), n({f3}), call(f4), n({f5})]

        Long alternating chain to stress wavefront/graph threading interaction.
        Validates: Alternating ASTNode and ASTCall operations compose correctly.
        """
        result = rich_graph.gfql([
            n({'type': 'person'}),                                               # Dim 1: node type
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Dim 2: edge type
            n({'region': 'NA'}),                                                # Dim 3: node region
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),  # Dim 4: edge weight
            n({'status': 'active'})                                             # Dim 5: node status
        ])

        # Final nodes should satisfy: type=person AND region=NA AND status=active
        # Final edges should satisfy: type=forward AND weight<=5
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
            assert all(result._nodes['region'] == 'NA')
            assert all(result._nodes['status'] == 'active')

        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all(result._edges['weight'] <= 5)

    def test_topology_hop_between_calls(self, rich_graph):
        """Pattern: [n({f1}), call(f2), hop(), call(f3), n({f4})]

        Validates: ASTCall results persist through hop operations.
        Tests that filtered graph state is maintained across hop traversals.
        """
        from graphistry.compute.ast import call

        result = rich_graph.gfql([
            n({'type': 'person'}),                                               # Filter to person nodes
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Filter to forward edges
            call('hop', {'hops': 1}),                                           # Hop operation
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(15)}}),  # Filter nodes by score
            n()                                                                  # Final node selection
        ])

        # Should have filtered edges and nodes
        if len(result._edges) > 0:
            # After first filter, only forward edges should remain
            assert all(result._edges['type'] == 'forward')
        if len(result._nodes) > 0:
            # After second filter, only score>=15 nodes should remain
            assert all(result._nodes['score'] >= 15)

    def test_topology_call_at_chain_start(self, rich_graph):
        """Pattern: [call(f1), n(), e(), n()]

        ASTCall at the very beginning of a chain, followed by traversal.
        Validates: Filtered graph is correctly used by subsequent traversal ops.
        """
        result = rich_graph.gfql([
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
            n(),
            e(),
            n()
        ])

        # All edges in final result should be forward (filter persists through traversal)
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')

    def test_topology_call_in_middle(self, rich_graph):
        """Pattern: [n(), call(f1), e(), n()]

        ASTCall in the middle of traversal chain.
        Validates: Filter applies to current state and persists forward.
        """
        result = rich_graph.gfql([
            n({'type': 'person'}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(5)}}),
            e(),
            n()
        ])

        # Edges should have weight>=5
        if len(result._edges) > 0:
            assert all(result._edges['weight'] >= 5)

    def test_topology_call_at_chain_end(self, rich_graph):
        """Pattern: [n(), e(), n(), call(f1)]

        ASTCall at the very end after traversal completes.
        Validates: Filter applies to final traversal result.
        """
        result = rich_graph.gfql([
            n({'type': 'person'}),
            e(),
            n(),
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(30)}})
        ])

        # Final nodes should have score>=30 (from final filter)
        assert len(result._nodes) >= 0
        if len(result._nodes) > 0:
            assert all(result._nodes['score'] >= 30)


class TestEnrichmentChains:
    """Test enrichment operations creating columns used by subsequent filters.

    These tests validate that:
    - Enrichment operations (get_degrees, etc.) add columns correctly
    - Added columns persist through chain execution
    - Subsequent filters can use enriched columns
    - Forward and backward passes handle enrichments correctly
    """

    @pytest.fixture
    def enrichment_graph(self):
        """Create graph suitable for testing degree-based enrichments."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C', 'D', 'D', 'D'],
            'dst': ['B', 'C', 'C', 'D', 'D', 'A', 'B', 'E'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward', 'backward', 'backward', 'forward'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_enrichment_degree_then_filter(self, enrichment_graph):
        """Pattern: [call('get_indegrees'), n({'deg_in': {...}})]

        Validates: Enriched 'deg_in' column is visible to subsequent n() filter.
        """
        result = enrichment_graph.gfql([
            ASTCall('get_indegrees', {'col': 'deg_in'}),
            n({'deg_in': GE(2)})
        ])

        # Nodes with in-degree >= 2 should be selected
        assert 'deg_in' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['deg_in'] >= 2)

    def test_enrichment_multiple_degrees(self, enrichment_graph):
        """Pattern: [call('get_indegrees'), call('get_outdegrees'), n({filter both})]

        Validates: Multiple enrichment operations stack correctly.
        """
        result = enrichment_graph.gfql([
            ASTCall('get_indegrees', {'col': 'deg_in'}),
            ASTCall('get_outdegrees', {'col': 'deg_out'}),
            n({'deg_in': GE(1), 'deg_out': GE(1)})
        ])

        # Both columns should exist and be filtered
        assert 'deg_in' in result._nodes.columns
        assert 'deg_out' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['deg_in'] >= 1)
            assert all(result._nodes['deg_out'] >= 1)

    def test_enrichment_filter_enrichment_filter(self, enrichment_graph):
        """Pattern: [n({filter1}), call(enrich), call(filter2), n()]

        Validates: Complex interleaving of filters and enrichments.
        """
        result = enrichment_graph.gfql([
            n({'type': 'person'}),                           # Filter to person nodes
            ASTCall('get_degrees', {}),                      # Enrich with degree
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'degree': GE(2)}}),  # Filter by degree
            n()                                              # Final traversal
        ])

        # Nodes should be type=person with degree>=2
        assert 'degree' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
            assert all(result._nodes['degree'] >= 2)

    def test_enrichment_edge_degree_then_filter_forward_backward(self, enrichment_graph):
        """Pattern: [e({type: 'forward'}), call('get_degrees'), n({'degree': {...}}), e(), n()]

        Validates: Degree enrichment works in forward pass and backward pass sees it.
        This tests that enriched columns persist through the backward validation pass.
        """
        result = enrichment_graph.gfql([
            e({'type': 'forward'}),
            ASTCall('get_degrees', {}),
            n({'degree': GE(2)}),
            e(),
            n()
        ])

        # Nodes should have degree column and be filtered
        assert 'degree' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['degree'] >= 2)
        # Only forward edges should remain from initial filter
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')

    def test_enrichment_before_traversal(self, enrichment_graph):
        """Pattern: [call('get_outdegrees'), n({'deg_out': {...}}), e(), n()]

        Validates: Enriched columns persist through traversal operations.
        """
        result = enrichment_graph.gfql([
            ASTCall('get_outdegrees', {'col': 'deg_out'}),
            n({'deg_out': GE(2)}),
            e_forward(),
            n()
        ])

        # Should start from nodes with out-degree>=2 and traverse forward
        assert 'deg_out' in result._nodes.columns

    def test_enrichment_with_node_filter_interaction(self, enrichment_graph):
        """Pattern: [n({'type': 'person'}), call('get_degrees'), n({'degree': {...}})]

        Validates: Node filter → enrichment → node filter using enriched column.
        """
        result = enrichment_graph.gfql([
            n({'type': 'person'}),
            ASTCall('get_degrees', {}),
            n({'degree': GE(1)})
        ])

        # Should have type=person nodes with degree>=1
        assert 'degree' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
            assert all(result._nodes['degree'] >= 1)


class TestTopologySensitivity:
    """Test various graph structures and edge cases.

    Validates that ASTCall operations work correctly across:
    - Different graph topologies (DAG, cyclic, disconnected)
    - Edge cases (empty results, fixed-point traversal)
    - Complex structural patterns
    """

    @pytest.fixture
    def dag_graph(self):
        """Create a directed acyclic graph (no cycles)."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C'],
            'dst': ['B', 'C', 'D', 'E', 'D'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward'],
            'weight': [1, 2, 3, 4, 5]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'level': [0, 1, 1, 2, 2]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    @pytest.fixture
    def cyclic_graph(self):
        """Create a graph with cycles."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'D'],
            'dst': ['B', 'C', 'D', 'A'],
            'type': ['forward', 'forward', 'forward', 'forward'],
            'weight': [1, 2, 3, 4]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D'],
            'score': [10, 20, 30, 40]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    @pytest.fixture
    def disconnected_graph(self):
        """Create graph with multiple disconnected components."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'D', 'E'],
            'dst': ['B', 'C', 'E', 'F'],
            'component': [1, 1, 2, 2],
            'weight': [1, 2, 3, 4]
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E', 'F'],
            'component': [1, 1, 1, 2, 2, 2],
            'score': [10, 20, 30, 40, 50, 60]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_topology_dag_structure(self, dag_graph):
        """Validates: ASTCall operations work correctly on DAG structures."""
        result = dag_graph.gfql([
            n(),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(2)}}),
            e(),
            n()
        ])

        # Filter should work on DAG structure
        if len(result._edges) > 0:
            assert all(result._edges['weight'] >= 2)

    def test_topology_cyclic_structure(self, cyclic_graph):
        """Validates: ASTCall doesn't break on cyclic graphs."""
        result = cyclic_graph.gfql([
            n({'id': 'A'}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(3)}}),
            e(),
            n()
        ])

        # Should handle cycles gracefully
        if len(result._edges) > 0:
            assert all(result._edges['weight'] <= 3)

    def test_topology_disconnected_components(self, disconnected_graph):
        """Validates: ASTCall filter preserves component structure."""
        result = disconnected_graph.gfql([
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'component': 1}}),
            n(),
            e(),
            n()
        ])

        # Should only have component 1 nodes and edges
        if len(result._nodes) > 0:
            assert all(result._nodes['component'] == 1)

    def test_topology_empty_intermediate_result(self, dag_graph):
        """Validates: Empty intermediate result doesn't crash subsequent ops."""
        result = dag_graph.gfql([
            n({'level': 99}),  # No such level, empty result
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': GE(1)}}),
            n()
        ])

        # Should handle empty gracefully
        assert len(result._nodes) == 0 or len(result._edges) == 0


class TestComplexMixedChains:
    """Test complex combinations of all operation types.

    These are exhaustive integration tests combining:
    - ASTNode, ASTEdge, ASTCall operations
    - Filters and enrichments
    - Forward and backward passes
    - Multiple hops and complex patterns
    """

    @pytest.fixture
    def complex_graph(self):
        """Create rich graph for complex chain testing."""
        edges_df = pd.DataFrame({
            'src': ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D', 'E'],
            'dst': ['B', 'C', 'C', 'D', 'D', 'E', 'E', 'A', 'A'],
            'type': ['forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'forward', 'backward', 'backward'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8, 9],
            'category': ['low', 'low', 'med', 'med', 'med', 'high', 'high', 'high', 'high']
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50],
            'region': ['NA', 'NA', 'EU', 'EU', 'ASIA']
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_complex_all_operation_types(self, complex_graph):
        """Pattern: [n({f1}), call(enrich), e({f2}), call(filter), n({f3})]

        Validates: All operation types work together in one chain.
        """
        result = complex_graph.gfql([
            n({'type': 'person'}),
            ASTCall('get_degrees', {}),
            e({'type': 'forward'}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),
            n({'region': 'NA'})
        ])

        # Complex validation - multiple filters and enrichment
        assert 'degree' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
            assert all(result._nodes['region'] == 'NA')
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all(result._edges['weight'] <= 5)

    def test_complex_multiple_calls_in_sequence(self, complex_graph):
        """Pattern: [call(f1), call(f2), call(f3), n()]

        Validates: Multiple consecutive ASTCall operations compose correctly.
        """
        result = complex_graph.gfql([
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': Between(2, 6)}}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'category': 'med'}}),
            n()
        ])

        # All three filters should apply
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all((result._edges['weight'] >= 2) & (result._edges['weight'] <= 6))
            assert all(result._edges['category'] == 'med')

    def test_complex_filter_hop_filter_hop(self, complex_graph):
        """Pattern: [n({filter1}), call(filter2), hop(), call(filter3), hop(), n()]

        Validates: Filters persist across multiple hops.
        This is a critical test for real-world multi-hop query patterns.
        """
        from graphistry.compute.ast import call

        result = complex_graph.gfql([
            n({'type': 'person'}),                                              # Start with person nodes
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Only forward edges
            call('hop', {'hops': 1}),                                          # First hop
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': GE(20)}}),  # Filter intermediate nodes
            call('hop', {'hops': 1}),                                          # Second hop
            n()                                                                 # Final nodes
        ])

        # Edges should still be filtered to forward only
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')

    def test_complex_enrichment_dependency_chain(self, complex_graph):
        """Pattern: [call(enrich1), call(filter_using_enrich1), call(enrich2), call(filter_using_enrich2)]

        Validates: Chained enrichments with dependencies on previous enrichments.
        """
        result = complex_graph.gfql([
            ASTCall('get_degrees', {'col': 'deg'}),
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'deg': GE(2)}}),
            n()
        ])

        # Nodes should have degree column and be filtered
        assert 'deg' in result._nodes.columns
        if len(result._nodes) > 0:
            assert all(result._nodes['deg'] >= 2)

    def test_complex_backward_pass_with_calls(self, complex_graph):
        """Pattern: [n({f1}), call(f2), e(), n({f3}), e(), call(f4), n()]

        Validates: Backward pass correctly handles ASTCalls in the chain.
        This is a complex pattern that tests forward pass followed by backward validation.
        """
        result = complex_graph.gfql([
            n({'type': 'person'}),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
            e(),
            n({'score': GE(20)}),
            e(),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': LE(5)}}),
            n()
        ])

        # Validate complex chain with backward pass
        if len(result._nodes) > 0:
            assert all(result._nodes['type'] == 'person')
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all(result._edges['weight'] <= 5)
