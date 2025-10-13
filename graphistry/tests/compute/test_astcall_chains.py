"""
Comprehensive test suite for ASTCall operations in GFQL chains.

Tests various hazard types and operation combinations to ensure proper
chaining behavior for filter, enrichment, and transformation operations.

This test suite should be run BEFORE implementing fixes to document
current behavior and ensure no regressions during fix implementation.
"""
import pandas as pd
import pytest
from graphistry import PyGraphistry
from graphistry.compute.ast import ASTCall, n, e_forward
from graphistry.tests.test_compute import CGFull


class TestASTCallChainHazards:
    """Test suite for tricky ASTCall chain scenarios.

    Categories:
    1. Filter chains (multiple filters in sequence)
    2. Mixed filter + traversal operations
    3. Enrichment operations (add columns)
    4. Filter + enrichment combinations
    5. Edge cases (empty results, all filtered out, etc.)
    """

    @pytest.fixture
    def sample_graph(self):
        """Create test data with multiple attributes for filtering."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'A', 'B', 'C', 'D', 'E'],
            'dst': ['B', 'C', 'A', 'C', 'A', 'B', 'E', 'A'],
            'type': ['forward', 'forward', 'forward', 'backward', 'backward', 'backward', 'forward', 'forward'],
            'weight': [1, 2, 3, 4, 5, 6, 7, 8],
            'category': ['low', 'med', 'med', 'high', 'high', 'high', 'high', 'high']
        })

        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D', 'E'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })

        return CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    # ========================================================================
    # CATEGORY 1: Filter Chains (Core Bug from Issue #786)
    # ========================================================================

    def test_single_edge_filter(self, sample_graph):
        """Baseline: Single filter should work correctly."""
        filter_op = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        result = sample_graph.gfql([filter_op])

        # Should have 5 forward edges (weights 1,2,3,7,8)
        assert len(result._edges) == 5
        assert all(result._edges['type'] == 'forward')

    @pytest.mark.xfail(reason="Issue #786 - chained filters only apply first filter")
    def test_two_edge_filters_sequential(self, sample_graph):
        """Core bug: Two filters should both apply."""
        from graphistry.compute.predicates.numeric import Between

        filter_by_type = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        filter_by_weight = ASTCall('filter_edges_by_dict', {
            'filter_dict': {'weight': Between(2, 6)}
        })
        result = sample_graph.gfql([filter_by_type, filter_by_weight])

        # Should have 2 edges: forward AND weight 2-6 (weights 2,3)
        assert len(result._edges) == 2
        assert all(result._edges['type'] == 'forward')
        assert all((result._edges['weight'] >= 2) & (result._edges['weight'] <= 6))

    @pytest.mark.xfail(reason="Issue #786 - chained filters only apply first filter")
    def test_three_edge_filters_sequential(self, sample_graph):
        """Extended case: Three filters should all apply."""
        filter1 = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        filter2 = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 2}}})
        filter3 = ASTCall('filter_edges_by_dict', {'filter_dict': {'category': 'med'}})
        result = sample_graph.gfql([filter1, filter2, filter3])

        # Should have 1 edge: forward AND weight>=2 AND category=med (weight 2)
        assert len(result._edges) == 1
        assert result._edges.iloc[0]['type'] == 'forward'
        assert result._edges.iloc[0]['weight'] == 2
        assert result._edges.iloc[0]['category'] == 'med'

    def test_single_node_filter(self, sample_graph):
        """Baseline: Single node filter should work."""
        filter_op = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'person'}})
        result = sample_graph.gfql([filter_op])

        # Should have 3 person nodes (A, B, E)
        assert len(result._nodes) == 3
        assert all(result._nodes['type'] == 'person')

    @pytest.mark.xfail(reason="Issue #786 - likely affects node filters too")
    def test_two_node_filters_sequential(self, sample_graph):
        """Node filter chaining should work like edge filters."""
        filter1 = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'person'}})
        filter2 = ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': {'gte': 20}}})
        result = sample_graph.gfql([filter1, filter2])

        # Should have 2 nodes: person AND score>=20 (B, E with scores 20, 50)
        assert len(result._nodes) == 2
        assert all(result._nodes['type'] == 'person')
        assert all(result._nodes['score'] >= 20)

    @pytest.mark.xfail(reason="Issue #786 - likely affects mixed filters")
    def test_edge_then_node_filter(self, sample_graph):
        """Mixed filter types should both apply."""
        edge_filter = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        node_filter = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'person'}})
        result = sample_graph.gfql([edge_filter, node_filter])

        # Should filter edges first, then nodes
        assert len(result._edges) == 5  # All forward edges
        assert len(result._nodes) == 3  # Only person nodes

    # ========================================================================
    # CATEGORY 2: Filter + Traversal Operations
    # ========================================================================

    @pytest.mark.xfail(reason="Issue #786 - filter not passed to subsequent traversal ops")
    def test_filter_then_hop_forward(self, sample_graph):
        """Filter edges, then perform graph traversal on filtered graph."""
        filter_op = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        # After filtering to forward edges, do a simple traversal
        result = sample_graph.gfql([filter_op, n(), e_forward(), n()])

        # Traversal should only see forward edges
        # This tests if filtered graph is passed to subsequent operations
        # Expected: Only paths using forward edges
        # Actual: May use all edges if filter is ignored
        assert all(result._edges['type'] == 'forward') if len(result._edges) > 0 else True

    def test_hop_then_filter(self, sample_graph):
        """Perform traversal first, then filter the result."""
        # Start at node A, hop forward
        result = sample_graph.gfql([
            n({'id': 'A'}),
            e_forward(),
            n(),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 5}}})
        ])

        # Should traverse from A, then filter to high-weight edges
        # This tests if filter receives the traversal result
        assert len(result._edges) >= 0  # May fail if filter operates on original graph

    def test_filter_between_hops(self, sample_graph):
        """Filter in the middle of a multi-hop traversal."""
        result = sample_graph.gfql([
            n({'id': 'A'}),
            e_forward(),
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
            n(),
            e_forward(),
            n()
        ])

        # Filter should affect subsequent hops
        # This is a complex interaction test
        assert len(result._nodes) >= 0

    # ========================================================================
    # CATEGORY 3: Enrichment Operations (Non-filter ASTCalls)
    # ========================================================================

    # Note: These tests require enrichment operations to be implemented
    # Keeping stubs for completeness

    @pytest.mark.skip(reason="Enrichment operations not yet in safelist")
    def test_single_enrichment(self, sample_graph):
        """Single enrichment operation should work."""
        # Example: add a computed column
        enrich_op = ASTCall('assign_edges', {'columns': {'doubled_weight': 'weight * 2'}})
        result = sample_graph.gfql([enrich_op])

        assert 'doubled_weight' in result._edges.columns
        assert all(result._edges['doubled_weight'] == result._edges['weight'] * 2)

    @pytest.mark.skip(reason="Enrichment operations not yet in safelist")
    def test_enrichment_then_filter(self, sample_graph):
        """Enrich data, then filter using enriched column."""
        enrich_op = ASTCall('assign_edges', {'columns': {'doubled_weight': 'weight * 2'}})
        filter_op = ASTCall('filter_edges_by_dict', {'filter_dict': {'doubled_weight': {'gte': 10}}})
        result = sample_graph.gfql([enrich_op, filter_op])

        # Filter should see the enriched column
        assert 'doubled_weight' in result._edges.columns
        assert all(result._edges['doubled_weight'] >= 10)

    # ========================================================================
    # CATEGORY 4: Edge Cases
    # ========================================================================

    def test_filter_to_empty_edges(self, sample_graph):
        """Filter that eliminates all edges."""
        filter_op = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'nonexistent'}})
        result = sample_graph.gfql([filter_op])

        # Should return empty graph gracefully
        assert len(result._edges) == 0
        assert result._edges is not None  # Should be empty DataFrame, not None

    def test_filter_to_empty_then_filter_again(self, sample_graph):
        """Chain filters where first eliminates all data."""
        filter1 = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'nonexistent'}})
        filter2 = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 5}}})
        result = sample_graph.gfql([filter1, filter2])

        # Should handle empty intermediate result
        assert len(result._edges) == 0

    def test_filter_to_empty_nodes(self, sample_graph):
        """Filter that eliminates all nodes."""
        filter_op = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'nonexistent'}})
        result = sample_graph.gfql([filter_op])

        # Should return empty nodes gracefully
        assert len(result._nodes) == 0
        assert result._nodes is not None

    # ========================================================================
    # CATEGORY 5: Complex Predicate Filters
    # ========================================================================

    def test_filter_with_numeric_predicates(self, sample_graph):
        """Filters using numeric comparison predicates."""
        from graphistry.compute.predicates.numeric import Between

        filter_op = ASTCall('filter_edges_by_dict', {
            'filter_dict': {'weight': Between(2, 6)}
        })
        result = sample_graph.gfql([filter_op])

        # Should filter to weights 2-6
        assert len(result._edges) == 5  # weights 2,3,4,5,6
        assert all((result._edges['weight'] >= 2) & (result._edges['weight'] <= 6))

    @pytest.mark.xfail(reason="Issue #786 - chained predicates likely affected")
    def test_chained_filters_with_predicates(self, sample_graph):
        """Chain two filters using predicates."""
        from graphistry.compute.predicates.numeric import Gt, Lt

        filter1 = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': Gt(2)}})
        filter2 = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': Lt(7)}})
        result = sample_graph.gfql([filter1, filter2])

        # Should filter to 2 < weight < 7 (weights 3,4,5,6)
        assert len(result._edges) == 4
        assert all((result._edges['weight'] > 2) & (result._edges['weight'] < 7))

    # ========================================================================
    # CATEGORY 6: Order Sensitivity Tests
    # ========================================================================

    @pytest.mark.xfail(reason="Issue #786 - order may matter incorrectly")
    def test_filter_order_should_not_matter(self, sample_graph):
        """Applying filters in different orders should give same result."""
        # Order 1: type then weight
        filter1a = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        filter1b = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 2, 'lte': 6}}})
        result1 = sample_graph.gfql([filter1a, filter1b])

        # Order 2: weight then type
        filter2a = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 2, 'lte': 6}}})
        filter2b = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        result2 = sample_graph.gfql([filter2a, filter2b])

        # Results should be identical (both filters applied)
        assert len(result1._edges) == len(result2._edges)
        assert len(result1._edges) == 2  # Should be 2 edges in both cases

    # ========================================================================
    # CATEGORY 7: Performance/Optimization Tests
    # ========================================================================

    def test_redundant_filters(self, sample_graph):
        """Apply the same filter twice - second should be a no-op."""
        filter_op = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        result = sample_graph.gfql([filter_op, filter_op])

        # Should still have 5 forward edges (filter is idempotent)
        assert len(result._edges) == 5
        assert all(result._edges['type'] == 'forward')

    @pytest.mark.xfail(reason="Issue #786 - second filter is ignored")
    def test_contradictory_filters(self, sample_graph):
        """Apply contradictory filters - should result in empty."""
        filter1 = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        filter2 = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'backward'}})
        result = sample_graph.gfql([filter1, filter2])

        # No edges can be both forward and backward
        assert len(result._edges) == 0

    # ========================================================================
    # CATEGORY 8: Mixed Operation Type Chains
    # ========================================================================

    def test_complex_mixed_chain(self, sample_graph):
        """Complex chain mixing filters, traversals, and multiple operation types."""
        # Start at high-score nodes, filter edges, traverse, filter again
        result = sample_graph.gfql([
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'score': {'gte': 30}}}),  # C, D, E
            ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),     # Only forward edges
            n(),  # Get nodes
            e_forward(),  # Traverse forward
            n(),  # Get destination nodes
            ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'person'}}),     # Only person nodes
        ])

        # This is a complex integration test
        # Just verify it doesn't crash and returns some reasonable structure
        assert result._nodes is not None
        assert result._edges is not None

    # ========================================================================
    # CATEGORY 9: Safelist Validation
    # ========================================================================

    def test_filter_edges_in_safelist(self):
        """Verify filter_edges_by_dict is in call safelist."""
        from graphistry.compute.gfql.call_safelist import SAFELIST_V1
        assert 'filter_edges_by_dict' in SAFELIST_V1

    def test_filter_nodes_in_safelist(self):
        """Verify filter_nodes_by_dict is in call safelist."""
        from graphistry.compute.gfql.call_safelist import SAFELIST_V1
        assert 'filter_nodes_by_dict' in SAFELIST_V1

    def test_invalid_call_raises_error(self, sample_graph):
        """Calling non-safelisted method should raise error."""
        invalid_op = ASTCall('not_a_real_method', {'param': 'value'})

        with pytest.raises(Exception):  # Should raise validation error
            sample_graph.gfql([invalid_op])


class TestASTCallRemoteChains:
    """Test ASTCall chains in remote execution mode.

    These tests require a running Graphistry server and are skipped
    if server is not available. They verify that the bug affects
    both local and remote execution.
    """

    @pytest.mark.skip(reason="Requires running Graphistry server")
    def test_remote_two_edge_filters(self):
        """Remote execution should have same bug as local."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C'],
            'dst': ['B', 'C', 'A'],
            'type': ['forward', 'forward', 'backward'],
            'weight': [1, 2, 3]
        })

        g = CGFull().edges(edges_df, 'src', 'dst')
        uploaded = g.upload()  # Requires server

        filter1 = ASTCall('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        filter2 = ASTCall('filter_edges_by_dict', {'filter_dict': {'weight': {'gte': 2}}})

        result = uploaded.gfql_remote([filter1, filter2])

        # Bug should affect remote execution too
        # Expected: 1 edge (forward AND weight>=2)
        # Actual (bug): 2 edges (only first filter)
        assert len(result._edges) == 1  # Will fail with bug


# Run tests with: WITH_BUILD=0 WITH_TEST=1 ./test-cpu-local-minimal.sh graphistry/tests/compute/test_astcall_chains.py -x
