"""
Tests for call() operations at chain boundaries (issue #792).

These tests verify that call() operations are allowed at the start/end
of chains as a convenience feature, while still rejecting interior mixing.
"""
import unittest
import pandas as pd
import pytest

from graphistry import PyGraphistry
from graphistry.compute.ast import n, e, call
from graphistry.compute.exceptions import GFQLValidationError, ErrorCode


class TestCallBoundaries(unittest.TestCase):
    """Test call() at chain boundaries."""

    def setUp(self):
        """Create test graph."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C', 'D'],
            'dst': ['B', 'C', 'D', 'A'],
            'type': ['forward', 'forward', 'backward', 'forward'],
            'weight': [1, 2, 3, 4]
        })
        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C', 'D'],
            'label': ['Node A', 'Node B', 'Node C', 'Node D']
        })
        self.g = PyGraphistry.edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_call_prefix_then_traversal(self):
        """
        Pattern: [call(...), n(), e(), n()]
        Should work - call() at start is boundary.
        """
        # After fix, this should succeed
        result = self.g.chain([
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Prefix
            n(),
            e(),
            n()
        ])

        # Verify result is valid
        assert result is not None
        # Filter should have removed backward edges
        assert len(result._edges) <= len(self.g._edges)
        # Should only have forward edges
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')

    def test_traversal_then_call_suffix(self):
        """
        Pattern: [n(), e(), n(), call(...)]
        Should work - call() at end is boundary.
        """
        result = self.g.chain([
            n(),
            e(),
            n(),
            call('get_degrees')  # Suffix
        ])

        # Verify result is valid and has degree columns
        assert result is not None
        assert 'degree' in result._nodes.columns or 'in_degree' in result._nodes.columns

    def test_call_prefix_traversal_call_suffix(self):
        """
        Pattern: [call(...), n(), e(), n(), call(...)]
        Should work - call() at both ends is boundary.
        """
        result = self.g.chain([
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Prefix
            n(),
            e(),
            n(),
            call('get_degrees')  # Suffix
        ])

        # Verify result is valid, filtered, and enriched
        assert result is not None
        # Should have degree columns from suffix
        assert 'degree' in result._nodes.columns or 'in_degree' in result._nodes.columns
        # Should have filtered edges from prefix
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')

    def test_multiple_prefix_calls(self):
        """
        Pattern: [call(...), call(...), n(), e(), n()]
        Should work - multiple calls at start are all prefix.
        """
        result = self.g.chain([
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),  # Prefix 1
            call('filter_edges_by_dict', {'filter_dict': {'weight': [1, 2]}}),  # Prefix 2
            n(),
            e(),
            n()
        ])

        # Verify result is valid and both filters applied
        assert result is not None
        # Both filters should be applied: type=forward AND weight in [1,2]
        if len(result._edges) > 0:
            assert all(result._edges['type'] == 'forward')
            assert all(result._edges['weight'].isin([1, 2]))

    def test_multiple_suffix_calls(self):
        """
        Pattern: [n(), e(), n(), call(...), call(...)]
        Should work - multiple calls at end are all suffix.
        """
        result = self.g.chain([
            n(),
            e(),
            n(),
            call('get_degrees'),  # Suffix 1
            # Note: Second filter needs columns that exist after get_degrees
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})  # Suffix 2
        ])

        # Verify result is valid
        assert result is not None
        # Should have degree columns
        assert 'degree' in result._nodes.columns or 'in_degree' in result._nodes.columns

    def test_interior_call_still_rejected(self):
        """
        Pattern: [n(), call(...), e()]
        Should ALWAYS fail - call() in middle has undefined semantics.

        This test should pass both before and after the fix,
        because interior mixing is always disallowed.
        """
        with pytest.raises(GFQLValidationError) as exc_info:
            self.g.chain([
                n(),
                call('get_degrees'),  # Interior - NOT ALLOWED
                e(),
                n()
            ])

        assert exc_info.value.code == ErrorCode.E201
        # This error message should be updated to clarify it's interior mixing
        assert "call()" in str(exc_info.value).lower()


class TestPureChains(unittest.TestCase):
    """Verify that pure chains (all call or all traversal) still work."""

    def setUp(self):
        """Create test graph."""
        edges_df = pd.DataFrame({
            'src': ['A', 'B', 'C'],
            'dst': ['B', 'C', 'A'],
            'type': ['forward', 'forward', 'backward']
        })
        nodes_df = pd.DataFrame({
            'id': ['A', 'B', 'C'],
            'label': ['Node A', 'Node B', 'Node C']
        })
        self.g = PyGraphistry.edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')

    def test_pure_call_chain(self):
        """Pure call() chain should still work."""
        # This should work both before and after the fix
        result = self.g.chain([
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}})
        ])
        assert result is not None
        assert len(result._edges) == 2  # Only 'forward' edges

    def test_pure_traversal_chain(self):
        """Pure traversal chain should still work."""
        # This should work both before and after the fix
        result = self.g.chain([
            n(),
            e(),
            n()
        ])
        assert result is not None

    def test_multiple_call_chain(self):
        """Multiple call() operations should still work."""
        # This should work both before and after the fix
        result = self.g.chain([
            call('filter_edges_by_dict', {'filter_dict': {'type': 'forward'}}),
            call('get_degrees')
        ])
        assert result is not None
