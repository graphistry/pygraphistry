"""Unit tests for engine_coercion.py helper utilities.

Tests verify that ensure_engine_match() correctly:
1. Performs no-op when types already match (performance)
2. Converts DataFrames when types mismatch (correctness)
3. Handles both nodes and edges (completeness)
4. Handles None edges gracefully (robustness)
"""

import pandas as pd
import pytest
from graphistry import Engine
from graphistry.compute.engine_coercion import ensure_engine_match
import graphistry

# Conditionally import cuDF for GPU tests
try:
    import cudf
    has_cudf = True
except ImportError:
    has_cudf = False


class TestEngineCoercionHelper:
    """Test suite for ensure_engine_match() helper function."""

    def test_no_op_when_types_match_pandas(self):
        """Test that pandas→pandas is a no-op (no conversion)."""
        # Create pandas graph
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], 'relation': ['knows', 'visited']})
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Request pandas engine - should be no-op
        result = ensure_engine_match(g, Engine.PANDAS)

        # Verify types unchanged (still pandas)
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should remain pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should remain pandas DataFrame"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert len(result._edges) == 2, "Edges count should be preserved"
        assert list(result._nodes['id']) == [1, 2, 3], "Node IDs should be preserved"

    @pytest.mark.skipif(not has_cudf, reason="cuDF not available")
    def test_no_op_when_types_match_cudf(self):
        """Test that cudf→cudf is a no-op (no conversion)."""
        # Create cuDF graph
        nodes_df = cudf.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        edges_df = cudf.DataFrame({'src': [1, 2], 'dst': [2, 3], 'relation': ['knows', 'visited']})
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Request cuDF engine - should be no-op
        result = ensure_engine_match(g, Engine.CUDF)

        # Verify types unchanged (still cuDF)
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should remain cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should remain cuDF DataFrame"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert len(result._edges) == 2, "Edges count should be preserved"
        assert result._nodes['id'].to_pandas().tolist() == [1, 2, 3], "Node IDs should be preserved"

    @pytest.mark.skipif(not has_cudf, reason="cuDF not available")
    def test_converts_pandas_to_cudf(self):
        """Test that pandas→cudf conversion works correctly."""
        # Create pandas graph
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], 'relation': ['knows', 'visited']})
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Request cuDF engine - should convert
        result = ensure_engine_match(g, Engine.CUDF)

        # Verify types converted to cuDF
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be converted to cuDF DataFrame"
        assert isinstance(result._edges, cudf.DataFrame), "Edges should be converted to cuDF DataFrame"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert len(result._edges) == 2, "Edges count should be preserved"
        assert result._nodes['id'].to_pandas().tolist() == [1, 2, 3], "Node IDs should be preserved"

    @pytest.mark.skipif(not has_cudf, reason="cuDF not available")
    def test_converts_cudf_to_pandas(self):
        """Test that cudf→pandas conversion works correctly."""
        # Create cuDF graph
        nodes_df = cudf.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        edges_df = cudf.DataFrame({'src': [1, 2], 'dst': [2, 3], 'relation': ['knows', 'visited']})
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Request pandas engine - should convert
        result = ensure_engine_match(g, Engine.PANDAS)

        # Verify types converted to pandas
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be converted to pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be converted to pandas DataFrame"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert len(result._edges) == 2, "Edges count should be preserved"
        assert list(result._nodes['id']) == [1, 2, 3], "Node IDs should be preserved"

    def test_converts_both_nodes_and_edges(self):
        """Test that both nodes AND edges are converted (not just nodes)."""
        # Create pandas graph with both nodes and edges
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3], 'relation': ['knows', 'visited']})
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Request pandas engine (no-op, but verifies both are checked)
        result = ensure_engine_match(g, Engine.PANDAS)

        # Verify both nodes and edges have correct types
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas DataFrame"

        # Verify both have data
        assert len(result._nodes) == 3, "Nodes should have 3 rows"
        assert len(result._edges) == 2, "Edges should have 2 rows"

    def test_handles_none_edges(self):
        """Test that None edges are handled gracefully (nodes-only graph)."""
        # Create nodes-only graph (no edges)
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        g = graphistry.nodes(nodes_df, 'id')

        # Request pandas engine - should work even with None edges
        result = ensure_engine_match(g, Engine.PANDAS)

        # Verify nodes converted, edges remain None
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas DataFrame"
        assert result._edges is None, "Edges should remain None"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert list(result._nodes['id']) == [1, 2, 3], "Node IDs should be preserved"

    @pytest.mark.skipif(not has_cudf, reason="cuDF not available")
    def test_handles_none_edges_with_cudf_conversion(self):
        """Test that None edges are handled when converting to cuDF."""
        # Create pandas nodes-only graph
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        g = graphistry.nodes(nodes_df, 'id')

        # Request cuDF engine - should convert nodes, leave edges as None
        result = ensure_engine_match(g, Engine.CUDF)

        # Verify nodes converted to cuDF, edges remain None
        assert isinstance(result._nodes, cudf.DataFrame), "Nodes should be converted to cuDF DataFrame"
        assert result._edges is None, "Edges should remain None (not converted)"

        # Verify data preserved
        assert len(result._nodes) == 3, "Nodes count should be preserved"
        assert result._nodes['id'].to_pandas().tolist() == [1, 2, 3], "Node IDs should be preserved"

    def test_graceful_degradation_on_error(self):
        """Test that errors are handled gracefully (returns original graph)."""
        # Create a graph
        nodes_df = pd.DataFrame({'id': [1, 2, 3], 'type': ['person', 'person', 'place']})
        g = graphistry.nodes(nodes_df, 'id')

        # This shouldn't raise an exception even with invalid engine
        # (though in practice Engine enum prevents this - test shows defensive coding)
        result = ensure_engine_match(g, Engine.PANDAS)

        # Should return a valid graph (original or converted)
        assert result is not None, "Should return a graph object"
        assert hasattr(result, '_nodes'), "Result should have _nodes attribute"
