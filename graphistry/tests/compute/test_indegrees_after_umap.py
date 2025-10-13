"""Test that get_indegrees() works after UMAP in GFQL with engine coercion."""
import pandas as pd
import pytest
from graphistry.compute.ast import ASTLet, ASTRef, n, call
from graphistry.tests.test_compute import CGFull
from graphistry.utils.lazy_import import lazy_umap_import

# Check if UMAP is available
has_umap, _, _ = lazy_umap_import()


class TestGetIndegreesAfterUMAP:
    """Test suite for verifying get_indegrees() works after UMAP with engine coercion."""

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_get_indegrees_after_umap_in_let(self):
        """Test get_indegrees() after UMAP via let() with pandas engine request.
        
        This verifies the fix for the bug where:
        1. UMAP in GPU mode creates cuDF edges
        2. Nodes remain pandas
        3. get_indegrees() tries to merge cuDF with pandas → fails
        
        Our engine coercion ensures both nodes AND edges are pandas,
        preventing the merge error in get_indegrees().
        """
        # Create test graph
        nodes_df = pd.DataFrame({
            'id': range(15),
            'x': [i * 0.1 for i in range(15)],
            'y': [i * 0.2 for i in range(15)],
            'category': [f'cat_{i % 3}' for i in range(15)]
        })
        edges_df = pd.DataFrame({
            'src': [i for i in range(14)],
            'dst': [i + 1 for i in range(14)],
            'weight': [1.0] * 14
        })
        
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        
        # Run UMAP in let with pandas engine request
        dag = ASTLet({
            'start': n({}),
            'umap': ASTRef('start', [call('umap', {
                'n_components': 2, 
                'n_neighbors': 3, 
                'umap_kwargs': {'random_state': 42, 'n_epochs': 3}, 
                'engine': 'auto'  # On CPU picks umap-learn → pandas
            })])
        })
        
        # Execute with engine='pandas' - should coerce everything to pandas
        result = g.gfql(dag, engine='pandas', output='umap')
        
        # Verify both nodes and edges are pandas
        assert isinstance(result._nodes, pd.DataFrame), "Nodes should be pandas after coercion"
        assert isinstance(result._edges, pd.DataFrame), "Edges should be pandas after coercion"
        
        # Now call get_indegrees() - should work without merge error
        # This is the bug we're testing: mixed DataFrame types cause merge to fail
        result_with_degrees = result.get_indegrees()
        
        # Verify it worked
        assert 'degree_in' in result_with_degrees._nodes.columns, "Should have degree_in column"
        assert isinstance(result_with_degrees._nodes, pd.DataFrame), "Result should still be pandas"
        assert len(result_with_degrees._nodes) == 15, "Should have all 15 nodes"

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_get_outdegrees_after_umap_in_let(self):
        """Test get_outdegrees() also works after UMAP with engine coercion."""
        nodes_df = pd.DataFrame({
            'id': range(15),
            'x': [i * 0.1 for i in range(15)],
            'y': [i * 0.2 for i in range(15)]
        })
        edges_df = pd.DataFrame({
            'src': [i for i in range(14)],
            'dst': [i + 1 for i in range(14)]
        })
        
        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        
        dag = ASTLet({
            'umap': call('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42, 'n_epochs': 3},
                'engine': 'auto'
            })
        })
        
        result = g.gfql(dag, engine='pandas')
        
        # Both nodes and edges should be pandas
        assert isinstance(result._nodes, pd.DataFrame)
        assert isinstance(result._edges, pd.DataFrame)
        
        # get_outdegrees should work without type mismatch
        result_with_degrees = result.get_outdegrees()
        
        assert 'degree_out' in result_with_degrees._nodes.columns
        assert isinstance(result_with_degrees._nodes, pd.DataFrame)
