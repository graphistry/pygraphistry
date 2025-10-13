"""
Tests for chain.py combine_steps() concat operations.

This test module ensures proper DataFrame engine handling in chain operations,
particularly when chaining with schema-changing operations like UMAP.

Related issues: #777
"""

import os
import pytest
import pandas as pd
import graphistry

from graphistry.compute.ast import n, e, call
from graphistry.tests.test_compute import CGFull
from graphistry.utils.lazy_import import lazy_umap_import

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)

# Check if UMAP is available (AI dependency, not minimal)
has_umap, _, _ = lazy_umap_import()


class TestChainCombineSteps:
    """Test combine_steps() function with different DataFrame engines"""

    def test_chain_node_filter_pandas(self):
        """Test basic node filter chain with pandas (CPU regression test)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'works_for', 'partners', 'knows']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with node filters - exercises combine_steps()
        result = g.chain([
            n({}),  # Identity filter
            e(),    # Edge filter
            n({})   # Another identity filter
        ], engine='pandas')

        # Should complete without error
        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert result._edges is not None
        assert isinstance(result._edges, pd.DataFrame)

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_chain_umap_with_node_filters_pandas(self):
        """Test UMAP with node filters in pandas mode (CPU baseline)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        # Use graphistry.Plotter (not CGFull) since UMAP requires UMAPMixin
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain: Node filter → UMAP → Node filter
        # This is the pattern from issue #777
        result = g.chain([
            n({}),  # Identity node filter
            call('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})  # Another identity filter
        ], engine='pandas')

        # Should complete successfully
        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        # UMAP creates new node embeddings, may have 0 edges
        assert result._edges is not None

    @skip_gpu
    def test_chain_node_filter_cudf(self):
        """Test basic node filter chain with cuDF (GPU test)"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person'],
            'score': [10, 20, 30, 40, 50]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'works_for', 'partners', 'knows']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with cuDF
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)

        # Chain with node filters - exercises combine_steps() with cuDF
        result = g.chain([
            n({}),  # Identity filter
            e(),    # Edge filter
            n({})   # Another identity filter
        ], engine='cudf')

        # Should complete without error and preserve cuDF
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cudf.DataFrame for nodes, got {type(result._nodes)}"
        assert result._edges is not None
        assert isinstance(result._edges, cudf.DataFrame), \
            f"Expected cudf.DataFrame for edges, got {type(result._edges)}"

    @skip_gpu
    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_chain_umap_with_node_filters_cudf(self):
        """Test UMAP with node filters in cuDF mode (GPU test for issue #777)"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [2.0, 3.0, 4.0, 5.0, 6.0]
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        # Use graphistry.Plotter (not CGFull) since UMAP requires UMAPMixin
        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Verify we're starting with cuDF
        assert isinstance(g._nodes, cudf.DataFrame)
        assert isinstance(g._edges, cudf.DataFrame)

        # Chain: Node filter → UMAP → Node filter
        # This is the exact pattern from issue #777
        result = g.chain([
            n({}),  # Identity node filter
            call('umap', {
                'n_components': 2,
                'n_neighbors': 3,
                'umap_kwargs': {'random_state': 42}
            }),
            n({})  # Another identity filter
        ], engine='cudf')

        # Should complete without TypeError
        # This was failing with: "cannot concatenate object of type '<class 'cudf.core.dataframe.DataFrame'>'"
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cudf.DataFrame for nodes, got {type(result._nodes)}"
        assert result._edges is not None
        # After UMAP schema change, edges should still be cuDF

    @skip_gpu
    def test_combine_steps_with_cudf_dataframes(self):
        """Direct test of combine_steps path with cuDF DataFrames"""
        import cudf

        # Create a graph with multiple operations that will go through combine_steps
        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'category': ['X', 'Y', 'X', 'Y']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'd']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain that exercises combine_steps with multiple steps
        result = g.chain([
            n({'category': 'X'}, name='step1'),
            e(),
            n({}, name='step2'),
            e(),
            n({}, name='step3')
        ], engine='cudf')

        # combine_steps should handle cuDF concat correctly
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        # Should have boolean columns for named steps
        assert 'step1' in result._nodes.columns
        assert 'step2' in result._nodes.columns
        assert 'step3' in result._nodes.columns

    def test_combine_steps_edge_concatenation_pandas(self):
        """Test combine_steps edge concatenation with pandas"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e']
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with multiple edge operations
        result = g.chain([
            n(),
            e(name='e1'),
            n(),
            e(name='e2')
        ], engine='pandas')

        # Should concatenate edges correctly
        assert result._edges is not None
        assert isinstance(result._edges, pd.DataFrame)
        assert 'e1' in result._edges.columns
        assert 'e2' in result._edges.columns

    @skip_gpu
    def test_combine_steps_edge_concatenation_cudf(self):
        """Test combine_steps edge concatenation with cuDF"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = CGFull().nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # Chain with multiple edge operations
        result = g.chain([
            n(),
            e(name='e1'),
            n(),
            e(name='e2')
        ], engine='cudf')

        # Should concatenate edges correctly with cuDF
        assert result._edges is not None
        assert isinstance(result._edges, cudf.DataFrame)
        assert 'e1' in result._edges.columns
        assert 'e2' in result._edges.columns
