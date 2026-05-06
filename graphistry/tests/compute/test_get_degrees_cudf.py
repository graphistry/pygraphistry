"""
Tests for get_degrees(), get_indegrees(), get_outdegrees() with cuDF/pandas compatibility.

This test module ensures proper DataFrame engine handling in degree calculations,
particularly after schema-changing operations like UMAP and hypergraph.

Related issues: #778
"""

import os
import pytest
import pandas as pd
import graphistry

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestGetDegrees:
    """Test get_degrees() and related functions with different DataFrame engines"""

    def test_get_degrees_pandas(self):
        """Test get_degrees with pandas DataFrames (baseline CPU test)"""
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_degrees()

        # Should complete without error
        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert 'degree' in result._nodes.columns
        assert 'degree_in' in result._nodes.columns
        assert 'degree_out' in result._nodes.columns

    @skip_gpu
    def test_get_degrees_cudf(self):
        """Test get_degrees with cuDF DataFrames"""
        import cudf

        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_degrees()

        # Should complete without error
        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'degree' in result._nodes.columns
        assert 'degree_in' in result._nodes.columns
        assert 'degree_out' in result._nodes.columns

    def test_get_indegrees_pandas(self):
        """Test get_indegrees with pandas DataFrames"""
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_indegrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert 'degree_in' in result._nodes.columns

    @skip_gpu
    def test_get_indegrees_cudf(self):
        """Test get_indegrees with cuDF DataFrames"""
        import cudf

        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_indegrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'degree_in' in result._nodes.columns

    def test_get_outdegrees_pandas(self):
        """Test get_outdegrees with pandas DataFrames"""
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_outdegrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert 'degree_out' in result._nodes.columns

    @skip_gpu
    def test_get_outdegrees_cudf(self):
        """Test get_outdegrees with cuDF DataFrames"""
        import cudf

        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_outdegrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'degree_out' in result._nodes.columns

    def test_get_degrees_with_existing_nodes_pandas(self):
        """Test get_degrees when nodes already exist (pandas)"""
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person']
        })
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        result = g.get_degrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert 'degree' in result._nodes.columns
        assert 'type' in result._nodes.columns  # Original column preserved

    @skip_gpu
    def test_get_degrees_with_existing_nodes_cudf(self):
        """Test get_degrees when nodes already exist (cuDF)"""
        import cudf

        nodes_df = cudf.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'org', 'org', 'person']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')
        result = g.get_degrees()

        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'degree' in result._nodes.columns
        assert 'type' in result._nodes.columns  # Original column preserved

    @skip_gpu
    def test_get_degrees_mixed_engines_nodes_pandas_edges_cudf(self):
        """Test get_degrees with mixed engines: pandas nodes, cuDF edges"""
        import cudf

        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e']
        })
        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.nodes(nodes_df, 'id').edges(edges_df, 'src', 'dst')

        # This should work - edges drive the engine, will be converted internally
        result = g.get_degrees()

        assert result._nodes is not None
        assert 'degree' in result._nodes.columns

    @skip_gpu
    def test_get_topological_levels_cudf(self):
        """Test get_topological_levels with cuDF DataFrames (uses get_degrees internally)"""
        import cudf

        edges_df = cudf.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_topological_levels()

        assert result._nodes is not None
        assert isinstance(result._nodes, cudf.DataFrame)
        assert 'level' in result._nodes.columns

    def test_get_topological_levels_pandas(self):
        """Test get_topological_levels with pandas DataFrames"""
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'e']
        })

        g = graphistry.edges(edges_df, 'src', 'dst')
        result = g.get_topological_levels()

        assert result._nodes is not None
        assert isinstance(result._nodes, pd.DataFrame)
        assert 'level' in result._nodes.columns
