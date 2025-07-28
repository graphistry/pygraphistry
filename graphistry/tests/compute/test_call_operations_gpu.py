"""GPU tests for GFQL Call operations."""

import os
import pytest
import pandas as pd

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTCall, ASTLet, n
from graphistry.compute.chain_dag import chain_dag_impl
from graphistry.compute.call_executor import execute_call
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


# Skip all tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestCallOperationsGPU:
    """Test Call operations with GPU/cudf."""
    
    @skip_gpu
    def test_call_with_cudf_dataframes(self):
        """Test that Call operations work with cudf DataFrames."""
        import cudf
        
        # Create cudf dataframes
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2],
            'target': [1, 2, 0, 3],
            'weight': [1.0, 2.0, 3.0, 4.0]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        
        edges_gdf = cudf.from_pandas(edges_df)
        nodes_gdf = cudf.from_pandas(nodes_df)
        
        # Create graph with cudf data
        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')
        
        # Execute Call operation
        result = execute_call(g, 'get_degrees', {'col': 'degree'}, Engine.CUDF)
        
        # Result should still have cudf nodes
        assert hasattr(result._nodes, '__cuda_array_interface__')
        assert 'degree' in result._nodes.columns
    
    @skip_gpu
    def test_filter_with_cudf(self):
        """Test filtering operations with cudf."""
        import cudf
        
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user'],
            'score': [0.5, 0.8, 0.9, 0.3]
        })
        nodes_gdf = cudf.from_pandas(nodes_df)
        
        g = CGFull().nodes(nodes_gdf).bind(node='node')
        
        # Filter nodes
        result = execute_call(
            g,
            'filter_nodes_by_dict',
            {'filter_dict': {'type': 'user'}},
            Engine.CUDF
        )
        
        # Should still be cudf and filtered
        assert hasattr(result._nodes, '__cuda_array_interface__')
        assert len(result._nodes) == 3
        assert all(result._nodes['type'] == 'user')
    
    @skip_gpu
    def test_compute_cugraph_call(self):
        """Test compute_cugraph through Call operation."""
        import cudf
        
        # Create a simple graph
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2, 3],
            'target': [1, 2, 0, 3, 0]
        })
        edges_gdf = cudf.from_pandas(edges_df)
        
        g = CGFull().edges(edges_gdf).bind(source='source', destination='target')
        
        # Skip if cugraph not available
        try:
            import cugraph
        except ImportError:
            pytest.skip("cugraph not installed")
        
        # Call compute_cugraph for pagerank
        result = execute_call(
            g,
            'compute_cugraph',
            {'alg': 'pagerank', 'out_col': 'pr_score'},
            Engine.CUDF
        )
        
        # Should have pagerank scores
        assert 'pr_score' in result._nodes.columns
        assert hasattr(result._nodes, '__cuda_array_interface__')
    
    @skip_gpu
    def test_layout_cugraph_call(self):
        """Test layout_cugraph through Call operation."""
        import cudf
        
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        edges_gdf = cudf.from_pandas(edges_df)
        
        g = CGFull().edges(edges_gdf).bind(source='source', destination='target')
        
        # Skip if cugraph not available
        try:
            import cugraph
        except ImportError:
            pytest.skip("cugraph not installed")
        
        # Call layout_cugraph
        result = execute_call(
            g,
            'layout_cugraph',
            {'layout': 'force_atlas2'},
            Engine.CUDF
        )
        
        # Should have x,y coordinates
        assert 'x' in result._nodes.columns
        assert 'y' in result._nodes.columns
        assert hasattr(result._nodes, '__cuda_array_interface__')
    
    @skip_gpu
    def test_chain_dag_with_gpu_calls(self):
        """Test DAG execution with Call operations on GPU."""
        import cudf
        
        edges_df = pd.DataFrame({
            'source': [0, 1, 2, 2, 3],
            'target': [1, 2, 0, 3, 0],
            'weight': [1.0, 2.0, 1.5, 3.0, 0.5]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'type': ['user', 'user', 'admin', 'user']
        })
        
        edges_gdf = cudf.from_pandas(edges_df)
        nodes_gdf = cudf.from_pandas(nodes_df)
        
        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')
        
        # Create DAG with Call operations
        dag = ASTLet({
            'filtered': n({'type': 'user'}),
            'with_degrees': ASTCall('get_degrees', {'col': 'degree'})
        })
        
        result = chain_dag_impl(g, dag, Engine.CUDF)
        
        # Should have GPU data with degrees
        assert hasattr(result._nodes, '__cuda_array_interface__')
        assert 'degree' in result._nodes.columns
    
    @skip_gpu
    def test_schema_validation_with_cudf(self):
        """Test schema validation works with cudf DataFrames."""
        import cudf
        
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'type': ['A', 'B', 'C']
        })
        nodes_gdf = cudf.from_pandas(nodes_df)
        
        g = CGFull().nodes(nodes_gdf).bind(node='node')
        
        # Valid call - column exists
        call = ASTCall('filter_nodes_by_dict', {'filter_dict': {'type': 'A'}})
        errors = validate_chain_schema(g, [call], collect_all=True)
        assert len(errors) == 0
        
        # Invalid call - column doesn't exist
        call = ASTCall('filter_nodes_by_dict', {'filter_dict': {'missing': 'X'}})
        errors = validate_chain_schema(g, [call], collect_all=True)
        assert len(errors) > 0
        assert any('missing' in str(e) for e in errors)
    
    @skip_gpu
    def test_encode_with_gpu(self):
        """Test visual encoding methods with GPU data."""
        import cudf
        
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2, 3],
            'category': ['A', 'B', 'A', 'C'],
            'score': [0.1, 0.5, 0.8, 0.3]
        })
        nodes_gdf = cudf.from_pandas(nodes_df)
        
        g = CGFull().nodes(nodes_gdf).bind(node='node')
        
        # Test encode_point_color
        result = execute_call(
            g,
            'encode_point_color',
            {'column': 'category'},
            Engine.CUDF
        )
        
        # Should still have GPU data
        assert hasattr(result._nodes, '__cuda_array_interface__')
        
        # Test encode_point_size
        result2 = execute_call(
            result,
            'encode_point_size',
            {'column': 'score'},
            Engine.CUDF
        )
        
        assert hasattr(result2._nodes, '__cuda_array_interface__')
        # Should have size encoding set
        assert result2._point_size == 'score'
