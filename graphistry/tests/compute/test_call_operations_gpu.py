"""GPU tests for GFQL Call operations."""

import os
import pytest
import pandas as pd

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTCall, ASTLet, n, e_forward
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.gfql.call_executor import execute_call
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
        """Test that Call operations work when starting with cudf DataFrames."""
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
        
        # Create graph with cudf data (may convert internally)
        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')
        
        # Execute Call operation with CUDF engine hint
        result = execute_call(g, 'get_degrees', {'col': 'degree'}, Engine.CUDF)
        
        # Result should have degree columns
        assert 'degree' in result._nodes.columns
        assert 'degree_in' in result._nodes.columns
        assert 'degree_out' in result._nodes.columns
        
        # Verify the computation is correct
        assert len(result._nodes) == 4
        # Node 2 has the highest degree (3 connections)
        # Use cuDF-compatible conversion
        import cudf
        if isinstance(result._nodes, cudf.DataFrame):
            degrees = result._nodes['degree'].to_arrow().to_pylist()
        else:
            degrees = result._nodes['degree'].tolist()
        assert max(degrees) == 3
    
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
        
        # Add edges to make it a valid graph
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3]
        })
        edges_gdf = cudf.from_pandas(edges_df)
        
        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')
        
        # Filter nodes
        result = execute_call(
            g,
            'filter_nodes_by_dict',
            {'filter_dict': {'type': 'user'}},
            Engine.CUDF
        )
        
        # Should be filtered
        assert len(result._nodes) == 3
        # Check the type column values
        types = result._nodes['type'].to_pandas() if hasattr(result._nodes, 'to_pandas') else result._nodes['type']
        assert all(types == 'user')
    
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
        # Verify scores are computed (all nodes should have scores)
        assert len(result._nodes) == 4  # 4 unique nodes
        # Use cuDF-compatible conversion
        import cudf
        if isinstance(result._nodes, cudf.DataFrame):
            scores = result._nodes['pr_score'].to_arrow().to_pylist()
        else:
            scores = result._nodes['pr_score'].tolist()
        assert all(score > 0 for score in scores)
    
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
        # Verify all nodes have coordinates
        assert len(result._nodes) == 3
        assert result._nodes['x'].notna().all()
        assert result._nodes['y'].notna().all()

    @skip_gpu
    def test_ring_continuous_layout_gpu(self):
        """Test ring_continuous_layout with cudf."""
        import cudf

        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 0]
        })
        nodes_df = pd.DataFrame({
            'node': [0, 1, 2],
            'score': [0.25, 0.5, 0.75]
        })

        edges_gdf = cudf.from_pandas(edges_df)
        nodes_gdf = cudf.from_pandas(nodes_df)

        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')

        result = execute_call(
            g,
            'ring_continuous_layout',
            {'ring_col': 'score'},
            Engine.CUDF
        )

        assert {'x', 'y', 'r'} <= set(result._nodes.columns)
        assert len(result._nodes) == 3
    
    @skip_gpu
    def test_chain_let_with_gpu_calls(self):
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
        
        result = chain_let_impl(g, dag, Engine.CUDF)
        
        # Should have degrees column
        assert 'degree' in result._nodes.columns
        # Check that we have the expected number of nodes
        # The DAG filters for 'user' type first (3 users) then computes degrees
        assert len(result._nodes) == 3  # 3 users after filtering
    
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
        # Note: filter_nodes_by_dict doesn't validate column existence at schema time
        # It will fail at runtime when the column is accessed
        # So we skip this negative test case for now
        # TODO: Enhance schema validation to check filter column existence
    
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
        
        # Add edges to make a valid graph
        edges_df = pd.DataFrame({
            'source': [0, 1, 2],
            'target': [1, 2, 3]
        })
        edges_gdf = cudf.from_pandas(edges_df)
        
        g = CGFull()\
            .edges(edges_gdf)\
            .nodes(nodes_gdf)\
            .bind(source='source', destination='target', node='node')
        
        # Test encode_point_color
        result = execute_call(
            g,
            'encode_point_color',
            {'column': 'category'},
            Engine.CUDF
        )
        
        # Should have color encoding set
        assert result._point_color == 'category'
        
        # Test encode_point_size
        result2 = execute_call(
            result,
            'encode_point_size',
            {'column': 'score'},
            Engine.CUDF
        )
        
        # Should have size encoding set
        assert result2._point_size == 'score'

    @skip_gpu
    def test_name_conflicts_any_policy_gpu(self):
        import cudf

        nodes_df = pd.DataFrame({
            'id': [0, 1, 2],
            'region': ['NA', 'EU', 'APAC']
        })
        edges_df = pd.DataFrame({
            'source': [0, 1],
            'target': [1, 2]
        })

        g = CGFull().nodes(cudf.from_pandas(nodes_df), 'id').edges(
            cudf.from_pandas(edges_df), 'source', 'target'
        )

        chain_ops = [
            n({'region': 'NA'}, name='dup'),
            e_forward(),
            n({'region': 'EU'}, name='dup'),
        ]

        result = g.gfql(chain_ops, engine='cudf')
        nodes = result._nodes.to_pandas()
        assert 'dup' in nodes.columns
        assert nodes['dup'].dtype == bool
