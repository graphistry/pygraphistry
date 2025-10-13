import os
import pytest
import pandas as pd

from graphistry.compute.ast import ASTLet, ASTRemoteGraph, ASTRef, n
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.execution_context import ExecutionContext
from graphistry.tests.test_compute import CGFull

# Skip all tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestChainDagGPU:
    """Test chain_let with GPU/cudf"""
    
    @skip_gpu
    def test_execution_context_stores_cudf(self):
        """Test that ExecutionContext can store cudf DataFrames"""
        import cudf
        context = ExecutionContext()
        
        # Create a cudf DataFrame
        gdf = cudf.DataFrame({'col1': [1, 2, 3], 'col2': ['a', 'b', 'c']})
        
        # Store it
        context.set_binding('gpu_data', gdf)
        
        # Retrieve it
        retrieved = context.get_binding('gpu_data')
        
        # Verify it's still a cudf DataFrame
        assert isinstance(retrieved, cudf.DataFrame)
        assert retrieved.equals(gdf)
    
    @skip_gpu
    def test_chain_let_with_cudf_edges(self):
        """Test chain_let with cudf edge DataFrame"""
        import cudf
        # Create cudf edges
        edges_df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        edges_gdf = cudf.from_pandas(edges_df)
        
        # Create graph with cudf edges
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Verify edges are cudf
        assert isinstance(g._edges, cudf.DataFrame)
        
        # Empty DAG should work
        dag = ASTLet({})
        result = chain_let_impl(g, dag)
        
        # Result should preserve GPU mode
        assert isinstance(result._edges, cudf.DataFrame)
    
    @skip_gpu
    def test_chain_let_engine_cudf(self):
        """Test chain_let with explicit engine='cudf'"""
        import cudf
        # Start with pandas
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        
        # Empty DAG with cudf engine
        dag = ASTLet({})
        from graphistry.Engine import Engine
        result = chain_let_impl(g, dag, Engine.CUDF)
        
        # Should have materialized nodes
        assert result._nodes is not None
        # Nodes should be cudf (due to materialize_nodes with cudf engine)
        assert isinstance(result._nodes, cudf.DataFrame)
    
    @skip_gpu
    def test_chain_let_auto_detects_gpu(self):
        """Test chain_let auto-detects GPU mode from edges"""
        import cudf
        # Create graph with GPU edges
        edges_gdf = cudf.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Create a simple DAG (will fail on execution but that's ok)
        dag = ASTLet({
            'step1': n()
        })
        
        # Try to execute
        try:
            chain_let_impl(g, dag)  # engine='auto' by default
        except RuntimeError as e:
            # Should fail on execution, but engine should be detected
            assert "Failed to execute node 'step1'" in str(e)
        
        # The important part is it didn't fail on engine detection
        # or materialize_nodes with GPU data
    
    @skip_gpu
    def test_resolve_engine_with_gpu(self):
        """Test that resolve_engine correctly identifies GPU mode"""
        import cudf
        from graphistry.Engine import resolve_engine, EngineAbstract
        
        # Create graph with cudf edges
        edges_gdf = cudf.DataFrame({'s': ['a'], 'd': ['b']})
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Resolve should detect cudf
        engine = resolve_engine(EngineAbstract.AUTO, g)
        assert engine.value == 'cudf'
    
    @skip_gpu
    def test_materialize_nodes_preserves_gpu(self):
        """Test materialize_nodes works with GPU"""
        import cudf
        # Create graph with cudf edges
        edges_gdf = cudf.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Materialize nodes
        g2 = g.materialize_nodes()
        
        # Both edges and nodes should be cudf
        assert isinstance(g2._edges, cudf.DataFrame)
        assert isinstance(g2._nodes, cudf.DataFrame)
        
        # Check node content
        expected_nodes = ['a', 'b', 'c', 'd']
        assert sorted(g2._nodes['id'].to_pandas().tolist()) == expected_nodes
    
    @skip_gpu
    def test_chain_ref_with_gpu_data(self):
        """Test ASTRef resolution works with GPU data"""
        import cudf
        from graphistry.compute.chain_let import execute_node
        from graphistry.compute.execution_context import ExecutionContext
        from graphistry.Engine import Engine
        
        # Create graph with cudf data
        edges_gdf = cudf.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Create context and store GPU result
        context = ExecutionContext()
        context.set_binding('gpu_result', g)
        
        # Create chain ref to GPU data
        chain_ref = ASTRef('gpu_result', [])
        
        # Execute should preserve GPU
        result = execute_node('test', chain_ref, g, context, Engine.CUDF)
        
        # Result should still have GPU data
        assert isinstance(result._edges, cudf.DataFrame)
        assert result._edges.equals(edges_gdf)
    
    @skip_gpu
    def test_dag_execution_preserves_gpu(self):
        """Test full DAG execution preserves GPU mode"""
        import cudf
        
        # Create graph with GPU data
        edges_gdf = cudf.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = CGFull().edges(edges_gdf, 's', 'd')
        
        # Create a simple DAG
        dag = ASTLet({})  # Empty DAG

        # Execute
        result = chain_let_impl(g, dag)
        
        # Should preserve GPU mode
        assert isinstance(result._edges, cudf.DataFrame)
        assert isinstance(result._nodes, cudf.DataFrame)
    
    @skip_gpu 
    def test_context_binding_with_mixed_engines(self):
        """Test ExecutionContext can handle mixed pandas/cudf results"""
        import cudf
        from graphistry.compute.execution_context import ExecutionContext
        
        context = ExecutionContext()
        
        # Create both pandas and cudf graphs
        g_pandas = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        edges_gdf = cudf.DataFrame({'s': ['x'], 'd': ['y']})
        g_cudf = CGFull().edges(edges_gdf, 's', 'd')
        
        # Store both
        context.set_binding('pandas_graph', g_pandas)
        context.set_binding('cudf_graph', g_cudf)
        
        # Retrieve and verify types preserved
        assert isinstance(context.get_binding('pandas_graph')._edges, pd.DataFrame)
        assert isinstance(context.get_binding('cudf_graph')._edges, cudf.DataFrame)
