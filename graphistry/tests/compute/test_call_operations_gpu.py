"""GPU tests for GFQL Call operations."""

import os
import pytest
import pandas as pd

from graphistry.tests.test_compute import CGFull
from graphistry.Engine import Engine
from graphistry.compute.ast import ASTCall, ASTLet, n
from graphistry.compute.chain_let import chain_let_impl
from graphistry.compute.gfql.call.executor import execute_call
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


class TestCpuOnlyPluginsCudfRoundTrip:
    """Verify CPU-only plugins (igraph, graphviz) handle cuDF input end-to-end.

    These tests use real cuDF DataFrames to validate:
    1. ensure_pandas converts cuDF to pandas before entering CPU libraries
    2. restore_engine converts output back to cuDF
    3. Nullable integer dtypes survive the round-trip via nullable=True
    """

    @skip_gpu
    def test_compute_igraph_cudf_round_trip(self):
        """compute_igraph accepts cuDF input, returns cuDF output."""
        import cudf

        edges_gdf = cudf.DataFrame({'s': [0, 1, 2, 2], 'd': [1, 2, 0, 3]})
        nodes_gdf = cudf.DataFrame({'n': [0, 1, 2, 3]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        assert isinstance(g._nodes, cudf.DataFrame)

        g2 = g.compute_igraph('pagerank')

        assert 'pagerank' in g2._nodes.columns
        assert isinstance(g2._nodes, cudf.DataFrame), \
            f"Expected cuDF output but got {type(g2._nodes)}"
        assert isinstance(g2._edges, cudf.DataFrame), \
            f"Expected cuDF edges but got {type(g2._edges)}"

    @skip_gpu
    def test_layout_igraph_cudf_round_trip(self):
        """layout_igraph accepts cuDF input, returns cuDF output."""
        import cudf

        edges_gdf = cudf.DataFrame({'s': [0, 1, 2, 2], 'd': [1, 2, 0, 3]})
        nodes_gdf = cudf.DataFrame({'n': [0, 1, 2, 3]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        g2 = g.layout_igraph('fr')

        assert 'x' in g2._nodes.columns
        assert 'y' in g2._nodes.columns
        assert isinstance(g2._nodes, cudf.DataFrame), \
            f"Expected cuDF output but got {type(g2._nodes)}"

    @skip_gpu
    def test_compute_igraph_preserves_nullable_int_dtypes(self):
        """Nullable integer columns survive the cuDF→pandas→igraph→pandas→cuDF round-trip."""
        import cudf

        nodes_gdf = cudf.DataFrame({
            'n': cudf.Series([0, 1, 2, 3], dtype='int64'),
            'group': cudf.Series([10, None, 30, None], dtype='Int64'),
        })
        edges_gdf = cudf.DataFrame({'s': [0, 1, 2, 2], 'd': [1, 2, 0, 3]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        g2 = g.compute_igraph('pagerank')

        assert isinstance(g2._nodes, cudf.DataFrame)
        # The 'group' column with nulls should not have become float
        assert g2._nodes['group'].null_count == 2

    @skip_gpu
    def test_execute_call_compute_igraph_cudf_engine(self):
        """execute_call with compute_igraph preserves cuDF through the GFQL call path."""
        import cudf

        edges_gdf = cudf.DataFrame({'source': [0, 1, 2, 2], 'target': [1, 2, 0, 3]})
        nodes_gdf = cudf.DataFrame({'node': [0, 1, 2, 3]})
        g = CGFull().edges(edges_gdf, 'source', 'target').nodes(nodes_gdf, 'node')

        result = execute_call(g, 'compute_igraph', {'alg': 'pagerank'}, Engine.CUDF)

        assert 'pagerank' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cuDF output but got {type(result._nodes)}"

    @skip_gpu
    def testensure_pandas_uses_nullable_on_real_cudf(self):
        """ensure_pandas calls to_pandas(nullable=True) on real cuDF DataFrames."""
        import cudf
        from graphistry.compute.engine_coercion import ensure_pandas

        gdf = cudf.DataFrame({
            'id': cudf.Series([1, 2, 3, None], dtype='Int64'),
        })

        result = ensure_pandas(gdf)

        assert isinstance(result, pd.DataFrame)
        assert not isinstance(result, cudf.DataFrame)
        # nullable=True should preserve Int64, not downgrade to float64
        assert result['id'].dtype == pd.Int64Dtype(), \
            f"Expected Int64 but got {result['id'].dtype}"

    @skip_gpu
    def testrestore_engine_converts_pandas_back_to_cudf(self):
        """restore_engine detects original cuDF engine and converts back."""
        import cudf
        from graphistry.compute.engine_coercion import restore_engine

        edges_gdf = cudf.DataFrame({'source': [0, 1, 2, 2], 'target': [1, 2, 0, 3]})
        nodes_gdf = cudf.DataFrame({'node': [0, 1, 2, 3]})
        g = CGFull().edges(edges_gdf, 'source', 'target').nodes(nodes_gdf, 'node')

        # Simulate what igraph does: convert to pandas result
        g_pandas = g.nodes(g._nodes.to_pandas(), 'node').edges(
            g._edges.to_pandas(), 'source', 'target')

        result = restore_engine(g_pandas, nodes_gdf, edges_gdf)

        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cuDF nodes but got {type(result._nodes)}"
        assert isinstance(result._edges, cudf.DataFrame), \
            f"Expected cuDF edges but got {type(result._edges)}"

    @skip_gpu
    def test_layout_graphviz_cudf_round_trip(self):
        """layout_graphviz accepts cuDF input, returns cuDF output."""
        import cudf
        try:
            import pygraphviz  # noqa: F401
        except ImportError:
            pytest.skip("pygraphviz not installed")

        edges_gdf = cudf.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        nodes_gdf = cudf.DataFrame({'n': [0, 1, 2]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        g2 = g.layout_graphviz('dot')

        assert 'x' in g2._nodes.columns
        assert 'y' in g2._nodes.columns
        assert isinstance(g2._nodes, cudf.DataFrame), \
            f"Expected cuDF output but got {type(g2._nodes)}"
        assert isinstance(g2._edges, cudf.DataFrame), \
            f"Expected cuDF edges but got {type(g2._edges)}"

    @skip_gpu
    def test_layout_graphviz_preserves_node_attributes(self):
        """layout_graphviz preserves existing cuDF node attributes through round-trip."""
        import cudf
        try:
            import pygraphviz  # noqa: F401
        except ImportError:
            pytest.skip("pygraphviz not installed")

        nodes_gdf = cudf.DataFrame({
            'n': [0, 1, 2],
            'score': [1.5, 2.5, 3.5],
        })
        edges_gdf = cudf.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        g2 = g.layout_graphviz('dot')

        assert 'score' in g2._nodes.columns
        assert 'x' in g2._nodes.columns
        assert len(g2._nodes) == 3

    @skip_gpu
    def test_layout_graphviz_preserves_nullable_int_dtypes(self):
        """Nullable integer columns survive the cuDF→graphviz→cuDF round-trip."""
        import cudf
        try:
            import pygraphviz  # noqa: F401
        except ImportError:
            pytest.skip("pygraphviz not installed")

        nodes_gdf = cudf.DataFrame({
            'n': cudf.Series([0, 1, 2], dtype='int64'),
            'group': cudf.Series([10, None, 30], dtype='Int64'),
        })
        edges_gdf = cudf.DataFrame({'s': [0, 1, 2], 'd': [1, 2, 0]})
        g = CGFull().edges(edges_gdf, 's', 'd').nodes(nodes_gdf, 'n')

        g2 = g.layout_graphviz('dot')

        assert isinstance(g2._nodes, cudf.DataFrame)
        assert g2._nodes['group'].null_count == 1

    @skip_gpu
    def test_execute_call_layout_graphviz_cudf_engine(self):
        """execute_call with layout_graphviz preserves cuDF through the GFQL call path."""
        import cudf
        try:
            import pygraphviz  # noqa: F401
        except ImportError:
            pytest.skip("pygraphviz not installed")

        edges_gdf = cudf.DataFrame({'source': [0, 1, 2], 'target': [1, 2, 0]})
        nodes_gdf = cudf.DataFrame({'node': [0, 1, 2]})
        g = CGFull().edges(edges_gdf, 'source', 'target').nodes(nodes_gdf, 'node')

        result = execute_call(g, 'layout_graphviz', {'prog': 'dot'}, Engine.CUDF)

        assert 'x' in result._nodes.columns
        assert isinstance(result._nodes, cudf.DataFrame), \
            f"Expected cuDF output but got {type(result._nodes)}"
