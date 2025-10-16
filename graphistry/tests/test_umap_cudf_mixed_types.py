"""
Tests for UMAP DataFrame type consistency with engine='cuml'.

This test ensures that when using engine='cuml', both nodes and edges DataFrames
are consistently cuDF types, preventing mixed DataFrame types that cause chain
concatenation to fail with TypeError.

This test would FAIL without the fix in graphistry/umap_utils.py:_bind_xy_from_umap()
(lines 1055-1079) that ensures DataFrame type consistency after edges are created.

Related PR: #794
"""

import os
import pytest
import pandas as pd
import graphistry
from graphistry.utils.lazy_import import lazy_umap_import

# Check if umap is available
has_umap, _, _ = lazy_umap_import()

# Skip GPU tests if TEST_CUDF not set
skip_gpu = pytest.mark.skipif(
    not ("TEST_CUDF" in os.environ and os.environ["TEST_CUDF"] == "1"),
    reason="cudf tests need TEST_CUDF=1"
)


class TestUMAPCuDFMixedTypes:
    """Test UMAP DataFrame type consistency with cuML engine"""

    @skip_gpu
    def test_umap_cuml_engine_produces_consistent_cudf_types(self):
        """
        Test that UMAP with engine='cuml' returns consistent cuDF DataFrame types.

        BEFORE FIX: This would fail because:
        - nodes._nodes would be pandas DataFrame
        - nodes._edges would be cuDF DataFrame
        - Mixed types cause chain operations to fail

        AFTER FIX: Both nodes and edges are cuDF DataFrames.
        """
        import cudf

        # Create test data as pandas (simulating typical input)
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(20)],
            'col1': range(20),
            'col2': range(20, 40),
            'col3': range(40, 60),
            'col4': range(60, 80),
            'col5': range(80, 100),
        })

        # Create graph with pandas input
        g = graphistry.nodes(df, node='node')

        # Run UMAP with engine='cuml' - this should convert everything to cuDF
        g_umap = g.umap(
            X=['col1', 'col2', 'col3', 'col4', 'col5'],
            engine='cuml',
            n_neighbors=5
        )

        # Verify both nodes and edges are cuDF DataFrames
        assert g_umap._nodes is not None, "Nodes should exist after UMAP"
        assert g_umap._edges is not None, "Edges should exist after UMAP"

        # This is the critical assertion that would FAIL without the fix
        assert isinstance(g_umap._nodes, cudf.DataFrame), \
            f"Expected nodes to be cuDF DataFrame, got {type(g_umap._nodes)}"
        assert isinstance(g_umap._edges, cudf.DataFrame), \
            f"Expected edges to be cuDF DataFrame, got {type(g_umap._edges)}"

        # Both assertions passed means both are cuDF - no mixed types

    @skip_gpu
    def test_umap_cuml_chain_operations_work(self):
        """
        Test that chain operations work after UMAP with engine='cuml'.

        This test verifies the original bug is fixed: chain concatenation
        should not fail with TypeError due to mixed DataFrame types.
        """
        import cudf

        # Create test data
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(20)],
            'col1': range(20),
            'col2': range(20, 40),
            'col3': range(40, 60),
            'col4': range(60, 80),
            'col5': range(80, 100),
        })

        g = graphistry.nodes(df, node='node')

        # Use gfql to chain UMAP operations - this triggers chain concatenation
        # BEFORE FIX: This would fail with TypeError due to mixed types
        # AFTER FIX: This should succeed
        try:
            result = g.gfql([
                {
                    "type": "Call",
                    "function": "umap",
                    "params": {"X": ["col1", "col2", "col3", "col4", "col5"]}
                },
                {
                    "type": "Call",
                    "function": "name",
                    "params": {"name": "umap_result"}
                }
            ], engine='cudf')

            # Should succeed without TypeError
            assert result._nodes is not None
            assert result._edges is not None
            assert isinstance(result._nodes, cudf.DataFrame)
            assert isinstance(result._edges, cudf.DataFrame)

        except TypeError as e:
            # If we get TypeError about mixed types, the fix didn't work
            if "can only concatenate" in str(e):
                pytest.fail(f"Chain operation failed with mixed DataFrame types: {e}")
            raise

    @pytest.mark.skipif(not has_umap, reason="requires umap feature dependencies")
    def test_umap_pandas_engine_still_works(self):
        """
        Test that UMAP with default/pandas engine still works correctly.

        This is a regression test to ensure the fix doesn't break pandas mode.
        """
        df = pd.DataFrame({
            'node': [f'node_{i}' for i in range(20)],
            'col1': range(20),
            'col2': range(20, 40),
            'col3': range(40, 60),
            'col4': range(60, 80),
            'col5': range(80, 100),
        })

        g = graphistry.nodes(df, node='node')

        # Run UMAP with default engine (umap_learn/pandas)
        g_umap = g.umap(
            X=['col1', 'col2', 'col3', 'col4', 'col5'],
            engine='umap_learn',
            n_neighbors=5
        )

        # Verify nodes exist and are pandas
        assert g_umap._nodes is not None
        assert isinstance(g_umap._nodes, pd.DataFrame)

        # If edges were created, they should also be pandas
        if g_umap._edges is not None:
            assert isinstance(g_umap._edges, pd.DataFrame), \
                "Edges should be pandas when using umap_learn engine"
