"""
Test for None dereference bugs in dgl_utils.py

These tests reproduce the bugs found by MyPy strict mode analysis.
Issue #801
"""
import pandas as pd
import pytest


class TestDGLNoneDereference:
    """Test None dereference scenarios in DGL utilities"""

    def test_entity_to_index_none_dereference(self):
        """
        Reproduces bug in dgl_utils.py:314

        When _entity_to_index is None, calling len() on it raises TypeError.
        This test ensures the bug is fixed by checking for None first.
        """
        try:
            import graphistry
        except ImportError:
            pytest.skip("Graphistry not available")

        # Create minimal graph
        df = pd.DataFrame({
            'src': ['A', 'B', 'C'],
            'dst': ['B', 'C', 'A']
        })

        g = graphistry.edges(df, 'src', 'dst')

        # Manually set _entity_to_index to None to trigger the bug scenario
        # This simulates the condition where the attribute hasn't been initialized
        g._entity_to_index = None

        # After fix: This should NOT crash (None check prevents the bug)
        # Before fix: This would raise TypeError from isin() or len()
        try:
            g._check_nodes_lineup_with_edges()
            # If we get here, the fix worked
        except TypeError as e:
            if "NoneType" in str(e):
                pytest.fail(f"Bug still exists: {e}")
