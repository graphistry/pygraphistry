"""
Test for None dereference bugs in sugiyamaLayout.py

These tests reproduce the bugs found by MyPy strict mode analysis.
Issue #801
"""
import pandas as pd
import pytest


class TestSugiyamaNoneDereference:
    """Test None dereference scenarios in Sugiyama layout"""

    def test_create_dummies_none_layer(self):
        """
        Reproduces bug in sugiyamaLayout.py:483

        When layoutVertices[vertex].layer is None, passing it to range() raises TypeError.
        This test ensures the bug is fixed by checking for None first.
        """
        try:
            from graphistry.layout.sugiyama.sugiyamaLayout import SugiyamaLayout
            from graphistry.layout.graph import Vertex, Edge, Graph
            from graphistry.layout.utils import LayoutVertex
        except ImportError:
            pytest.skip("Sugiyama layout not available")

        # Create minimal graph
        v1 = Vertex('A')
        v2 = Vertex('B')
        e = Edge(v1, v2)
        g = Graph([v1, v2], [e])

        # Create layout
        sug = SugiyamaLayout(g)

        # Manually set layer to None to trigger the bug scenario
        sug.layoutVertices[v1].layer = None
        sug.layoutVertices[v2].layer = None

        # After fix: This should NOT crash (None check prevents the bug)
        # Before fix: This would raise TypeError from range()
        try:
            sug.create_dummies(e)
            # If we get here, the fix worked
        except TypeError as e_err:
            if "NoneType" in str(e_err):
                pytest.fail(f"Bug still exists: {e_err}")

    def test_layout_edges_none_layer(self):
        """
        Reproduces bugs in sugiyamaLayout.py:758,760

        When layoutVertices[vertex].layer is None, passing it to range() raises TypeError.
        This test ensures the bug is fixed by checking for None first.
        """
        try:
            from graphistry.layout.sugiyama.sugiyamaLayout import SugiyamaLayout
            from graphistry.layout.graph import Vertex, Edge, Graph
            from graphistry.layout.utils import Rectangle
        except ImportError:
            pytest.skip("Sugiyama layout not available")

        # Create minimal graph with views that have setpath method
        v1 = Vertex('A')
        v2 = Vertex('B')
        v1.view = Rectangle()
        v2.view = Rectangle()
        v1.view.xy = (0, 0)
        v2.view.xy = (1, 1)
        e = Edge(v1, v2)
        e.view = Rectangle()
        # Add setpath method to avoid AttributeError
        e.view.setpath = lambda x: None
        g = Graph([v1, v2], [e])

        # Create layout
        sug = SugiyamaLayout(g)

        # Set up dummy controls to trigger layout_edges path
        sug.ctrls[e] = {}

        # Manually set layer to None to trigger the bug scenario
        sug.layoutVertices[v1].layer = None
        sug.layoutVertices[v2].layer = None

        # After fix: This should NOT crash (None check prevents the bug)
        # Before fix: This would raise TypeError from range()
        try:
            sug.layout_edges()
            # If we get here, the fix worked
        except TypeError as e_err:
            if "NoneType" in str(e_err):
                pytest.fail(f"Bug still exists: {e_err}")
