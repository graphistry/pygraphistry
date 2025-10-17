"""
Test dataset_id and file_id invalidation logic.

These tests verify that PlotterBase methods correctly invalidate dataset_id
when modifying dataset-relevant attributes (encodings, name, description).

Based on AST analysis findings from plans/fix_dataset_file_id_invalidation/
"""
import pandas as pd
import pytest

import graphistry


class TestDatasetIDInvalidation:
    """Test that dataset_id is invalidated when dataset-relevant attributes change."""

    def setup_method(self):
        """Create a simple graph for testing."""
        self.nodes = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'color_val': ['red', 'blue', 'green'],
            'size_val': [10, 20, 30]
        })
        self.edges = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'weight': [1.0, 2.0]
        })
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    def test_encode_point_color_invalidates_dataset_id(self):
        """Bug #1: encode_point_color() should invalidate dataset_id."""
        # Simulate having a dataset_id (as if from upload)
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # encode_point_color modifies _complex_encodings (part of dataset)
        g2 = g1.encode_point_color('color_val', categorical_mapping={'red': 0xFF0000})

        # Should invalidate dataset_id because encoding changed
        assert g2._dataset_id is None, \
            "encode_point_color() should invalidate dataset_id (bug: it doesn't)"

    def test_encode_axis_invalidates_dataset_id(self):
        """Bug #2: encode_axis() should invalidate dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # encode_axis modifies _complex_encodings (part of dataset)
        g2 = g1.encode_axis(['x', 'y'])

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "encode_axis() should invalidate dataset_id (bug: it doesn't)"

    def test_name_invalidates_dataset_id(self):
        """Bug #3: name() should invalidate dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # name() modifies _name (part of dataset metadata)
        g2 = g1.name('My Graph')

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "name() should invalidate dataset_id (bug: it doesn't)"

    def test_description_invalidates_dataset_id(self):
        """Bug #4: description() should invalidate dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # description() modifies _description (part of dataset metadata)
        g2 = g1.description('My Description')

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "description() should invalidate dataset_id (bug: it doesn't)"

    def test_bind_point_color_invalidates_dataset_id(self):
        """Bug #5: bind() with encoding params should invalidate dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # bind() with point_color modifies simple bindings (part of dataset)
        g2 = g1.bind(point_color='color_val')

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "bind(point_color=...) should invalidate dataset_id (bug: it doesn't)"

    def test_bind_dataset_id_preserves_dataset_id(self):
        """bind() with dataset_id param should SET the ID, not invalidate it."""
        g1 = self.g
        assert g1._dataset_id is None

        # bind() with dataset_id parameter should SET it
        g2 = g1.bind(dataset_id='new_id_456')

        # Should preserve the ID we just set
        assert g2._dataset_id == 'new_id_456', \
            "bind(dataset_id=...) should SET dataset_id, not invalidate it"

    def test_settings_preserves_dataset_id(self):
        """settings() changes URL params only, should NOT invalidate dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # settings() modifies _url_params, _height, _render (NOT part of dataset)
        g2 = g1.settings(url_params={'filter': 'value'})

        # Should preserve dataset_id (already correct behavior)
        assert g2._dataset_id == 'test_id_123', \
            "settings() should preserve dataset_id (URL params not in dataset)"


class TestFileIDInvalidation:
    """Test that file IDs are invalidated when node/edge DataFrames change."""

    def setup_method(self):
        """Create a simple graph for testing."""
        self.nodes = pd.DataFrame({'id': [1, 2, 3]})
        self.edges = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    def test_nodes_invalidates_all_ids(self):
        """nodes() should invalidate all IDs (already correct)."""
        g1 = self.g.bind(
            dataset_id='dataset_123',
            nodes_file_id='nodes_456',
            edges_file_id='edges_789'
        )
        assert g1._dataset_id == 'dataset_123'
        assert g1._nodes_file_id == 'nodes_456'
        assert g1._edges_file_id == 'edges_789'

        # Change nodes - should invalidate everything
        new_nodes = pd.DataFrame({'id': [1, 2, 3, 4]})
        g2 = g1.nodes(new_nodes, 'id')

        # All IDs should be invalidated (already correct behavior)
        assert g2._dataset_id is None, "nodes() should invalidate dataset_id"
        assert g2._nodes_file_id is None, "nodes() should invalidate nodes_file_id"
        assert g2._edges_file_id is None, "nodes() should invalidate edges_file_id"

    def test_edges_invalidates_all_ids(self):
        """edges() should invalidate all IDs (already correct)."""
        g1 = self.g.bind(
            dataset_id='dataset_123',
            nodes_file_id='nodes_456',
            edges_file_id='edges_789'
        )

        # Change edges - should invalidate everything
        new_edges = pd.DataFrame({'src': [1], 'dst': [2]})
        g2 = g1.edges(new_edges, 'src', 'dst')

        # All IDs should be invalidated (already correct behavior)
        assert g2._dataset_id is None, "edges() should invalidate dataset_id"
        assert g2._nodes_file_id is None, "edges() should invalidate nodes_file_id"
        assert g2._edges_file_id is None, "edges() should invalidate edges_file_id"


class TestMixedScenarios:
    """Test realistic usage scenarios combining multiple operations."""

    def setup_method(self):
        """Create a simple graph for testing."""
        self.nodes = pd.DataFrame({
            'id': [1, 2, 3],
            'color_val': ['red', 'blue', 'green']
        })
        self.edges = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    def test_encode_after_upload_simulation(self):
        """
        Simulate: upload -> bind IDs -> encode -> should need re-upload.

        This is the original bug scenario: after upload, encoding changes
        should invalidate dataset_id, forcing re-upload.
        """
        # Simulate upload returning IDs
        g1 = self.g.bind(
            dataset_id='uploaded_dataset',
            nodes_file_id='uploaded_nodes',
            edges_file_id='uploaded_edges'
        )

        # User changes encoding
        g2 = g1.encode_point_color('color_val', categorical_mapping={'red': 0xFF0000})

        # dataset_id should be invalidated (encoding changed)
        # But file IDs should be preserved (DataFrames didn't change)
        assert g2._dataset_id is None, \
            "Encoding after upload should invalidate dataset_id"
        # Note: Currently file IDs also get invalidated, which is conservative but correct
        # A future optimization could preserve file IDs when only encodings change

    def test_chained_modifications(self):
        """Test multiple modifications in a chain."""
        g1 = self.g.bind(dataset_id='test_id')

        # Chain: name -> description -> encode
        g2 = g1.name('Test').description('Test Desc').encode_point_color(
            'color_val', categorical_mapping={'red': 0xFF0000}
        )

        # Final graph should have invalidated dataset_id
        assert g2._dataset_id is None, \
            "Chained modifications should invalidate dataset_id"


class TestPysaDiscoveredBugs:
    """Test bugs discovered by Pysa call graph analysis that AST analysis missed."""

    def setup_method(self):
        """Create a simple graph for testing."""
        self.nodes = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'color_val': ['red', 'blue', 'green']
        })
        self.edges = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3]
        })
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    def test_style_invalidates_dataset_id(self):
        """Pysa Bug #1: style() calls bind() and modifies _style without invalidating."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # style() modifies _style attribute (affects visualization)
        g2 = g1.style(bg={'color': 'black'}, fg={'blendMode': 'screen'})

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "style() should invalidate dataset_id (Pysa found this bug)"

    def test_addStyle_invalidates_dataset_id(self):
        """Pysa Bug #2: addStyle() calls bind() and modifies _style without invalidating."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # addStyle() modifies _style attribute (affects visualization)
        g2 = g1.addStyle(bg={'color': 'black'})

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            "addStyle() should invalidate dataset_id (Pysa found this bug)"

    def test_infer_labels_invalidates_dataset_id(self):
        """Pysa Bug #3: infer_labels() calls bind(point_title=...) which should invalidate."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # infer_labels() calls bind(point_title=...) to set label encoding
        g2 = g1.infer_labels()

        # Should invalidate dataset_id (via bind's conditional invalidation)
        assert g2._dataset_id is None, \
            "infer_labels() should invalidate dataset_id (calls bind with encoding)"


class TestAllEncodingMethods:
    """Comprehensive parametric tests for all encoding methods to prevent future regressions."""

    def setup_method(self):
        """Create a graph with all necessary columns for encoding tests."""
        self.nodes = pd.DataFrame({
            'id': [1, 2, 3],
            'color_val': ['red', 'blue', 'green'],
            'size_val': [10, 20, 30],
            'icon_val': ['user', 'star', 'heart'],
            'badge_val': ['A', 'B', 'C']
        })
        self.edges = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'edge_color': ['red', 'blue'],
            'edge_icon': ['arrow', 'line'],
            'edge_badge': ['X', 'Y']
        })
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    @pytest.mark.parametrize("method,kwargs", [
        ("encode_point_color", {"column": "color_val"}),
        ("encode_point_size", {"column": "size_val"}),
        ("encode_point_icon", {"column": "icon_val"}),
        ("encode_point_badge", {"column": "badge_val"}),
        ("encode_edge_color", {"column": "edge_color"}),
        ("encode_edge_icon", {"column": "edge_icon"}),
        ("encode_edge_badge", {"column": "edge_badge"}),
        ("encode_axis", {"rows": []}),
    ])
    def test_encoding_method_invalidates_dataset_id(self, method, kwargs):
        """ALL encoding methods must invalidate dataset_id - future-proof parametric test."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # Call the encoding method dynamically
        g2 = getattr(g1, method)(**kwargs)

        # Should invalidate dataset_id
        assert g2._dataset_id is None, \
            f"{method}() should invalidate dataset_id"


class TestBindConditionalLogic:
    """Test edge cases for bind()'s conditional invalidation logic."""

    def setup_method(self):
        """Create a simple graph for testing."""
        self.nodes = pd.DataFrame({
            'id': [1, 2, 3],
            'color_val': ['red', 'blue', 'green']
        })
        self.edges = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'weight': [1.0, 2.0]
        })
        self.g = graphistry.edges(self.edges, 'src', 'dst').nodes(self.nodes, 'id')

    def test_bind_multiple_encodings_invalidates(self):
        """bind() with multiple encoding params should invalidate."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # Multiple encodings at once
        g2 = g1.bind(point_color='color_val', edge_size='weight')

        # Should invalidate
        assert g2._dataset_id is None, \
            "bind() with multiple encodings should invalidate dataset_id"

    def test_bind_no_params_preserves(self):
        """bind() with no params should preserve dataset_id."""
        g1 = self.g.bind(dataset_id='test_id_123')
        assert g1._dataset_id == 'test_id_123'

        # Call bind() with no parameters
        g2 = g1.bind()

        # Should preserve the ID
        assert g2._dataset_id == 'test_id_123', \
            "bind() with no params should preserve dataset_id"

    def test_bind_dataset_id_with_encoding_preserves_id(self):
        """bind() setting dataset_id should preserve it even if also setting encoding."""
        g1 = self.g.bind(dataset_id='old_id')
        assert g1._dataset_id == 'old_id'

        # Set NEW dataset_id AND encoding simultaneously
        g2 = g1.bind(dataset_id='new_id', point_color='color_val')

        # Should preserve the NEW id we're explicitly setting
        assert g2._dataset_id == 'new_id', \
            "bind() with dataset_id param should preserve that ID even if also setting encodings"
