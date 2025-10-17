"""
Test GFQL remote metadata hydration.

These tests verify that gfql_remote() properly hydrates server-computed metadata
back into the returned Plottable. When remote GFQL operations (like call('umap'))
modify bindings, encodings, or other metadata, the client should receive and apply
those changes.
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import graphistry
from graphistry.compute.ast import ASTNode


# Mock metadata structures mirroring arrow uploader format
MOCK_UMAP_METADATA = {
    'bindings': {
        'node': 'umap_node_id',
        'source': 'umap_src',
        'destination': 'umap_dst'
    },
    'encodings': {
        'point_color': 'umap_cluster',
        'point_size': 'umap_importance',
        'complex_encodings': {
            'node_encodings': {
                'default': {
                    'pointColorEncoding': {
                        'graphType': 'point',
                        'encodingType': 'color',
                        'attribute': 'umap_cluster',
                        'variation': 'categorical'
                    }
                }
            }
        }
    }
}

MOCK_NAME_DESCRIPTION_METADATA = {
    'metadata': {
        'name': 'UMAP Clustered Graph',
        'description': 'Graph after UMAP dimensionality reduction and clustering'
    }
}

MOCK_STYLE_METADATA = {
    'style': {
        'bg': {'color': '#000000'},
        'fg': {'blendMode': 'screen'},
        'page': {'title': 'UMAP Analysis'}
    }
}

MOCK_FULL_METADATA = {
    'bindings': {
        'node': 'umap_node_id',
        'source': 'umap_src',
        'destination': 'umap_dst',
        'edge': 'edge_id'
    },
    'encodings': {
        'point_color': 'umap_cluster',
        'point_size': 'umap_importance',
        'edge_color': 'edge_type',
        'complex_encodings': {
            'node_encodings': {
                'default': {
                    'pointColorEncoding': {
                        'graphType': 'point',
                        'encodingType': 'color',
                        'attribute': 'umap_cluster',
                        'variation': 'categorical'
                    }
                }
            }
        }
    },
    'metadata': {
        'name': 'Complete UMAP Analysis',
        'description': 'Full graph with UMAP and clustering'
    },
    'style': {
        'bg': {'color': '#000000'},
        'fg': {'blendMode': 'screen'}
    }
}


class TestGFQLRemoteMetadataHydration(unittest.TestCase):
    """Test that gfql_remote() hydrates server metadata into returned Plottable."""

    def setUp(self):
        """Set up test fixtures."""
        self.nodes_df = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'x': [1.0, 2.0, 3.0],
            'y': [1.0, 2.0, 3.0]
        })
        self.edges_df = pd.DataFrame({
            'src': [1, 2],
            'dst': [2, 3],
            'weight': [1.0, 2.0]
        })
        self.g = graphistry.edges(self.edges_df, 'src', 'dst').nodes(self.nodes_df, 'id')

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_umap_bindings_hydrated(self, mock_post):
        """UMAP changes src/dst bindings - verify bindings transfer back."""
        # Mock response with updated bindings
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_UMAP_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Set dataset_id to avoid upload
        self.g._dataset_id = 'test_dataset_123'

        # Execute remote GFQL (simulating call('umap'))
        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify bindings were hydrated
        assert result._node == 'umap_node_id', \
            f"Expected node binding 'umap_node_id', got '{result._node}'"
        assert result._source == 'umap_src', \
            f"Expected source binding 'umap_src', got '{result._source}'"
        assert result._destination == 'umap_dst', \
            f"Expected destination binding 'umap_dst', got '{result._destination}'"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_umap_encodings_hydrated(self, mock_post):
        """UMAP adds color encoding - verify complex_encodings transfer back."""
        # Mock response with encoding metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_UMAP_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify simple encodings were hydrated
        assert result._point_color == 'umap_cluster', \
            f"Expected point_color 'umap_cluster', got '{result._point_color}'"
        assert result._point_size == 'umap_importance', \
            f"Expected point_size 'umap_importance', got '{result._point_size}'"

        # Verify complex encodings were hydrated
        assert hasattr(result, '_complex_encodings'), \
            "Result should have _complex_encodings attribute"
        assert result._complex_encodings is not None, \
            "Complex encodings should not be None"
        assert 'node_encodings' in result._complex_encodings, \
            "Complex encodings should contain node_encodings"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_name_description_hydrated(self, mock_post):
        """call('name') - verify metadata transfers back."""
        # Mock response with name/description metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_NAME_DESCRIPTION_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify name and description were hydrated
        assert result._name == 'UMAP Clustered Graph', \
            f"Expected name 'UMAP Clustered Graph', got '{result._name}'"
        assert result._description == 'Graph after UMAP dimensionality reduction and clustering', \
            f"Expected description to match, got '{result._description}'"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_style_hydrated(self, mock_post):
        """call('style') - verify style transfers back."""
        # Mock response with style metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_STYLE_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify style was hydrated
        assert hasattr(result, '_style'), "Result should have _style attribute"
        assert result._style is not None, "Style should not be None"
        assert 'bg' in result._style, "Style should contain bg"
        assert result._style['bg']['color'] == '#000000', \
            f"Expected bg color '#000000', got '{result._style['bg']['color']}'"
        assert result._style['fg']['blendMode'] == 'screen', \
            f"Expected fg blendMode 'screen', got '{result._style['fg']['blendMode']}'"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_empty_metadata_doesnt_break(self, mock_post):
        """No metadata or partial metadata - should not error."""
        # Mock response without metadata field
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123'
            # No 'metadata' field
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        # Should not raise an error
        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify we got a valid result
        assert result is not None
        assert hasattr(result, '_nodes')
        assert hasattr(result, '_edges')

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_partial_metadata_hydrated(self, mock_post):
        """Partial metadata (only bindings) - should hydrate what's present."""
        # Mock response with only bindings metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': {
                'bindings': {
                    'source': 'new_src',
                    'destination': 'new_dst'
                    # No 'node' or 'edge'
                }
                # No 'encodings', 'metadata', or 'style'
            }
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify partial bindings were hydrated
        assert result._source == 'new_src', \
            f"Expected source 'new_src', got '{result._source}'"
        assert result._destination == 'new_dst', \
            f"Expected destination 'new_dst', got '{result._destination}'"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_full_metadata_hydrated(self, mock_post):
        """Complete metadata - verify all fields transfer back."""
        # Mock response with complete metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_FULL_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify all bindings
        assert result._node == 'umap_node_id'
        assert result._source == 'umap_src'
        assert result._destination == 'umap_dst'
        assert result._edge == 'edge_id'

        # Verify all encodings
        assert result._point_color == 'umap_cluster'
        assert result._point_size == 'umap_importance'
        assert result._edge_color == 'edge_type'
        assert result._complex_encodings is not None

        # Verify metadata
        assert result._name == 'Complete UMAP Analysis'
        assert result._description == 'Full graph with UMAP and clustering'

        # Verify style
        assert result._style is not None
        assert result._style['bg']['color'] == '#000000'

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_metadata_preserves_existing_dataframes(self, mock_post):
        """Metadata hydration should not modify existing DataFrames."""
        # Mock response with metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': MOCK_UMAP_METADATA
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Verify DataFrames are present and have correct data
        assert result._nodes is not None
        assert result._edges is not None
        assert len(result._nodes) == len(self.nodes_df)
        assert len(result._edges) == len(self.edges_df)

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_zip_format_metadata_hydrated(self, mock_post):
        """Zip format (parquet) - verify metadata hydration works."""
        import zipfile
        import json
        from io import BytesIO

        # Create mock zip response with metadata
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zip_ref:
            # Add node/edge data
            nodes_parquet = BytesIO()
            self.nodes_df.to_parquet(nodes_parquet)
            zip_ref.writestr('nodes.parquet', nodes_parquet.getvalue())

            edges_parquet = BytesIO()
            self.edges_df.to_parquet(edges_parquet)
            zip_ref.writestr('edges.parquet', edges_parquet.getvalue())

            # Add metadata.json with GFQL-computed metadata
            metadata = {
                'dataset_id': 'zip_test_123',
                'gfql_metadata': MOCK_UMAP_METADATA  # GFQL-computed metadata separate from persistence metadata
            }
            zip_ref.writestr('metadata.json', json.dumps(metadata))

        mock_response = MagicMock()
        mock_response.content = zip_buffer.getvalue()
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='parquet', output_type='all', api_token='test_token')

        # Verify GFQL metadata was hydrated (this will fail until we implement it)
        # Note: This is separate from persistence metadata (dataset_id)
        assert result._source == 'umap_src', \
            "Zip format should also support GFQL metadata hydration"


class TestMetadataHydrationEdgeCases(unittest.TestCase):
    """Test edge cases for metadata hydration."""

    def setUp(self):
        """Set up test fixtures."""
        self.nodes_df = pd.DataFrame({'id': [1, 2, 3]})
        self.edges_df = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
        self.g = graphistry.edges(self.edges_df, 'src', 'dst').nodes(self.nodes_df, 'id')

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_malformed_metadata_graceful_handling(self, mock_post):
        """Malformed metadata should be handled gracefully without crashing."""
        # Mock response with malformed metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': {
                'bindings': 'invalid_should_be_dict',  # Wrong type
                'encodings': ['invalid', 'should', 'be', 'dict']  # Wrong type
            }
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        # Should not crash, but may log warnings
        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Should still return a valid result
        assert result is not None
        assert result._nodes is not None

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_metadata_with_none_values(self, mock_post):
        """Metadata with None values should be handled gracefully."""
        # Mock response with None values in metadata
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': {
                'bindings': {
                    'node': None,
                    'source': 'src',
                    'destination': None
                },
                'encodings': {
                    'point_color': None
                }
            }
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        self.g._dataset_id = 'test_dataset_123'

        result = self.g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Should hydrate non-None values
        assert result._source == 'src', \
            "Should hydrate non-None values"

    @patch('graphistry.compute.chain_remote.requests.post')
    def test_metadata_overrides_existing_bindings(self, mock_post):
        """Server metadata should override existing client bindings."""
        # Start with initial bindings
        g = self.g.bind(source='old_src', destination='old_dst', point_color='old_color')

        # Mock response with new bindings
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': self.nodes_df.to_dict('records'),
            'edges': self.edges_df.to_dict('records'),
            'dataset_id': 'test_123',
            'metadata': {
                'bindings': {
                    'source': 'new_src',
                    'destination': 'new_dst'
                },
                'encodings': {
                    'point_color': 'new_color'
                }
            }
        }
        mock_response.ok = True
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        g._dataset_id = 'test_dataset_123'

        result = g.gfql_remote([ASTNode()], format='json', api_token='test_token')

        # Server metadata should override client bindings
        assert result._source == 'new_src', \
            "Server metadata should override existing bindings"
        assert result._destination == 'new_dst', \
            "Server metadata should override existing bindings"
        assert result._point_color == 'new_color', \
            "Server metadata should override existing encodings"


if __name__ == '__main__':
    unittest.main()
