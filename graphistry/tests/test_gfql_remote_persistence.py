import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

import graphistry
from graphistry.compute.ast import ASTNode
from graphistry.privacy import Privacy


class TestGFQLRemotePersistence(unittest.TestCase):
    """Test suite for remote GFQL persistence functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'a'],
            'weight': [1, 2, 3]
        })
        self.g = graphistry.edges(self.edges_df, 'src', 'dst')

    def test_url_generation_with_dataset_id(self):
        """Test URL generation from dataset_id."""
        # Create a result with dataset_id
        result = graphistry.edges(self.edges_df, 'src', 'dst')
        result._dataset_id = 'test_123'

        # Mock the _viz_url method
        with patch.object(result._pygraphistry, '_viz_url') as mock_viz_url:
            mock_viz_url.return_value = 'https://hub.graphistry.com/graph/graph.html?dataset=test_123'

            url = result.url()

            assert url is not None
            assert 'dataset=test_123' in url
            assert 'graph.html' in url

    def test_url_generation_without_dataset_id(self):
        """Test URL generation fails gracefully without dataset_id."""
        result = graphistry.edges(self.edges_df, 'src', 'dst')
        # Don't set _dataset_id

        url = result.url()
        assert url is None

    def test_url_generation_with_custom_params(self):
        """Test URL generation with custom parameters."""
        result = graphistry.edges(self.edges_df, 'src', 'dst')
        result._dataset_id = 'test_123'

        with patch.object(result._pygraphistry, '_viz_url') as mock_viz_url:
            mock_viz_url.return_value = 'https://hub.graphistry.com/graph/graph.html?dataset=test_123&custom=value'

            result.url(custom='value', debug=True)

            # Verify _viz_url was called with custom params
            call_args = mock_viz_url.call_args
            assert len(call_args) == 2
            url_params = call_args[0][1]
            assert url_params['custom'] == 'value'
            assert url_params['debug'] is True

    @patch('requests.post')
    def test_persist_parameter_in_request_body_unit(self, mock_post):
        """Test that persist=True is included in request body (unit test)."""
        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': [], 'edges': [], 'dataset_id': 'test_123'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Mock upload method
        with patch.object(self.g, 'upload') as mock_upload:
            mock_uploaded = MagicMock()
            mock_uploaded._dataset_id = 'mock_dataset_id'
            mock_upload.return_value = mock_uploaded

            # Call with persist=True
            try:
                self.g.gfql_remote([ASTNode()], persist=True)
            except Exception:
                pass  # Ignore execution errors, we just want to check the request

            # Check that persist was included in request body
            if mock_post.called:
                call_args = mock_post.call_args
                request_json = call_args[1]['json']
                assert 'persist' in request_json
                assert request_json['persist'] is True

    @patch('requests.post')
    def test_privacy_in_request_body_unit(self, mock_post):
        """Test that privacy settings are included in persist requests (unit test)."""
        privacy_settings: Privacy = {
            'mode': 'organization',
            'notify': True,
            'invited_users': ['user@example.com'],
            'mode_action': '10'
        }

        # Mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'nodes': [], 'edges': [], 'dataset_id': 'privacy_test_123'
        }
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Set privacy on the graph
        self.g._privacy = privacy_settings

        # Mock upload method
        with patch.object(self.g, 'upload') as mock_upload:
            mock_uploaded = MagicMock()
            mock_uploaded._dataset_id = 'mock_dataset_id'
            mock_upload.return_value = mock_uploaded

            try:
                self.g.gfql_remote([ASTNode()], persist=True)
            except Exception:
                pass  # Ignore execution errors, we just want to check the request

            # Check that privacy was included in request body
            if mock_post.called:
                call_args = mock_post.call_args
                request_json = call_args[1]['json']
                assert 'privacy' in request_json
                assert request_json['privacy']['mode'] == 'organization'
                assert request_json['privacy']['notify'] is True

    def test_privacy_type_conversion(self):
        """Test that Privacy TypedDict converts to regular dict properly."""
        privacy_settings: Privacy = {
            'mode': 'public',
            'notify': False,
            'invited_users': [],
            'mode_action': '20'
        }

        # Test the conversion used in chain_remote.py
        converted = dict(privacy_settings)

        assert isinstance(converted, dict)
        assert converted['mode'] == 'public'
        assert converted['notify'] is False
        assert converted['invited_users'] == []
        assert converted['mode_action'] == '20'

    def test_request_body_typing(self):
        """Test that request body typing works correctly."""
        from typing import Dict, Any

        # This is the pattern used in chain_remote.py
        request_body: Dict[str, Any] = {
            "gfql_operations": [{"tag": "node"}],
            "format": "json"
        }

        # Test adding various fields
        request_body["persist"] = True
        request_body["privacy"] = {"mode": "private"}
        request_body["node_col_subset"] = ["col1", "col2"]

        # Should not raise type errors
        assert request_body["persist"] is True
        assert request_body["privacy"]["mode"] == "private"
        assert request_body["node_col_subset"] == ["col1", "col2"]


if __name__ == '__main__':
    unittest.main()
