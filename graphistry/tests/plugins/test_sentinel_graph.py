import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import requests

from graphistry.PlotterBase import PlotterBase
from graphistry.plugins.sentinel_graph import SentinelGraphMixin
from graphistry.plugins_types.sentinel_graph_types import (
    SentinelGraphConfig,
    SentinelGraphConnectionError,
    SentinelGraphQueryError
)


# Sample response data for testing
SAMPLE_RESPONSE_FULL = {
    "Graph": {
        "Nodes": [
            {"Id": "node1", "Label": ["THREATACTOR"], "Properties": {"name": "Test Actor"}},
            {"Id": "node2", "Label": ["IDENTITY"], "Properties": {"name": "Test Identity"}}
        ],
        "Edges": []
    },
    "RawData": {
        "Rows": [
            {
                "Cols": [
                    {"Value": '{"_id": "node1", "_label": "THREATACTOR", "name": "Test Actor", "description": "A test threat actor"}'}
                ]
            },
            {
                "Cols": [
                    {"Value": '{"_id": "node2", "_label": "IDENTITY", "name": "Test Identity"}'}
                ]
            },
            {
                "Cols": [
                    {"Value": '{"_sourceId": "node1", "_targetId": "node2", "_label": "Targets", "count": 5}'}
                ]
            }
        ]
    }
}

SAMPLE_RESPONSE_RAWDATA_ONLY = {
    "RawData": {
        "Rows": [
            {"Cols": [{"Value": '{"_id": "node3", "_label": "MALWARE", "name": "Test Malware"}'}]},
            {"Cols": [{"Value": '{"_sourceId": "node3", "_targetId": "node1", "_label": "Uses"}'}]}
        ]
    }
}

SAMPLE_RESPONSE_EMPTY = {
    "Graph": {"Nodes": [], "Edges": []},
    "RawData": {"Rows": []}
}

SAMPLE_RESPONSE_MALFORMED = {
    "RawData": {
        "Rows": [
            {"Cols": [{"Value": 'not valid json'}]},
            {"Cols": [{"Value": '{"_id": "node4", "_label": "VALID"}'}]},
            {"Cols": [{"Value": None}]}
        ]
    }
}


class TestSentinelGraphConfiguration:
    """Test configuration and setup methods"""

    def test_configure_with_defaults(self):
        """Test basic configuration with default values"""
        g = PlotterBase()
        result = g.configure_sentinel_graph(graph_instance="TestInstance")

        assert g.session.sentinel_graph is not None
        assert g.session.sentinel_graph.graph_instance == "TestInstance"
        assert g.session.sentinel_graph.api_endpoint == "api.securityplatform.microsoft.com"
        assert g.session.sentinel_graph.timeout == 60
        assert g.session.sentinel_graph.max_retries == 3
        assert result is g  # Check method chaining

    def test_configure_with_custom_params(self):
        """Test configuration with custom parameters"""
        g = PlotterBase()
        g.configure_sentinel_graph(
            graph_instance="CustomInstance",
            api_endpoint="custom.endpoint.com",
            auth_scope="custom-scope/.default",
            timeout=120,
            max_retries=5,
            retry_backoff_factor=3.0
        )

        cfg = g.session.sentinel_graph
        assert cfg.graph_instance == "CustomInstance"
        assert cfg.api_endpoint == "custom.endpoint.com"
        assert cfg.auth_scope == "custom-scope/.default"
        assert cfg.timeout == 120
        assert cfg.max_retries == 5
        assert cfg.retry_backoff_factor == 3.0

    def test_configure_with_service_principal(self):
        """Test configuration with service principal credentials"""
        g = PlotterBase()
        g.configure_sentinel_graph(
            graph_instance="TestInstance",
            tenant_id="test-tenant",
            client_id="test-client",
            client_secret="test-secret"
        )

        cfg = g.session.sentinel_graph
        assert cfg.tenant_id == "test-tenant"
        assert cfg.client_id == "test-client"
        assert cfg.client_secret == "test-secret"

    def test_sentinel_graph_from_credential(self):
        """Test configuration using existing credential"""
        mock_credential = Mock()
        g = PlotterBase()
        result = g.sentinel_graph_from_credential(
            mock_credential,
            "TestInstance"
        )

        assert g.session.sentinel_graph.credential is mock_credential
        assert g.session.sentinel_graph.graph_instance == "TestInstance"
        assert result is g

    def test_config_not_configured_error(self):
        """Test error when accessing config before configuration"""
        g = PlotterBase()
        with pytest.raises(ValueError, match="not configured"):
            _ = g._sentinel_graph_config

    def test_sentinel_graph_close(self):
        """Test closing and clearing token cache"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")
        g.session.sentinel_graph._token = "test-token"
        g.session.sentinel_graph._token_expiry = 12345.0

        g.sentinel_graph_close()

        assert g.session.sentinel_graph._token is None
        assert g.session.sentinel_graph._token_expiry is None


class TestAuthenticationToken:
    """Test authentication token retrieval and caching"""

    @patch('graphistry.plugins.sentinel_graph.InteractiveBrowserCredential')
    def test_get_auth_token_interactive(self, mock_cred_class):
        """Test token retrieval with interactive browser credential"""
        mock_token = Mock()
        mock_token.token = "test-token-123"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        mock_credential = Mock()
        mock_credential.get_token.return_value = mock_token
        mock_cred_class.return_value = mock_credential

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        token = g._get_auth_token()

        assert token == "test-token-123"
        assert g.session.sentinel_graph._token == "test-token-123"
        mock_credential.get_token.assert_called_once_with(
            "73c2949e-da2d-457a-9607-fcc665198967/.default"
        )

    @patch('graphistry.plugins.sentinel_graph.ClientSecretCredential')
    def test_get_auth_token_service_principal(self, mock_cred_class):
        """Test token retrieval with service principal"""
        mock_token = Mock()
        mock_token.token = "sp-token-456"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        mock_credential = Mock()
        mock_credential.get_token.return_value = mock_token
        mock_cred_class.return_value = mock_credential

        g = PlotterBase()
        g.configure_sentinel_graph(
            graph_instance="TestInstance",
            tenant_id="tenant",
            client_id="client",
            client_secret="secret"
        )

        token = g._get_auth_token()

        assert token == "sp-token-456"
        mock_cred_class.assert_called_once_with(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret"
        )

    def test_token_caching(self):
        """Test that valid tokens are cached and reused"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        # Manually set a valid cached token
        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        g.session.sentinel_graph._token = "cached-token"
        g.session.sentinel_graph._token_expiry = future_time

        with patch('graphistry.plugins.sentinel_graph.InteractiveBrowserCredential') as mock_cred:
            token = g._get_auth_token()

            # Should use cached token, not call credential
            assert token == "cached-token"
            mock_cred.assert_not_called()

    def test_token_refresh_when_expired(self):
        """Test that expired tokens trigger refresh"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        # Set an expired token
        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        g.session.sentinel_graph._token = "expired-token"
        g.session.sentinel_graph._token_expiry = past_time

        mock_token = Mock()
        mock_token.token = "new-token"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        with patch('graphistry.plugins.sentinel_graph.InteractiveBrowserCredential') as mock_cred_class:
            mock_credential = Mock()
            mock_credential.get_token.return_value = mock_token
            mock_cred_class.return_value = mock_credential

            token = g._get_auth_token()

            assert token == "new-token"
            assert g.session.sentinel_graph._token == "new-token"


class TestQueryExecution:
    """Test query execution and HTTP handling"""

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_success(self, mock_auth, mock_post):
        """Test successful query execution"""
        mock_auth.return_value = "test-token"

        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = json.dumps(SAMPLE_RESPONSE_FULL).encode('utf-8')
        mock_post.return_value = mock_response

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._sentinel_graph_query("MATCH (n) RETURN n", "GQL")

        assert result == mock_response.content
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['query'] == "MATCH (n) RETURN n"
        assert call_kwargs['json']['queryLanguage'] == "GQL"
        assert call_kwargs['headers']['Authorization'] == "Bearer test-token"
        assert call_kwargs['timeout'] == 60

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_http_error(self, mock_auth, mock_post):
        """Test query execution with HTTP error"""
        mock_auth.return_value = "test-token"

        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request: Invalid query syntax"
        mock_post.return_value = mock_response

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        with pytest.raises(SentinelGraphQueryError, match="400"):
            g._sentinel_graph_query("INVALID QUERY", "GQL")

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch('time.sleep')  # Mock sleep to speed up test
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_retry_on_timeout(self, mock_auth, mock_sleep, mock_post):
        """Test retry logic on timeout"""
        mock_auth.return_value = "test-token"

        # First 2 calls timeout, 3rd succeeds
        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            Mock(status_code=200, content=b'{"result": "success"}')
        ]

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance", max_retries=3)

        result = g._sentinel_graph_query("MATCH (n) RETURN n", "GQL")

        assert result == b'{"result": "success"}'
        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2  # Slept between retries

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch('time.sleep')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_max_retries_exceeded(self, mock_auth, mock_sleep, mock_post):
        """Test failure after max retries"""
        mock_auth.return_value = "test-token"

        mock_post.side_effect = requests.exceptions.ConnectionError("Connection failed")

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance", max_retries=3)

        with pytest.raises(SentinelGraphConnectionError, match="3 retries"):
            g._sentinel_graph_query("MATCH (n) RETURN n", "GQL")

        assert mock_post.call_count == 3

    @patch.object(SentinelGraphMixin, '_sentinel_graph_query')
    @patch.object(SentinelGraphMixin, '_parse_graph_response')
    def test_sentinel_graph_main_method(self, mock_parse, mock_query):
        """Test main sentinel_graph method"""
        mock_query.return_value = b'test-response'
        mock_parse.return_value = Mock()

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g.sentinel_graph("MATCH (n) RETURN n")

        mock_query.assert_called_once_with("MATCH (n) RETURN n", 'GQL')
        mock_parse.assert_called_once_with(b'test-response')
        assert result is mock_parse.return_value


class TestResponseParsing:
    """Test node and edge extraction from various response formats"""

    def test_extract_nodes_full_response(self):
        """Test node extraction from complete response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_FULL)

        assert len(nodes_df) == 2
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns
        assert set(nodes_df['id']) == {'node1', 'node2'}

    def test_extract_nodes_rawdata_only(self):
        """Test node extraction from RawData only"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_RAWDATA_ONLY)

        assert len(nodes_df) == 1
        assert nodes_df.iloc[0]['id'] == 'node3'
        assert nodes_df.iloc[0]['label'] == 'MALWARE'

    def test_extract_nodes_deduplication(self):
        """Test node deduplication keeps most complete record"""
        duplicate_response = {
            "RawData": {
                "Rows": [
                    {"Cols": [{"Value": '{"_id": "dup1", "_label": "TEST"}'}]},
                    {"Cols": [{"Value": '{"_id": "dup1", "_label": "TEST", "name": "Complete", "description": "Full info"}'}]},
                    {"Cols": [{"Value": '{"_id": "dup1", "_label": "TEST", "name": "Partial"}'}]}
                ]
            }
        }

        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(duplicate_response)

        assert len(nodes_df) == 1
        assert nodes_df.iloc[0]['name'] == 'Complete'  # Most complete record
        assert nodes_df.iloc[0]['description'] == 'Full info'

    def test_extract_nodes_malformed_data(self):
        """Test graceful handling of malformed data"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_MALFORMED)

        # Should extract the valid node and skip invalid ones
        assert len(nodes_df) == 1
        assert nodes_df.iloc[0]['id'] == 'node4'

    def test_extract_nodes_empty_response(self):
        """Test extraction from empty response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_EMPTY)

        assert len(nodes_df) == 0
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns

    def test_extract_edges_full_response(self):
        """Test edge extraction from complete response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_FULL)

        assert len(edges_df) == 1
        assert edges_df.iloc[0]['source'] == 'node1'
        assert edges_df.iloc[0]['target'] == 'node2'
        assert edges_df.iloc[0]['edge'] == 'Targets'
        assert edges_df.iloc[0]['count'] == 5

    def test_extract_edges_rawdata_only(self):
        """Test edge extraction from RawData only"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_RAWDATA_ONLY)

        assert len(edges_df) == 1
        assert edges_df.iloc[0]['source'] == 'node3'
        assert edges_df.iloc[0]['target'] == 'node1'

    def test_extract_edges_empty_response(self):
        """Test edge extraction from empty response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_EMPTY)

        assert len(edges_df) == 0
        assert 'source' in edges_df.columns
        assert 'target' in edges_df.columns


class TestGraphConversion:
    """Test full graph conversion workflow"""

    def test_convert_bytes_response(self):
        """Test conversion from bytes response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response_bytes = json.dumps(SAMPLE_RESPONSE_FULL).encode('utf-8')
        result = g._parse_graph_response(response_bytes)

        assert result._node is not None
        assert result._edge is not None
        assert len(result._node) == 2
        assert len(result._edge) == 1

    def test_convert_dict_response(self):
        """Test conversion from dict response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._parse_graph_response(SAMPLE_RESPONSE_FULL)

        assert result._node is not None
        assert result._edge is not None

    def test_convert_invalid_json(self):
        """Test error on invalid JSON bytes"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        with pytest.raises(SentinelGraphQueryError, match="parse.*JSON"):
            g._parse_graph_response(b'not valid json')

    def test_convert_empty_response(self):
        """Test conversion of empty response"""
        g = PlotterBase()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._parse_graph_response(SAMPLE_RESPONSE_EMPTY)

        assert len(result._node) == 0
        assert len(result._edge) == 0


# Integration test markers
@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires live API credentials")
class TestSentinelGraphIntegration:
    """Integration tests requiring live API access"""

    def test_live_query(self):
        """Test actual query against live API (requires credentials)"""
        # This would be run manually with real credentials
        pass
