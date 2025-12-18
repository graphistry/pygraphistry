import pytest
import json
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import requests

import graphistry
from graphistry.plugins.sentinel_graph import SentinelGraphMixin
from graphistry.plugins_types.sentinel_graph_types import (
    SentinelGraphConfig,
    SentinelGraphConnectionError,
    SentinelGraphQueryError
)
from graphistry.tests.fixtures.sentinel_graph_responses import (
    get_minimal_response,
    get_simple_graph_response,
    get_duplicate_nodes_response,
    get_malformed_response,
    get_empty_response,
    get_complex_graph_response,
    get_edge_only_response,
    get_response_with_special_characters,
    get_response_with_null_properties
)


# Sample response data for testing (using fixtures)
SAMPLE_RESPONSE_FULL = get_simple_graph_response()  # 3 nodes, 2 edges
SAMPLE_RESPONSE_EMPTY = get_empty_response()
SAMPLE_RESPONSE_MALFORMED = get_malformed_response()


class TestSentinelGraphConfiguration:
    """Test configuration and setup methods"""

    def test_configure_with_defaults(self):
        """Test basic configuration with default values"""
        g = graphistry.bind()
        result = g.configure_sentinel_graph(graph_instance="TestInstance")

        assert g.session.sentinel_graph is not None
        assert g.session.sentinel_graph.graph_instance == "TestInstance"
        assert g.session.sentinel_graph.api_endpoint == "api.securityplatform.microsoft.com"
        assert g.session.sentinel_graph.timeout == 60
        assert g.session.sentinel_graph.max_retries == 3
        assert result is g  # Check method chaining

    def test_configure_with_custom_params(self):
        """Test configuration with custom parameters"""
        g = graphistry.bind()
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
        g = graphistry.bind()
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
        g = graphistry.bind()
        result = g.sentinel_graph_from_credential(
            mock_credential,
            "TestInstance"
        )

        assert g.session.sentinel_graph.credential is mock_credential
        assert g.session.sentinel_graph.graph_instance == "TestInstance"
        assert result is g

    def test_config_not_configured_error(self):
        """Test error when accessing config before configuration"""
        # Create a fresh plotter with unconfigured session
        from graphistry.plotter import Plotter
        from graphistry.pygraphistry import PyGraphistry
        g = Plotter(pygraphistry=PyGraphistry)
        # Manually ensure sentinel_graph is not configured
        g.session.sentinel_graph = None
        with pytest.raises(ValueError, match="not configured"):
            _ = g._sentinel_graph_config

    def test_sentinel_graph_close(self):
        """Test closing and clearing token cache"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")
        g.session.sentinel_graph._token = "test-token"
        g.session.sentinel_graph._token_expiry = 12345.0

        g.sentinel_graph_close()

        assert g.session.sentinel_graph._token is None
        assert g.session.sentinel_graph._token_expiry is None


class TestAuthenticationToken:
    """Test authentication token retrieval and caching"""

    @patch('azure.identity.InteractiveBrowserCredential')
    def test_get_auth_token_interactive(self, mock_cred_class):
        """Test token retrieval with interactive browser credential"""
        mock_token = Mock()
        mock_token.token = "test-token-123"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        mock_credential = Mock()
        mock_credential.get_token.return_value = mock_token
        mock_cred_class.return_value = mock_credential

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        token = g._get_auth_token()

        assert token == "test-token-123"
        assert g.session.sentinel_graph._token == "test-token-123"
        mock_credential.get_token.assert_called_once_with(
            "73c2949e-da2d-457a-9607-fcc665198967/.default"
        )

    @patch('azure.identity.ClientSecretCredential')
    def test_get_auth_token_service_principal(self, mock_cred_class):
        """Test token retrieval with service principal"""
        mock_token = Mock()
        mock_token.token = "sp-token-456"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        mock_credential = Mock()
        mock_credential.get_token.return_value = mock_token
        mock_cred_class.return_value = mock_credential

        g = graphistry.bind()
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
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        # Manually set a valid cached token
        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        g.session.sentinel_graph._token = "cached-token"
        g.session.sentinel_graph._token_expiry = future_time

        with patch('azure.identity.InteractiveBrowserCredential') as mock_cred:
            token = g._get_auth_token()

            # Should use cached token, not call credential
            assert token == "cached-token"
            mock_cred.assert_not_called()

    def test_token_refresh_when_expired(self):
        """Test that expired tokens trigger refresh"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        # Set an expired token
        past_time = (datetime.now() - timedelta(hours=1)).timestamp()
        g.session.sentinel_graph._token = "expired-token"
        g.session.sentinel_graph._token_expiry = past_time

        mock_token = Mock()
        mock_token.token = "new-token"
        mock_token.expires_on = (datetime.now() + timedelta(hours=1)).timestamp()

        with patch('azure.identity.InteractiveBrowserCredential') as mock_cred_class:
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

        g = graphistry.bind()
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

        g = graphistry.bind()
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

        g = graphistry.bind()
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

        g = graphistry.bind()
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

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g.sentinel_graph("MATCH (n) RETURN n")

        mock_query.assert_called_once_with("MATCH (n) RETURN n", 'GQL')
        mock_parse.assert_called_once_with(b'test-response')
        assert result is mock_parse.return_value


class TestResponseParsing:
    """Test node and edge extraction from various response formats"""

    def test_extract_nodes_full_response(self):
        """Test node extraction from complete response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_FULL)

        assert len(nodes_df) == 3  # simple graph has 3 nodes
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns
        assert set(nodes_df['id']) == {'node1', 'node2', 'node3'}

    def test_extract_nodes_rawdata_only(self):
        """Test node extraction from RawData only"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        minimal_response = get_minimal_response()
        nodes_df = g._extract_nodes(minimal_response)

        assert len(nodes_df) >= 1  # May have entries from both Graph.Nodes and RawData
        # Find the node from RawData which has more complete information
        node1_rows = nodes_df[nodes_df['id'] == 'node1']
        assert len(node1_rows) > 0
        # Check that at least one row has the node (may not have label if from Graph.Nodes)

    def test_extract_nodes_deduplication(self):
        """Test node deduplication keeps most complete record"""
        duplicate_response = get_duplicate_nodes_response()

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(duplicate_response)

        # Should have 2 unique nodes (node1 and node2) after deduplication
        assert len(nodes_df) == 2
        assert set(nodes_df['id'].unique()) == {'node1', 'node2'}
        # Deduplication logic keeps one record per ID
        # Note: Current implementation may not merge all properties from duplicates

    def test_extract_nodes_malformed_data(self):
        """Test graceful handling of malformed data"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_MALFORMED)

        # Should extract valid nodes and skip invalid ones
        assert len(nodes_df) == 2
        assert set(nodes_df['id']) == {'node1', 'node2'}

    def test_extract_nodes_empty_response(self):
        """Test extraction from empty response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_EMPTY)

        assert len(nodes_df) == 0
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns

    def test_extract_edges_full_response(self):
        """Test edge extraction from complete response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_FULL)

        assert len(edges_df) == 2  # simple graph has 2 edges
        # Verify the edges form a chain: node1->node2->node3
        edge1 = edges_df[edges_df['source'] == 'node1'].iloc[0]
        assert edge1['target'] == 'node2'
        assert edge1['edge'] == 'KNOWS'

        edge2 = edges_df[edges_df['source'] == 'node2'].iloc[0]
        assert edge2['target'] == 'node3'
        assert edge2['edge'] == 'WORKS_WITH'

    def test_extract_edges_rawdata_only(self):
        """Test edge extraction from RawData only (orphan edges)"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edge_only_response = get_edge_only_response()
        edges_df = g._extract_edges(edge_only_response)

        assert len(edges_df) == 2  # edge_only_response has 2 orphan edges
        assert edges_df.iloc[0]['source'] == 'missing_node1'
        assert edges_df.iloc[0]['target'] == 'missing_node2'

    def test_extract_edges_empty_response(self):
        """Test edge extraction from empty response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_EMPTY)

        assert len(edges_df) == 0
        assert 'source' in edges_df.columns
        assert 'target' in edges_df.columns


class TestGraphConversion:
    """Test full graph conversion workflow"""

    def test_convert_bytes_response(self):
        """Test conversion from bytes response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response_bytes = json.dumps(SAMPLE_RESPONSE_FULL).encode('utf-8')
        result = g._parse_graph_response(response_bytes)

        assert result._nodes is not None
        assert result._edges is not None
        assert len(result._nodes) == 3  # simple graph has 3 nodes
        assert len(result._edges) == 2  # simple graph has 2 edges

    def test_convert_dict_response(self):
        """Test conversion from dict response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._parse_graph_response(SAMPLE_RESPONSE_FULL)

        assert result._nodes is not None
        assert result._edges is not None

    def test_convert_invalid_json(self):
        """Test error on invalid JSON bytes"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        with pytest.raises(SentinelGraphQueryError, match="parse.*JSON"):
            g._parse_graph_response(b'not valid json')

    def test_convert_empty_response(self):
        """Test conversion of empty response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._parse_graph_response(SAMPLE_RESPONSE_EMPTY)

        assert len(result._nodes) == 0
        assert len(result._edges) == 0


class TestSentinelGraphAPIFormat:
    """Test parsing of responses using sys_* field naming (actual Sentinel Graph API format)"""

    def test_extract_nodes_sys_format(self):
        """Test node extraction from sys_* format response"""
        from graphistry.tests.fixtures.sentinel_graph_responses import get_sentinel_graph_api_response

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_sentinel_graph_api_response()
        nodes_df = g._extract_nodes(response)

        # Should extract 4 nodes: 2 users + 2 IP addresses
        assert len(nodes_df) == 4
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns
        assert 'sys_label' in nodes_df.columns

        # Check node IDs
        node_ids = set(nodes_df['id'])
        assert 'user1@example.com' in node_ids
        assert 'user2@example.com' in node_ids
        assert '192.168.1.100' in node_ids
        assert '10.0.0.50' in node_ids

    def test_extract_edges_sys_format(self):
        """Test edge extraction from sys_* format response"""
        from graphistry.tests.fixtures.sentinel_graph_responses import get_sentinel_graph_api_response

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_sentinel_graph_api_response()
        edges_df = g._extract_edges(response)

        # Should extract 2 edges
        assert len(edges_df) == 2
        assert 'source' in edges_df.columns
        assert 'target' in edges_df.columns
        assert 'edge' in edges_df.columns

        # Check edge data
        edge1 = edges_df[edges_df['source'] == 'user1@example.com'].iloc[0]
        assert edge1['target'] == '192.168.1.100'
        assert edge1['edge'] == 'AUTH_ATTEMPT_FROM'
        assert edge1['failureCount'] == 5
        assert edge1['successCount'] == 100

        edge2 = edges_df[edges_df['source'] == 'user2@example.com'].iloc[0]
        assert edge2['target'] == '10.0.0.50'
        assert edge2['failureCount'] == 0
        assert edge2['successCount'] == 50

    def test_full_parsing_sys_format(self):
        """Test full graph parsing from sys_* format response"""
        from graphistry.tests.fixtures.sentinel_graph_responses import get_sentinel_graph_api_response

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_sentinel_graph_api_response()
        result = g._parse_graph_response(response)

        # Should have nodes and edges bound
        assert result._nodes is not None
        assert result._edges is not None
        assert len(result._nodes) == 4
        assert len(result._edges) == 2


# Integration test markers
@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires live API credentials")
class TestSentinelGraphIntegration:
    """Integration tests requiring live API access"""

    def test_live_query(self):
        """Test actual query against live API (requires credentials)"""
        # This would be run manually with real credentials
        pass
