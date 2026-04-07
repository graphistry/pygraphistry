import pytest
import json
from unittest.mock import Mock, patch
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
    get_response_with_null_properties,
    get_graph_list_response,
    get_table_format_response,
)


SAMPLE_RESPONSE_FULL = get_simple_graph_response()   # 3 nodes (node-a, node-b, node-c), 2 edges
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
        assert g.session.sentinel_graph.response_formats == ["Graph"]
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

    def test_configure_with_custom_response_formats(self):
        """Test configuration with custom response_formats"""
        g = graphistry.bind()
        g.configure_sentinel_graph(
            graph_instance="TestInstance",
            response_formats=["Table", "Graph"]
        )
        assert g.session.sentinel_graph.response_formats == ["Table", "Graph"]

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
        from graphistry.plotter import Plotter
        from graphistry.pygraphistry import PyGraphistry
        g = Plotter(pygraphistry=PyGraphistry)
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

        future_time = (datetime.now() + timedelta(hours=1)).timestamp()
        g.session.sentinel_graph._token = "cached-token"
        g.session.sentinel_graph._token_expiry = future_time

        with patch('azure.identity.InteractiveBrowserCredential') as mock_cred:
            token = g._get_auth_token()

            assert token == "cached-token"
            mock_cred.assert_not_called()

    def test_token_refresh_when_expired(self):
        """Test that expired tokens trigger refresh"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

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

        result = g._sentinel_graph_query("MATCH (n) RETURN n", "GQL", ["Graph"])

        assert result == mock_response.content
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs['json']['query'] == "MATCH (n) RETURN n"
        assert call_kwargs['json']['queryLanguage'] == "GQL"
        assert call_kwargs['json']['responseFormats'] == ["Graph"]
        assert call_kwargs['headers']['Authorization'] == "Bearer test-token"
        assert call_kwargs['timeout'] == 60

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_http_error(self, mock_auth, mock_post):
        """Test query execution with HTTP error"""
        mock_auth.return_value = "test-token"

        mock_response = Mock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        with pytest.raises(SentinelGraphQueryError, match="400"):
            g._sentinel_graph_query("INVALID QUERY", "GQL", ["Graph"])

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch('time.sleep')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_execute_query_retry_on_timeout(self, mock_auth, mock_sleep, mock_post):
        """Test retry logic on timeout"""
        mock_auth.return_value = "test-token"

        mock_post.side_effect = [
            requests.exceptions.Timeout("Timeout 1"),
            requests.exceptions.Timeout("Timeout 2"),
            Mock(status_code=200, content=json.dumps(SAMPLE_RESPONSE_FULL).encode())
        ]

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance", max_retries=3)

        result = g._sentinel_graph_query("MATCH (n) RETURN n", "GQL", ["Graph"])

        assert mock_post.call_count == 3
        assert mock_sleep.call_count == 2

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
            g._sentinel_graph_query("MATCH (n) RETURN n", "GQL", ["Graph"])

        assert mock_post.call_count == 3

    @patch.object(SentinelGraphMixin, '_sentinel_graph_query')
    @patch.object(SentinelGraphMixin, '_parse_graph_response')
    def test_sentinel_graph_main_method(self, mock_parse, mock_query):
        """Test main sentinel_graph method threads response_formats"""
        mock_query.return_value = b'test-response'
        mock_parse.return_value = Mock()

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g.sentinel_graph("MATCH (n) RETURN n")

        mock_query.assert_called_once_with("MATCH (n) RETURN n", 'GQL', ["Graph"])
        mock_parse.assert_called_once_with(b'test-response')
        assert result is mock_parse.return_value


class TestResponseFormats:
    """Test response_formats parameter threading"""

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_default_format_is_graph(self, mock_auth, mock_post):
        """Default responseFormats should be ["Graph"]"""
        mock_auth.return_value = "test-token"
        mock_post.return_value = Mock(
            status_code=200,
            content=json.dumps(SAMPLE_RESPONSE_EMPTY).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")
        g.sentinel_graph("MATCH (n) RETURN n")
        payload = mock_post.call_args[1]['json']
        assert payload['responseFormats'] == ["Graph"]

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_custom_format_passed_through(self, mock_auth, mock_post):
        """Custom response_formats should be sent to the API"""
        mock_auth.return_value = "test-token"
        mock_post.return_value = Mock(
            status_code=200,
            content=json.dumps(SAMPLE_RESPONSE_EMPTY).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")
        g.sentinel_graph("MATCH (n) RETURN n", response_formats=["Table", "Graph"])
        payload = mock_post.call_args[1]['json']
        assert payload['responseFormats'] == ["Table", "Graph"]

    @patch('graphistry.plugins.sentinel_graph.requests.post')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_format_configured_at_configure_time(self, mock_auth, mock_post):
        """response_formats set during configure_sentinel_graph should be used"""
        mock_auth.return_value = "test-token"
        mock_post.return_value = Mock(
            status_code=200,
            content=json.dumps(SAMPLE_RESPONSE_EMPTY).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance", response_formats=["Table"])
        g.sentinel_graph("MATCH (n) RETURN n")
        payload = mock_post.call_args[1]['json']
        assert payload['responseFormats'] == ["Table"]


class TestResponseParsing:
    """Test node and edge extraction from various response formats"""

    def test_extract_nodes_full_response(self):
        """Test node extraction from complete response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_FULL)

        assert len(nodes_df) == 3
        assert 'id' in nodes_df.columns
        assert 'label' in nodes_df.columns
        assert set(nodes_df['id']) == {'node-a', 'node-b', 'node-c'}

    def test_extract_nodes_labels_mapped(self):
        """Test that labels list is mapped to label column"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_FULL)
        node_a = nodes_df[nodes_df['id'] == 'node-a'].iloc[0]
        assert node_a['label'] == 'User'
        assert node_a['labels'] == ['User']

    def test_extract_nodes_properties_spread(self):
        """Test that node properties are spread as top-level columns"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_FULL)
        node_a = nodes_df[nodes_df['id'] == 'node-a'].iloc[0]
        assert node_a['name'] == 'Alice'
        assert node_a['department'] == 'Engineering'

    def test_extract_nodes_deduplication(self):
        """Test node deduplication keeps most complete record"""
        duplicate_response = get_duplicate_nodes_response()

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(duplicate_response)

        # 3 entries in fixture (node-dup x2, node-other x1) -> 2 unique IDs after dedup
        assert len(nodes_df) == 2
        assert set(nodes_df['id'].unique()) == {'node-dup', 'node-other'}
        # The richer record (with email + department) should be kept
        dup_row = nodes_df[nodes_df['id'] == 'node-dup'].iloc[0]
        assert dup_row.get('email') == 'bob@contoso.com'

    def test_extract_nodes_malformed_skips_missing_id(self):
        """Node entry missing 'id' should be skipped"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(SAMPLE_RESPONSE_MALFORMED)

        # Only 'node-valid' should be present; the entry without 'id' is skipped
        assert len(nodes_df) == 1
        assert nodes_df.iloc[0]['id'] == 'node-valid'

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

        assert len(edges_df) == 2
        edge_ab = edges_df[edges_df['source'] == 'node-a'].iloc[0]
        assert edge_ab['target'] == 'node-b'
        assert edge_ab['edge'] == 'MemberOf'

        edge_bc = edges_df[edges_df['source'] == 'node-b'].iloc[0]
        assert edge_bc['target'] == 'node-c'
        assert edge_bc['edge'] == 'HasAccess'

    def test_extract_edges_properties_spread(self):
        """Test that edge properties are spread as top-level columns"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_FULL)
        edge_ab = edges_df[edges_df['source'] == 'node-a'].iloc[0]
        assert edge_ab['since'] == '2024-01-01'

    def test_extract_edges_only_response(self):
        """Test edge extraction when no nodes are present"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(get_edge_only_response())

        assert len(edges_df) == 1
        assert edges_df.iloc[0]['source'] == 'ghost-node-a'
        assert edges_df.iloc[0]['target'] == 'ghost-node-b'

    def test_extract_edges_empty_response(self):
        """Test edge extraction from empty response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        edges_df = g._extract_edges(SAMPLE_RESPONSE_EMPTY)

        assert len(edges_df) == 0
        assert 'source' in edges_df.columns
        assert 'target' in edges_df.columns

    def test_extract_nodes_minimal(self):
        """Test minimal response with 1 node"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(get_minimal_response())

        assert len(nodes_df) == 1
        assert nodes_df.iloc[0]['id'] == 'node-001'
        assert nodes_df.iloc[0]['label'] == 'Device'

    def test_null_properties_preserved(self):
        """None values in properties are passed through"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        nodes_df = g._extract_nodes(get_response_with_null_properties())

        assert len(nodes_df) == 1
        node = nodes_df.iloc[0]
        assert node['name'] == 'Eve'
        assert node['role'] == 'analyst'


class TestTableFormatParsing:
    """Test rawData.tables secondary path (table format responses)"""

    def test_extract_nodes_from_table_format(self):
        """Nodes should be extracted from rawData.tables when graph section is empty"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_table_format_response()
        nodes_df = g._extract_nodes(response)

        assert len(nodes_df) == 2
        assert set(nodes_df['id']) == {'table-node-001', 'table-node-002'}

    def test_extract_edges_from_table_format(self):
        """Edges should be extracted from rawData.tables using sourceOid/targetOid"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_table_format_response()
        edges_df = g._extract_edges(response)

        assert len(edges_df) == 1
        assert edges_df.iloc[0]['source'] == 'table-node-001'
        assert edges_df.iloc[0]['target'] == 'table-node-002'
        assert edges_df.iloc[0]['edge'] == 'HasRole'

    def test_table_format_node_labels_mapped(self):
        """Table format nodes should have label mapped from labels[0]"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        response = get_table_format_response()
        nodes_df = g._extract_nodes(response)

        node_001 = nodes_df[nodes_df['id'] == 'table-node-001'].iloc[0]
        assert node_001['label'] == 'User'


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
        assert len(result._nodes) == 3
        assert len(result._edges) == 2

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

    def test_convert_missing_result_key(self):
        """Old response format without 'result' key raises clear error"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        old_format = {"Graph": {"Nodes": [], "Edges": []}, "RawData": {"Rows": []}}
        with pytest.raises(SentinelGraphQueryError, match="result"):
            g._parse_graph_response(old_format)

    def test_convert_empty_response(self):
        """Test conversion of empty response"""
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="TestInstance")

        result = g._parse_graph_response(SAMPLE_RESPONSE_EMPTY)

        assert len(result._nodes) == 0
        assert len(result._edges) == 0


class TestSentinelGraphList:
    """Tests for sentinel_graph_list() method"""

    @patch('graphistry.plugins.sentinel_graph.requests.get')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_list_returns_dataframe(self, mock_auth, mock_get):
        """sentinel_graph_list returns a DataFrame with graph instance metadata"""
        mock_auth.return_value = "test-token"
        mock_get.return_value = Mock(
            status_code=200,
            content=json.dumps(get_graph_list_response()).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="placeholder")
        result = g.sentinel_graph_list()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert "name" in result.columns
        assert "instanceStatus" in result.columns
        assert result.iloc[0]["name"] == "TestGraph"
        assert result.iloc[0]["instanceStatus"] == "Ready"
        assert result.iloc[1]["name"] == "StagingGraph"
        assert result.iloc[1]["instanceStatus"] == "Creating"

    @patch('graphistry.plugins.sentinel_graph.requests.get')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_list_uses_correct_url_and_params(self, mock_auth, mock_get):
        """List endpoint uses correct URL and graphTypes=Custom query param"""
        mock_auth.return_value = "test-token"
        mock_get.return_value = Mock(
            status_code=200,
            content=json.dumps({"value": []}).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="placeholder")
        g.sentinel_graph_list()

        call_args = mock_get.call_args
        url = call_args[0][0]
        params = call_args[1]['params']
        assert "graph-instances" in url
        assert "api.securityplatform.microsoft.com" in url
        assert params == {"graphTypes": "Custom"}

    @patch('graphistry.plugins.sentinel_graph.requests.get')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_list_empty_returns_empty_dataframe(self, mock_auth, mock_get):
        """Empty list returns DataFrame with expected columns"""
        mock_auth.return_value = "test-token"
        mock_get.return_value = Mock(
            status_code=200,
            content=json.dumps({"value": []}).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="placeholder")
        result = g.sentinel_graph_list()

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0
        assert set(result.columns) >= {"name", "graphDefinitionName", "instanceStatus"}

    @patch('graphistry.plugins.sentinel_graph.requests.get')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_list_http_error_raises(self, mock_auth, mock_get):
        """Non-200 HTTP response raises SentinelGraphQueryError"""
        mock_auth.return_value = "test-token"
        mock_get.return_value = Mock(status_code=403)

        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="placeholder")

        with pytest.raises(SentinelGraphQueryError, match="403"):
            g.sentinel_graph_list()

    @patch('graphistry.plugins.sentinel_graph.requests.get')
    @patch.object(SentinelGraphMixin, '_get_auth_token')
    def test_list_uses_bearer_token(self, mock_auth, mock_get):
        """List endpoint sends correct Authorization header"""
        mock_auth.return_value = "my-bearer-token"
        mock_get.return_value = Mock(
            status_code=200,
            content=json.dumps({"value": []}).encode()
        )
        g = graphistry.bind()
        g.configure_sentinel_graph(graph_instance="placeholder")
        g.sentinel_graph_list()

        call_kwargs = mock_get.call_args[1]
        assert call_kwargs['headers']['Authorization'] == "Bearer my-bearer-token"


# Integration test markers
@pytest.mark.integration
@pytest.mark.skipif(True, reason="Requires live API credentials")
class TestSentinelGraphIntegration:
    """Integration tests requiring live API access"""

    def test_live_query(self):
        """Test actual query against live API (requires credentials)"""
        pass
