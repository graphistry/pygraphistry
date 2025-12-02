import unittest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd
from datetime import datetime, timedelta
from typing import List

from graphistry.plugins.sentinel import SentinelMixin
from graphistry.plugins_types.sentinel_types import (
    SentinelConfig,
    SentinelConnectionError,
    SentinelQueryError,
    SentinelQueryResult
)


class TestSentinelMixin(unittest.TestCase):
    """Test cases for SentinelMixin functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock Plotter instance with SentinelMixin
        from graphistry.plugins.sentinel import SentinelMixin

        class MockPlotter(SentinelMixin):
            def __init__(self):
                self.session = MagicMock()
                self.session.sentinel = None

        self.plotter = MockPlotter()
        self.workspace_id = "12345678-1234-1234-1234-123456789abc"

    def test_configure_sentinel_basic(self):
        """Test basic Sentinel configuration."""
        result = self.plotter.configure_sentinel(
            workspace_id=self.workspace_id
        )

        self.assertEqual(result, self.plotter)
        self.assertIsNotNone(self.plotter.session.sentinel)
        self.assertEqual(self.plotter.session.sentinel.workspace_id, self.workspace_id)
        self.assertEqual(self.plotter.session.sentinel.default_timespan, timedelta(hours=24))

    def test_configure_sentinel_service_principal(self):
        """Test Sentinel configuration with service principal."""
        result = self.plotter.configure_sentinel(
            workspace_id=self.workspace_id,
            tenant_id="tenant-123",
            client_id="client-456",
            client_secret="secret-789"
        )

        self.assertEqual(result, self.plotter)
        config = self.plotter.session.sentinel
        self.assertEqual(config.workspace_id, self.workspace_id)
        self.assertEqual(config.tenant_id, "tenant-123")
        self.assertEqual(config.client_id, "client-456")
        self.assertEqual(config.client_secret, "secret-789")

    def test_configure_sentinel_custom_timespan(self):
        """Test Sentinel configuration with custom default timespan."""
        custom_timespan = timedelta(days=7)
        self.plotter.configure_sentinel(
            workspace_id=self.workspace_id,
            default_timespan=custom_timespan
        )

        self.assertEqual(self.plotter.session.sentinel.default_timespan, custom_timespan)

    @patch('graphistry.plugins.sentinel.init_sentinel_client')
    def test_sentinel_client_lazy_initialization(self, mock_init):
        """Test that Sentinel client is lazily initialized."""
        mock_client = MagicMock()
        mock_init.return_value = mock_client

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        # Client should not be initialized yet
        mock_init.assert_not_called()

        # Access client property
        client = self.plotter.sentinel_client

        # Now client should be initialized
        mock_init.assert_called_once()
        self.assertEqual(client, mock_client)

        # Accessing again should not reinitialize
        client2 = self.plotter.sentinel_client
        mock_init.assert_called_once()
        self.assertEqual(client2, mock_client)

    @patch('graphistry.plugins.sentinel.LogsQueryClient')
    def test_sentinel_from_client(self, mock_client_class):
        """Test configuration from existing client."""
        existing_client = MagicMock()

        result = self.plotter.sentinel_from_client(
            client=existing_client,
            workspace_id=self.workspace_id
        )

        self.assertEqual(result, self.plotter)
        self.assertEqual(self.plotter.session.sentinel.workspace_id, self.workspace_id)
        self.assertEqual(self.plotter.session.sentinel._client, existing_client)

    def test_sentinel_close(self):
        """Test closing Sentinel connection."""
        self.plotter.configure_sentinel(workspace_id=self.workspace_id)
        self.plotter.session.sentinel._client = MagicMock()

        self.plotter.sentinel_close()

        self.assertIsNone(self.plotter.session.sentinel._client)

    @patch.object(SentinelMixin, '_sentinel_query')
    def test_kql_single_table(self, mock_query):
        """Test KQL query with single table result."""
        # Mock query result
        mock_result = SentinelQueryResult(
            data=[['value1', 'value2'], ['value3', 'value4']],
            column_names=['col1', 'col2'],
            column_types=['string', 'string']
        )
        mock_query.return_value = [mock_result]

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        query = "SecurityEvent | take 10"
        df = self.plotter.kql(query)

        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df.columns), ['col1', 'col2'])
        mock_query.assert_called_once()

    @patch.object(SentinelMixin, '_sentinel_query')
    def test_kql_multiple_tables(self, mock_query):
        """Test KQL query with multiple table results."""
        # Mock query results
        mock_results = [
            SentinelQueryResult(
                data=[['data1']],
                column_names=['col1'],
                column_types=['string']
            ),
            SentinelQueryResult(
                data=[['data2']],
                column_names=['col2'],
                column_types=['string']
            )
        ]
        mock_query.return_value = mock_results

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        query = "SecurityEvent | take 5; SecurityAlert | take 5"
        dfs = self.plotter.kql(query, single_table=False)

        self.assertIsInstance(dfs, list)
        self.assertEqual(len(dfs), 2)
        self.assertIsInstance(dfs[0], pd.DataFrame)
        self.assertIsInstance(dfs[1], pd.DataFrame)

    @patch.object(SentinelMixin, '_sentinel_query')
    def test_kql_with_timespan(self, mock_query):
        """Test KQL query with custom timespan."""
        mock_query.return_value = []

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        custom_timespan = timedelta(days=30)
        with self.assertRaises(ValueError):  # No results
            self.plotter.kql("test query", timespan=custom_timespan)

        mock_query.assert_called_with("test query", timespan=custom_timespan)

    @patch.object(SentinelMixin, 'kql')
    def test_kql_last(self, mock_kql):
        """Test kql_last convenience method."""
        mock_df = pd.DataFrame({'col': [1, 2, 3]})
        mock_kql.return_value = mock_df

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        result = self.plotter.kql_last("test query", hours=48)

        self.assertEqual(result, mock_df)
        mock_kql.assert_called_with("test query", timespan=timedelta(hours=48))

    @patch.object(SentinelMixin, 'kql')
    def test_sentinel_tables(self, mock_kql):
        """Test sentinel_tables method."""
        mock_df = pd.DataFrame({'TableName': ['Table1', 'Table2']})
        mock_kql.return_value = mock_df

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        result = self.plotter.sentinel_tables()

        self.assertEqual(result, mock_df)
        mock_kql.assert_called_with(
            "union withsource=TableName * | distinct TableName | sort by TableName asc",
            timespan=timedelta(minutes=5)
        )

    @patch.object(SentinelMixin, 'kql')
    def test_sentinel_schema(self, mock_kql):
        """Test sentinel_schema method."""
        mock_df = pd.DataFrame({
            'ColumnName': ['Col1', 'Col2'],
            'DataType': ['string', 'datetime']
        })
        mock_kql.return_value = mock_df

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        result = self.plotter.sentinel_schema("SecurityEvent")

        self.assertEqual(result, mock_df)
        mock_kql.assert_called_with(
            "SecurityEvent | getschema",
            timespan=timedelta(minutes=5)
        )

    @patch.object(SentinelMixin, '_sentinel_query')
    def test_sentinel_health_check_success(self, mock_query):
        """Test successful health check."""
        mock_query.return_value = [MagicMock()]

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        # Should not raise
        self.plotter.sentinel_health_check()

        mock_query.assert_called_with("Heartbeat | take 1", timespan=timedelta(hours=1))

    @patch.object(SentinelMixin, '_sentinel_query')
    def test_sentinel_health_check_failure(self, mock_query):
        """Test health check failure."""
        mock_query.side_effect = Exception("Connection failed")

        self.plotter.configure_sentinel(workspace_id=self.workspace_id)

        with self.assertRaises(SentinelConnectionError) as ctx:
            self.plotter.sentinel_health_check()

        self.assertIn("Health check failed", str(ctx.exception))


class TestSentinelUtils(unittest.TestCase):
    """Test cases for Sentinel utility functions."""

    def test_unwrap_nested_simple(self):
        """Test unwrapping simple nested data."""
        from graphistry.plugins.sentinel import _unwrap_nested

        result = SentinelQueryResult(
            data=[
                [{'key': 'value1', 'nested': {'inner': 'data1'}}],
                [{'key': 'value2', 'nested': {'inner': 'data2'}}]
            ],
            column_names=['data'],
            column_types=['object']
        )

        df = _unwrap_nested(result)

        self.assertIn('data.key', df.columns)
        self.assertIn('data.nested.inner', df.columns)
        self.assertEqual(len(df), 2)

    def test_unwrap_nested_json_string(self):
        """Test unwrapping JSON strings."""
        from graphistry.plugins.sentinel import _unwrap_nested

        result = SentinelQueryResult(
            data=[
                ['{"key": "value1", "number": 42}'],
                ['{"key": "value2", "number": 84}']
            ],
            column_names=['json_data'],
            column_types=['string']
        )

        df = _unwrap_nested(result)

        self.assertIn('json_data.key', df.columns)
        self.assertIn('json_data.number', df.columns)
        self.assertEqual(df['json_data.key'].iloc[0], 'value1')
        self.assertEqual(df['json_data.number'].iloc[0], 42)

    def test_should_unwrap_detection(self):
        """Test detection of nested data."""
        from graphistry.plugins.sentinel import _should_unwrap

        # Should unwrap - has object type
        result1 = SentinelQueryResult(
            data=[[{'nested': 'data'}]],
            column_names=['col'],
            column_types=['object']
        )
        self.assertTrue(_should_unwrap(result1))

        # Should unwrap - has dict data
        result2 = SentinelQueryResult(
            data=[[{'key': 'value'}]],
            column_names=['col'],
            column_types=['string']
        )
        self.assertTrue(_should_unwrap(result2))

        # Should not unwrap - simple data
        result3 = SentinelQueryResult(
            data=[['simple', 'text']],
            column_names=['col1', 'col2'],
            column_types=['string', 'string']
        )
        self.assertFalse(_should_unwrap(result3))


class TestSentinelAuthentication(unittest.TestCase):
    """Test cases for Sentinel authentication."""

    @patch('graphistry.plugins.sentinel.LogsQueryClient')
    @patch('graphistry.plugins.sentinel.DefaultAzureCredential')
    def test_init_default_credential(self, mock_credential_class, mock_client_class):
        """Test initialization with DefaultAzureCredential."""
        from graphistry.plugins.sentinel import init_sentinel_client

        mock_credential = MagicMock()
        mock_credential_class.return_value = mock_credential

        config = SentinelConfig(workspace_id="test-workspace")
        init_sentinel_client(config)

        mock_credential_class.assert_called_once()
        mock_client_class.assert_called_once_with(mock_credential)

    @patch('graphistry.plugins.sentinel.LogsQueryClient')
    @patch('graphistry.plugins.sentinel.ClientSecretCredential')
    def test_init_service_principal(self, mock_credential_class, mock_client_class):
        """Test initialization with service principal."""
        from graphistry.plugins.sentinel import init_sentinel_client

        mock_credential = MagicMock()
        mock_credential_class.return_value = mock_credential

        config = SentinelConfig(
            workspace_id="test-workspace",
            tenant_id="tenant",
            client_id="client",
            client_secret="secret"
        )
        init_sentinel_client(config)

        mock_credential_class.assert_called_once_with(
            tenant_id="tenant",
            client_id="client",
            client_secret="secret"
        )
        mock_client_class.assert_called_once_with(mock_credential)

    @patch('graphistry.plugins.sentinel.LogsQueryClient')
    def test_init_custom_credential(self, mock_client_class):
        """Test initialization with custom credential."""
        from graphistry.plugins.sentinel import init_sentinel_client

        custom_credential = MagicMock()
        config = SentinelConfig(
            workspace_id="test-workspace",
            credential=custom_credential
        )

        init_sentinel_client(config)

        mock_client_class.assert_called_once_with(custom_credential)


if __name__ == '__main__':
    unittest.main()
