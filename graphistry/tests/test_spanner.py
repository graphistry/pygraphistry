import os
import tempfile
import pytest
import pandas as pd
from unittest.mock import Mock, patch
import graphistry

SPANNER_PROJECT_ID = os.getenv("SPANNER_PROJECT_ID")
SPANNER_INSTANCE_ID = os.getenv("SPANNER_INSTANCE_ID")
SPANNER_DATABASE_ID = os.getenv("SPANNER_DATABASE_ID")
SPANNER_SERVICE_ACCOUNT_JSON = os.getenv("SPANNER_SERVICE_ACCOUNT_JSON")

HAS_SPANNER_CREDENTIALS = all([SPANNER_PROJECT_ID, SPANNER_INSTANCE_ID, SPANNER_DATABASE_ID, SPANNER_SERVICE_ACCOUNT_JSON])

SPANNER_SERVICE_ACCOUNT_JSON_PATH = "/dev/null"
if HAS_SPANNER_CREDENTIALS:
    tmp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    tmp_file.write(SPANNER_SERVICE_ACCOUNT_JSON)
    tmp_file.close()
    SPANNER_SERVICE_ACCOUNT_JSON_PATH = tmp_file.name

class TestSpannerIntegration:
    """Test suite for Google Cloud Spanner integration."""

    @pytest.mark.skipif(
        not HAS_SPANNER_CREDENTIALS,
        reason="Spanner credentials not configured"
    )
    def test_spanner_gql_basic(self):
        """Test basic Spanner GQL query execution."""
        # Configure with real credentials
        g_client = graphistry.client().configure_spanner(
            instance_id=SPANNER_INSTANCE_ID,
            database_id=SPANNER_DATABASE_ID,
            credentials_file=SPANNER_SERVICE_ACCOUNT_JSON_PATH
        )
        
        # Test simple SQL query  
        query = "SELECT 1 as test_col"
        df = g_client.spanner_gql_to_df(query)
        
        # Verify the dataframe was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'test_col' in df.columns
        assert df['test_col'].iloc[0] == 1

    @pytest.mark.skipif(
        not HAS_SPANNER_CREDENTIALS,
        reason="Spanner credentials not configured"
    )
    def test_spanner_gql_to_df(self):
        """Test Spanner GQL to DataFrame conversion."""
        g_client = graphistry.client().configure_spanner(
            instance_id=SPANNER_INSTANCE_ID,
            database_id=SPANNER_DATABASE_ID,
            credentials_file=SPANNER_SERVICE_ACCOUNT_JSON_PATH
        )
        
        # Test query with multiple columns
        query = "SELECT 'Alice' as name, 30 as age, 'Engineer' as role"
        df = g_client.spanner_gql_to_df(query)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'name' in df.columns
        assert 'age' in df.columns
        assert 'role' in df.columns
        assert df['name'].iloc[0] == 'Alice'
        assert df['age'].iloc[0] == 30
        assert df['role'].iloc[0] == 'Engineer'

    def test_spanner_from_client(self):
        """Test creating Spanner client from existing connection."""
        with patch('google.cloud.spanner_dbapi.connect') as mock_connect:
            mock_connection = Mock()
            mock_connect.return_value = mock_connection
            
            # Test from_client method
            g = graphistry.client().spanner_from_client(mock_connection)
            
            # Verify configuration was set
            assert g is not None

    @pytest.mark.skipif(
        not HAS_SPANNER_CREDENTIALS,
        reason="Spanner credentials not configured"
    )
    def test_spanner_gql_complex_path(self):
        """Test Spanner GQL with more complex graph query."""
        g_client = graphistry.client().configure_spanner(
            instance_id=SPANNER_INSTANCE_ID,
            database_id=SPANNER_DATABASE_ID,
            credentials_file=SPANNER_SERVICE_ACCOUNT_JSON_PATH
        )
        
        g = g_client.spanner_gql("""
        SELECT 
            'node1' as source,
            'node2' as destination,
            'edge_type1' as edge_type,
            1.0 as weight
        UNION ALL
        SELECT 
            'node2' as source,
            'node3' as destination,
            'edge_type2' as edge_type,
            2.0 as weight
        """)
        
        assert g._nodes is not None and g._nodes.empty is False, "Nodes are empty"
        assert g._edges is not None and g._edges.empty is False, "Edges are empty"

    def test_spanner_error_handling(self):
        """Test error handling for invalid configurations."""
        with pytest.raises(TypeError):
            graphistry.configure_spanner()  # type: ignore


class TestSpannerMocked:
    """Test Spanner functionality with mocked dependencies when credentials not available."""
    
    @patch('google.cloud.spanner_dbapi.connect')
    def test_spanner_client_creation_mocked(self, mock_connect):
        """Test Spanner client creation without real credentials."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_cursor.fetchall.return_value = [('test_value',)]
        mock_cursor.description = [('test_col',)]
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Configure Spanner
        g_client = graphistry.client().configure_spanner(
            project_id="test-project",
            instance_id="test-instance",
            database_id="test-database"
        )
        
        # Ensure the mock connection is used by setting it directly
        g_client.session.spanner._client = mock_connection
        
        df = g_client.spanner_gql_to_df("SELECT 'test' as test_col")
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'test_col' in df.columns
