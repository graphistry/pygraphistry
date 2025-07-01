import os
import pytest
import pandas as pd
from unittest.mock import Mock, patch, PropertyMock
import graphistry

# Check if azure module is available
try:
    import azure.kusto.data
    HAS_AZURE_MODULE = True
except ImportError:
    HAS_AZURE_MODULE = False


KUSTO_CLUSTER_URI = os.getenv("KUSTO_CLUSTER_URI")
KUSTO_DATABASE = os.getenv("KUSTO_DATABASE")
KUSTO_CLIENT_ID = os.getenv("KUSTO_CLIENT_ID")
KUSTO_CLIENT_SECRET = os.getenv("KUSTO_CLIENT_SECRET")
KUSTO_TENANT_ID = os.getenv("KUSTO_TENANT_ID")

HAS_KUSTO_CREDENTIALS = all([KUSTO_CLUSTER_URI, KUSTO_DATABASE, KUSTO_CLIENT_ID, KUSTO_CLIENT_SECRET, KUSTO_TENANT_ID])

# Test queries similar to kusto_example.txt
TEST_QUERIES = [
    "print hello='Hello World'",
    "print x=1, y=2",
    """
    datatable(user:string, metadata:dynamic)
    [
        "alice", dynamic({"age": 30, "city": "Delhi"}),
        "bob",   dynamic({"age": 25, "city": "Mumbai"}),
        "cara",  dynamic({}),                                
        "dave",  dynamic(null)                               
    ]
    """,
    """
    datatable(user:string, actions:dynamic)
    [
        "alice", dynamic([{"type": "click",  "ts": "2024-01-01"},
                          {"type": "scroll", "ts": "2024-01-02"}]),
        "bob",   dynamic([{"type": "click",  "ts": "2024-01-03"}]),
        "cara",  dynamic([]),                                  
        "dave",  dynamic(null)
    ]
    """,
]


class TestKustoIntegration:
    """Test Kusto integration with real queries using actual credentials when available."""

    @pytest.mark.skipif(not HAS_KUSTO_CREDENTIALS, reason="Kusto credentials not configured")
    def test_kql_basic(self):
        """Test basic KQL query execution."""
        # Configure with real credentials
        graphistry.configure_kusto(
            cluster=KUSTO_CLUSTER_URI,
            database=KUSTO_DATABASE,
            client_id=KUSTO_CLIENT_ID,
            client_secret=KUSTO_CLIENT_SECRET,
            tenant_id=KUSTO_TENANT_ID
        )
        
        # Test simple print query with single_table=True (default behavior)
        query = "print hello='Hello World'"
        df = graphistry.kql(query)
        
        # Verify a single dataframe was returned
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'hello' in df.columns
        assert df['hello'].iloc[0] == 'Hello World'
        
        # Test with single_table=False
        dfs = graphistry.kql(query, single_table=False)
        assert isinstance(dfs, list)
        assert len(dfs) == 1
        df = dfs[0]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert 'hello' in df.columns
        assert df['hello'].iloc[0] == 'Hello World'

    @pytest.mark.skipif(not HAS_KUSTO_CREDENTIALS,reason="Kusto credentials not configured")
    def test_kql_datatable_with_graph(self):
        """Test creating a graph from KQL datatable results."""
        # Configure with real credentials
        graphistry.configure_kusto(
            cluster=KUSTO_CLUSTER_URI,
            database=KUSTO_DATABASE,
            client_id=KUSTO_CLIENT_ID,
            client_secret=KUSTO_CLIENT_SECRET,
            tenant_id=KUSTO_TENANT_ID
        )
        
        # Create a simple graph datatable
        query = """
        datatable(source:string, destination:string, weight:int)
        [
            "A", "B", 1,
            "B", "C", 2,
            "C", "A", 3
        ]
        """
        
        # Test single_table behavior (default)
        df = graphistry.kql(query)
        assert isinstance(df, pd.DataFrame)
        
        # Create graph from the results
        g = graphistry.bind(source='source', destination='destination').edges(df)
        
        # Verify the graph was created
        assert g is not None
        assert g._source == 'source'
        assert g._destination == 'destination'
        assert len(df) == 3

    @pytest.mark.skipif(not HAS_AZURE_MODULE, reason="Azure module not installed")
    def test_kusto_from_client(self):
        """Test creating Kusto client from credentials."""
        with patch('azure.kusto.data.KustoClient') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            g = graphistry.kusto_from_client(mock_client, "test-database")
            
            assert g._kusto_config._client is not None, "Kusto client not set"

    def test_client_kusto_isolation(self):
        """Test that different graphistry clients have isolated Kusto configs."""
        client1 = graphistry.client()
        client2 = graphistry.client()
        
        client1.configure_kusto(
            cluster="https://cluster1.kusto.windows.net",
            database="db1",
            client_id="id1",
            client_secret="secret1",
            tenant_id="tenant1"
        )
        
        client2.configure_kusto(
            cluster="https://cluster2.kusto.windows.net",
            database="db2",
            client_id="id2",
            client_secret="secret2",
            tenant_id="tenant2"
        )
        
        # TODO: Verify isolation of configs


    @pytest.mark.skipif(
        not HAS_KUSTO_CREDENTIALS,
        reason="Kusto credentials not configured"
    )
    def test_kql_dynamic_columns(self):
        """Test KQL query with dynamic column handling."""
        # Configure with real credentials
        graphistry.configure_kusto(
            cluster=KUSTO_CLUSTER_URI,
            database=KUSTO_DATABASE,
            client_id=KUSTO_CLIENT_ID,
            client_secret=KUSTO_CLIENT_SECRET,
            tenant_id=KUSTO_TENANT_ID
        )
        
        # Test query with dynamic columns
        query = TEST_QUERIES[2]  # datatable with metadata
        
        df = graphistry.kql(query)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 4
        assert 'user' in df.columns
        
        # Dynamic columns should be expanded
        if 'metadata.age' in df.columns:
            assert df.loc[df['user'] == 'alice', 'metadata.age'].iloc[0] == 30
            assert df.loc[df['user'] == 'bob', 'metadata.age'].iloc[0] == 25

    @pytest.mark.skipif(
        not HAS_KUSTO_CREDENTIALS,
        reason="Kusto credentials not configured"
    )
    def test_kql_multiple_tables(self):
        """Test KQL query returning multiple tables always returns list."""
        # Configure with real credentials
        graphistry.configure_kusto(
            cluster=KUSTO_CLUSTER_URI,
            database=KUSTO_DATABASE,
            client_id=KUSTO_CLIENT_ID,
            client_secret=KUSTO_CLIENT_SECRET,
            tenant_id=KUSTO_TENANT_ID
        )
        
        # Query that returns two tables
        query = "print x=1, y=2; print a='foo', b='bar'"
        
        # Multiple tables with single_table=True returns first table with warning
        df = graphistry.kql(query)
        assert isinstance(df, pd.DataFrame)
        
        # Test with single_table=False to get all tables
        dfs = graphistry.kql(query, single_table=False)
        assert isinstance(dfs, list)
        assert len(dfs) == 2
        
        # First table
        df1 = dfs[0]
        assert isinstance(df1, pd.DataFrame)
        assert 'x' in df1.columns
        assert 'y' in df1.columns
        assert df1['x'].iloc[0] == 1
        assert df1['y'].iloc[0] == 2
        
        # Second table
        df2 = dfs[1]
        assert isinstance(df2, pd.DataFrame)
        assert 'a' in df2.columns
        assert 'b' in df2.columns
        assert df2['a'].iloc[0] == 'foo'
        assert df2['b'].iloc[0] == 'bar'

    def test_kusto_error_handling(self):
        """Test error handling for invalid configurations."""
        # Test missing required parameters
        with pytest.raises(TypeError):
            graphistry.configure_kusto()  # type: ignore



class TestKustoMocked:
    """Test Kusto functionality with mocked dependencies when credentials not available."""
    
    @pytest.mark.skipif(not HAS_AZURE_MODULE, reason="Azure module not installed")
    @patch('azure.kusto.data.KustoClient')
    def test_kusto_client_creation_mocked(self, mock_client_class):
        """Test Kusto client creation without real credentials."""
        mock_client = Mock()
        mock_response = Mock()
        
        # Create a mock result that has the proper structure
        mock_result = Mock()
        mock_result.rows = [[1], [2], [3]]  # Each row is a list
        mock_result.columns = [Mock(column_name='result', column_type='int')]
        
        # primary_results should be an iterable containing result objects
        mock_response.primary_results = [mock_result]
        
        mock_client_class.return_value = mock_client
        mock_client.execute.return_value = mock_response
        
        # Configure Kusto
        graphistry.configure_kusto(
            cluster="https://test.kusto.windows.net",
            database="test-database",
            client_id="test-id",
            client_secret="test-secret",
            tenant_id="test-tenant"
        )
        
        # Create a mock query execution
        with patch('graphistry.plugins.kusto.KustoMixin.kusto_client', new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client
            
            # Test single_table behavior (default)
            df = graphistry.kql("print 'test'")
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
            
            # Test with single_table=False
            dfs = graphistry.kql("print 'test'", single_table=False)
            assert isinstance(dfs, list)
            assert len(dfs) == 1
            df = dfs[0]
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 3
