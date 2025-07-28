"""Integration tests for remote graph functionality in chain_dag.

These tests require a real Graphistry server and authentication.
Enable with: TEST_REMOTE_INTEGRATION=1

Additional optional env vars:
- GRAPHISTRY_USERNAME: Username for authentication
- GRAPHISTRY_PASSWORD: Password for authentication  
- GRAPHISTRY_API_KEY: API key (alternative to username/password)
- GRAPHISTRY_SERVER: Server URL (defaults to hub.graphistry.com)
- GRAPHISTRY_TEST_DATASET_ID: Known dataset ID to test with
"""

import os
import pytest
import pandas as pd
from unittest.mock import patch

from graphistry import PyGraphistry
from graphistry.compute.ast import ASTLet, ASTRemoteGraph, ASTRef, n
from graphistry.tests.test_compute import CGFull


# Check if remote integration tests are enabled
REMOTE_INTEGRATION_ENABLED = os.environ.get("TEST_REMOTE_INTEGRATION") == "1"
skip_remote = pytest.mark.skipif(
    not REMOTE_INTEGRATION_ENABLED,
    reason="Remote integration tests need TEST_REMOTE_INTEGRATION=1"
)


@skip_remote
class TestRemoteGraphIntegration:
    """Integration tests that connect to a real Graphistry server."""
    
    @classmethod
    def setup_class(cls):
        """Set up authentication for remote tests."""
        # Configure PyGraphistry with env vars if available
        server = os.environ.get("GRAPHISTRY_SERVER", "hub.graphistry.com")
        protocol = "https" if "443" in server or "https" in server else "http"
        
        if os.environ.get("GRAPHISTRY_API_KEY"):
            PyGraphistry.register(
                api=3,
                protocol=protocol,
                server=server,
                api_key=os.environ["GRAPHISTRY_API_KEY"]
            )
        elif os.environ.get("GRAPHISTRY_USERNAME") and os.environ.get("GRAPHISTRY_PASSWORD"):
            PyGraphistry.register(
                api=3,
                protocol=protocol,
                server=server,
                username=os.environ["GRAPHISTRY_USERNAME"],
                password=os.environ["GRAPHISTRY_PASSWORD"]
            )
        else:
            pytest.skip("Need GRAPHISTRY_API_KEY or GRAPHISTRY_USERNAME/PASSWORD for remote tests")
    
    def test_remote_graph_fetch_real_dataset(self):
        """Test fetching a real dataset from Graphistry server."""
        # First, upload a test dataset to get a real dataset_id
        test_edges = pd.DataFrame({
            'src': ['a', 'b', 'c'],
            'dst': ['b', 'c', 'a'],
            'weight': [1.0, 2.0, 3.0]
        })
        test_nodes = pd.DataFrame({
            'id': ['a', 'b', 'c'],
            'label': ['Node A', 'Node B', 'Node C']
        })
        
        g = CGFull().edges(test_edges, 'src', 'dst').nodes(test_nodes, 'id')
        uploaded = g.upload()
        dataset_id = uploaded._dataset_id
        assert dataset_id is not None
        
        # Now test fetching it via ASTRemoteGraph
        dag = ASTLet({
            'remote_data': ASTRemoteGraph(dataset_id)
        })
        
        # CGFull() creates empty graph, need one with edges for materialize_nodes
        g_base = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        g2 = g_base.gfql(dag)
        
        # Verify we got the data back
        assert len(g2._edges) == 3
        assert len(g2._nodes) == 3
        assert set(g2._edges['src'].values) == {'a', 'b', 'c'}
        assert set(g2._nodes['label'].values) == {'Node A', 'Node B', 'Node C'}
    
    def test_remote_graph_with_token(self):
        """Test using explicit token with RemoteGraph."""
        # Get current token
        PyGraphistry.refresh()
        token = PyGraphistry.api_token()
        
        if not token:
            pytest.skip("No API token available")
        
        # Upload test data
        g = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        uploaded = g.upload()
        dataset_id = uploaded._dataset_id
        
        # Fetch with explicit token
        dag = ASTLet({
            'data': ASTRemoteGraph(dataset_id, token=token)
        })
        
        # Need graph with edges for materialize_nodes
        g_base = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        result = g_base.gfql(dag)
        assert len(result._edges) == 1
    
    def test_remote_graph_in_complex_dag(self):
        """Test RemoteGraph as part of a complex DAG."""
        # Upload test dataset
        edges_df = pd.DataFrame({
            'src': ['a', 'b', 'c', 'd'],
            'dst': ['b', 'c', 'd', 'a'],
            'type': ['friend', 'friend', 'enemy', 'enemy']
        })
        nodes_df = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd'],
            'category': ['person', 'person', 'bot', 'bot']
        })
        
        g = CGFull().edges(edges_df, 'src', 'dst').nodes(nodes_df, 'id')
        uploaded = g.upload()
        dataset_id = uploaded._dataset_id
        
        # Create complex DAG with remote data
        dag = ASTLet({
            'remote': ASTRemoteGraph(dataset_id),
            'persons': ASTRef('remote', [n({'category': 'person'})]),
            'friends': ASTRef('persons', [n(edge_query="type == 'friend'")])
        })
        
        # Execute and verify - need graph with edges for materialize_nodes
        g_base = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        result = g_base.gfql(dag, output='friends')
        
        # Should only have person nodes
        assert all(result._nodes['category'] == 'person')
        # Should only have friend edges between persons
        assert len(result._edges) > 0
    
    def test_remote_graph_error_handling(self):
        """Test error handling for invalid dataset IDs."""
        dag = ASTLet({
            'bad_remote': ASTRemoteGraph('invalid-dataset-id-12345')
        })
        
        # Need graph with edges for materialize_nodes
        g = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        with pytest.raises(Exception) as exc_info:
            g.gfql(dag)
        
        # Should get some kind of HTTP error or validation error
        assert 'dataset' in str(exc_info.value).lower() or 'not found' in str(exc_info.value).lower()
    
    @pytest.mark.skipif(
        not os.environ.get("GRAPHISTRY_TEST_DATASET_ID"),
        reason="Need GRAPHISTRY_TEST_DATASET_ID env var for this test"
    )
    def test_remote_graph_known_dataset(self):
        """Test with a known dataset ID from env var."""
        dataset_id = os.environ["GRAPHISTRY_TEST_DATASET_ID"]
        
        dag = ASTLet({
            'data': ASTRemoteGraph(dataset_id)
        })
        
        # Need graph with edges for materialize_nodes
        g_base = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        result = g_base.gfql(dag)
        
        # Basic validation - should have some data
        assert result._edges is not None or result._nodes is not None
        if result._edges is not None:
            print(f"Fetched {len(result._edges)} edges from {dataset_id}")
        if result._nodes is not None:
            print(f"Fetched {len(result._nodes)} nodes from {dataset_id}")


class TestRemoteGraphMocked:
    """Tests with mocked remote calls (always run)."""
    
    @patch('graphistry.compute.chain_remote.chain_remote')
    def test_remote_graph_execution_mocked(self, mock_chain_remote):
        """Test that RemoteGraph calls chain_remote correctly."""
        # This test always runs, even without remote server
        mock_result = CGFull().edges(pd.DataFrame({'s': ['x'], 'd': ['y']}), 's', 'd')
        mock_chain_remote.return_value = mock_result
        
        dag = ASTLet({
            'remote': ASTRemoteGraph('test-dataset-123', token='test-token')
        })
        
        # Mock result should be used, but we still need edges for materialize_nodes
        g_base = CGFull().edges(pd.DataFrame({'s': ['a'], 'd': ['b']}), 's', 'd')
        result = g_base.gfql(dag)
        assert result is not None  # Verify result was returned
        
        # Verify chain_remote was called correctly
        mock_chain_remote.assert_called_once()
        call_args = mock_chain_remote.call_args
        assert call_args[1]['dataset_id'] == 'test-dataset-123'
        assert call_args[1]['api_token'] == 'test-token'
