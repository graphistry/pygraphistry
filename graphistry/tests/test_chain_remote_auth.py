"""
Tests for chain_remote and python_remote authentication to prevent regression.

These tests verify that chain_remote and python_remote use the instance's
session for authentication rather than the global PyGraphistry singleton.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd

from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.python_remote import python_remote_generic


class TestChainRemoteAuth:
    """Test that chain_remote uses instance session, not global PyGraphistry"""

    def test_chain_remote_uses_instance_session_refresh(self):
        """Verify chain_remote calls self._pygraphistry.refresh() not PyGraphistry.refresh()"""
        
        # Create mock plottable with session and _pygraphistry
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "test_token_123"
        mock_plottable.session.certificate_validation = True
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_123"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()  # Add empty DataFrame to satisfy type check
        
        # Mock the chain to pass validation
        chain = {'chain': []}
        
        with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
            # Setup mock response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response
            
            # Call chain_remote without providing api_token
            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,  # Force it to get token from session
                output_type="shape"
            )
            
            # Verify refresh was called on instance, not global
            mock_plottable._pygraphistry.refresh.assert_called_once()
            
            # Verify the token came from session
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer test_token_123"

    def test_chain_remote_gets_token_from_session(self):
        """Verify chain_remote accesses self.session.api_token"""
        
        # Create mock plottable
        mock_plottable = Mock()
        mock_session = Mock()
        mock_session.api_token = "session_token_456"
        mock_session.certificate_validation = True
        mock_plottable.session = mock_session
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_456"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        chain = {'chain': []}
        
        with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
            # Setup mock response
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response
            
            # Call without api_token to force session usage
            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify token was accessed from session
            # The token should be used in the Authorization header
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer session_token_456"

    def test_chain_remote_with_provided_token(self):
        """Verify chain_remote uses provided token over session token"""
        
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "session_token"
        mock_plottable.session.certificate_validation = True
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_789"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        chain = {'chain': []}
        
        with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response
            
            # Call with explicit api_token
            chain_remote_generic(
                mock_plottable,
                chain,
                api_token="explicit_token_789",
                output_type="shape"
            )
            
            # Should NOT call refresh when token is provided
            mock_plottable._pygraphistry.refresh.assert_not_called()
            
            # Should use the provided token
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer explicit_token_789"


class TestPythonRemoteAuth:
    """Test that python_remote uses instance session, not global PyGraphistry"""

    def test_python_remote_uses_instance_session_refresh(self):
        """Verify python_remote calls self._pygraphistry.refresh()"""
        
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "python_token_123"
        mock_plottable.session.certificate_validation = True  # Add certificate_validation
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        code = "def task(g): return g"
        
        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_response.content = b'{"nodes": [], "edges": []}'  # Add bytes content
            mock_post.return_value = mock_response
            
            # Call without api_token
            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )
            
            # Verify refresh was called
            mock_plottable._pygraphistry.refresh.assert_called_once()
            
            # Verify session token was used
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer python_token_123"

    def test_python_remote_gets_token_from_session(self):
        """Verify python_remote accesses self.session.api_token"""
        
        mock_plottable = Mock()
        mock_session = Mock()
        mock_session.api_token = "python_session_456"
        mock_session.certificate_validation = True  # Add certificate_validation
        mock_plottable.session = mock_session
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python2"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        code = "def task(g): return g"
        
        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_response.content = b'{"nodes": [], "edges": []}'  # Add bytes content
            mock_post.return_value = mock_response
            
            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )
            
            # Verify correct token was used
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer python_session_456"


class TestClientIsolation:
    """Test that multiple clients maintain separate authentication"""

    def test_two_clients_different_tokens_chain_remote(self):
        """Verify two clients with different tokens don't interfere in chain_remote"""
        
        # Create first client mock
        client1 = Mock()
        client1.session = Mock()
        client1.session.api_token = "client1_token"
        client1.session.certificate_validation = True
        client1._pygraphistry = Mock()
        client1._dataset_id = "dataset1"
        client1.base_url_server = Mock(return_value="https://test.server")
        client1._edges = pd.DataFrame()
        
        # Create second client mock
        client2 = Mock()
        client2.session = Mock()
        client2.session.api_token = "client2_token"
        client2.session.certificate_validation = True
        client2._pygraphistry = Mock()
        client2._dataset_id = "dataset2"
        client2.base_url_server = Mock(return_value="https://test.server")
        client2._edges = pd.DataFrame()
        
        chain = {'chain': []}
        
        with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response
            
            # Call chain_remote for client1
            chain_remote_generic(
                client1,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify client1's token was used
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer client1_token"
            
            # Call chain_remote for client2
            chain_remote_generic(
                client2,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify client2's token was used (not client1's)
            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer client2_token"
            
            # Verify each client's refresh was called
            client1._pygraphistry.refresh.assert_called_once()
            client2._pygraphistry.refresh.assert_called_once()

    def test_client_does_not_use_global_pygraphistry(self):
        """Verify that we don't import or use global PyGraphistry"""
        
        # This test verifies the fix by checking the actual code doesn't import PyGraphistry
        import graphistry.compute.chain_remote as cr_module
        import graphistry.compute.python_remote as pr_module
        
        # Check chain_remote.py source
        with open(cr_module.__file__, 'r') as f:
            chain_remote_source = f.read()
            # Should NOT contain the problematic import
            assert "from graphistry.pygraphistry import PyGraphistry" not in chain_remote_source
            # Should use instance's _pygraphistry
            assert "self._pygraphistry.refresh()" in chain_remote_source
            assert "self.session.api_token" in chain_remote_source
        
        # Check python_remote.py source
        with open(pr_module.__file__, 'r') as f:
            python_remote_source = f.read()
            # Should NOT contain the problematic import
            assert "from graphistry.pygraphistry import PyGraphistry" not in python_remote_source
            # Should use instance's _pygraphistry
            assert "self._pygraphistry.refresh()" in python_remote_source
            assert "self.session.api_token" in python_remote_source
