"""
Tests for certificate_validation field being respected in session-based clients.

These tests verify that chain_remote, python_remote, and other remote operations
properly use the session's certificate_validation setting instead of ignoring it.
"""

import pytest
from unittest.mock import Mock, patch, PropertyMock
import pandas as pd

from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.python_remote import python_remote_generic


class TestCertificateValidationInRemoteCalls:
    """Test that remote operations respect session's certificate_validation setting"""

    def test_chain_remote_respects_certificate_validation_true(self):
        """Verify chain_remote passes verify=True when certificate_validation=True"""
        
        # Create mock plottable with certificate_validation=True
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "test_token"
        mock_plottable.session.certificate_validation = True  # Should use SSL verification
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_123"
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
            
            # Call chain_remote without providing api_token
            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify that verify=True was passed to requests.post
            assert 'verify' in mock_post.call_args[1], "verify parameter should be passed to requests.post"
            assert mock_post.call_args[1]['verify'] is True, "verify should be True when certificate_validation=True"

    def test_chain_remote_respects_certificate_validation_false(self):
        """Verify chain_remote passes verify=False when certificate_validation=False"""
        
        # Create mock plottable with certificate_validation=False
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "test_token"
        mock_plottable.session.certificate_validation = False  # Should disable SSL verification
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
            
            # Call chain_remote
            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify that verify=False was passed to requests.post
            assert 'verify' in mock_post.call_args[1], "verify parameter should be passed to requests.post"
            assert mock_post.call_args[1]['verify'] is False, "verify should be False when certificate_validation=False"

    def test_python_remote_respects_certificate_validation_true(self):
        """Verify python_remote passes verify=True when certificate_validation=True"""
        
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "python_token"
        mock_plottable.session.certificate_validation = True  # Should use SSL verification
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        code = "def task(g): return g"
        
        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"result": "success"}'
            mock_response.json = Mock(return_value={"result": "success"})
            mock_response.content = b'{"result": "success"}'
            mock_post.return_value = mock_response
            
            # Call python_remote
            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )
            
            # Verify that verify=True was passed to requests.post
            assert 'verify' in mock_post.call_args[1], "verify parameter should be passed to requests.post"
            assert mock_post.call_args[1]['verify'] is True, "verify should be True when certificate_validation=True"

    def test_python_remote_respects_certificate_validation_false(self):
        """Verify python_remote passes verify=False when certificate_validation=False"""
        
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "python_token"
        mock_plottable.session.certificate_validation = False  # Should disable SSL verification
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python2"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        
        code = "def task(g): return g"
        
        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"result": "success"}'
            mock_response.json = Mock(return_value={"result": "success"})
            mock_response.content = b'{"result": "success"}'
            mock_post.return_value = mock_response
            
            # Call python_remote
            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )
            
            # Verify that verify=False was passed to requests.post
            assert 'verify' in mock_post.call_args[1], "verify parameter should be passed to requests.post"
            assert mock_post.call_args[1]['verify'] is False, "verify should be False when certificate_validation=False"


class TestMultipleClientsWithDifferentCertSettings:
    """Test that multiple clients can have different certificate_validation settings"""

    def test_two_clients_different_cert_validation_chain_remote(self):
        """Verify two clients with different certificate settings don't interfere"""
        
        # Create first client with certificate_validation=True
        client1 = Mock()
        client1.session = Mock()
        client1.session.api_token = "client1_token"
        client1.session.certificate_validation = True
        client1._pygraphistry = Mock()
        client1._dataset_id = "dataset1"
        client1.base_url_server = Mock(return_value="https://test.server")
        client1._edges = pd.DataFrame()
        
        # Create second client with certificate_validation=False
        client2 = Mock()
        client2.session = Mock()
        client2.session.api_token = "client2_token"
        client2.session.certificate_validation = False
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
            
            # Call chain_remote for client1 (should use verify=True)
            chain_remote_generic(
                client1,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify client1's certificate_validation was used
            assert 'verify' in mock_post.call_args[1]
            assert mock_post.call_args[1]['verify'] is True
            
            # Call chain_remote for client2 (should use verify=False)
            chain_remote_generic(
                client2,
                chain,
                api_token=None,
                output_type="shape"
            )
            
            # Verify client2's certificate_validation was used
            assert 'verify' in mock_post.call_args[1]
            assert mock_post.call_args[1]['verify'] is False

    def test_global_pygraphistry_vs_client_session_cert_validation(self):
        """Test that client sessions can override global PyGraphistry certificate_validation"""
        
        # Mock global PyGraphistry with certificate_validation=True
        with patch('graphistry.pygraphistry.PyGraphistry') as MockPyGraphistry:
            mock_global = MockPyGraphistry.return_value
            mock_global.session = Mock()
            mock_global.session.certificate_validation = True
            
            # Create client with different certificate_validation=False
            client = Mock()
            client.session = Mock()
            client.session.api_token = "client_token"
            client.session.certificate_validation = False  # Override global setting
            client._pygraphistry = Mock()
            client._dataset_id = "dataset_client"
            client.base_url_server = Mock(return_value="https://test.server")
            client._edges = pd.DataFrame()
            
            chain = {'chain': []}
            
            with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.text = '{"nodes": [], "edges": []}'
                mock_response.json = Mock(return_value={"nodes": [], "edges": []})
                mock_post.return_value = mock_response
                
                # Call with client (should use client's verify=False, not global's True)
                chain_remote_generic(
                    client,
                    chain,
                    api_token=None,
                    output_type="shape"
                )
                
                # Verify client's certificate_validation was used, not global
                assert 'verify' in mock_post.call_args[1]
                assert mock_post.call_args[1]['verify'] is False, "Should use client's certificate_validation, not global"
