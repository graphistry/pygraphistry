"""Tests that chain_remote/python_remote use instance sessions, not global PyGraphistry."""

import pytest
from unittest.mock import Mock, MagicMock, patch, PropertyMock
import pandas as pd

from graphistry.compute.chain_remote import chain_remote_generic
from graphistry.compute.python_remote import python_remote_generic


class TestChainRemoteAuth:

    def test_chain_remote_uses_instance_session_refresh(self):

        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "test_token_123"
        mock_plottable.session.certificate_validation = True
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_123"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()

        chain = {'chain': []}

        with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response

            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,
                output_type="shape"
            )

            mock_plottable._pygraphistry.refresh.assert_called_once()

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer test_token_123"

    def test_chain_remote_gets_token_from_session(self):

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
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_post.return_value = mock_response

            chain_remote_generic(
                mock_plottable,
                chain,
                api_token=None,
                output_type="shape"
            )

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer session_token_456"

    def test_chain_remote_with_provided_token(self):

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

            chain_remote_generic(
                mock_plottable,
                chain,
                api_token="explicit_token_789",
                output_type="shape"
            )

            mock_plottable._pygraphistry.refresh.assert_not_called()

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer explicit_token_789"

    def test_chain_remote_injects_traceparent(self):
        mock_plottable = Mock()
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "session_token_999"
        mock_plottable.session.certificate_validation = True
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_trace"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()

        chain = {'chain': []}
        traceparent = "00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01"

        with patch('graphistry.compute.chain_remote.inject_trace_headers') as mock_inject:
            mock_inject.side_effect = lambda headers: {**headers, "traceparent": traceparent}
            with patch('graphistry.compute.chain_remote.requests.post') as mock_post:
                mock_response = Mock()
                mock_response.raise_for_status = Mock()
                mock_response.text = '{"nodes": [], "edges": []}'
                mock_response.json = Mock(return_value={"nodes": [], "edges": []})
                mock_post.return_value = mock_response

                chain_remote_generic(
                    mock_plottable,
                    chain,
                    api_token=None,
                    output_type="shape"
                )

                headers = mock_post.call_args[1]["headers"]
                assert headers["traceparent"] == traceparent


class TestPythonRemoteAuth:

    def test_python_remote_uses_instance_session_refresh(self):

        from graphistry.Plottable import Plottable

        mock_plottable = Mock(spec=Plottable)
        mock_plottable.session = Mock()
        mock_plottable.session.api_token = "python_token_123"
        mock_plottable.session.certificate_validation = True
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        mock_plottable._nodes = None
        mock_plottable.edges = Mock(return_value=mock_plottable)
        mock_plottable.nodes = Mock(return_value=mock_plottable)

        code = "def task(g): return g"

        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_response.content = b'{"nodes": [], "edges": []}'
            mock_post.return_value = mock_response

            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )

            mock_plottable._pygraphistry.refresh.assert_called_once()

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer python_token_123"

    def test_python_remote_gets_token_from_session(self):

        from graphistry.Plottable import Plottable

        mock_plottable = Mock(spec=Plottable)
        mock_session = Mock()
        mock_session.api_token = "python_session_456"
        mock_session.certificate_validation = True
        mock_plottable.session = mock_session
        mock_plottable._pygraphistry = Mock()
        mock_plottable._dataset_id = "dataset_python2"
        mock_plottable.base_url_server = Mock(return_value="https://test.server")
        mock_plottable._edges = pd.DataFrame()
        mock_plottable._nodes = None
        mock_plottable.edges = Mock(return_value=mock_plottable)
        mock_plottable.nodes = Mock(return_value=mock_plottable)

        code = "def task(g): return g"

        with patch('graphistry.compute.python_remote.requests.post') as mock_post:
            mock_response = Mock()
            mock_response.raise_for_status = Mock()
            mock_response.text = '{"nodes": [], "edges": []}'
            mock_response.json = Mock(return_value={"nodes": [], "edges": []})
            mock_response.content = b'{"nodes": [], "edges": []}'
            mock_post.return_value = mock_response

            python_remote_generic(
                mock_plottable,
                code,
                api_token=None,
                format='json',
                output_type='json'
            )

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer python_session_456"


class TestClientIsolation:

    def test_two_clients_different_tokens_chain_remote(self):

        client1 = Mock()
        client1.session = Mock()
        client1.session.api_token = "client1_token"
        client1.session.certificate_validation = True
        client1._pygraphistry = Mock()
        client1._dataset_id = "dataset1"
        client1.base_url_server = Mock(return_value="https://test.server")
        client1._edges = pd.DataFrame()

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

            chain_remote_generic(
                client1,
                chain,
                api_token=None,
                output_type="shape"
            )

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer client1_token"

            chain_remote_generic(
                client2,
                chain,
                api_token=None,
                output_type="shape"
            )

            assert mock_post.call_args[1]['headers']['Authorization'] == "Bearer client2_token"

            client1._pygraphistry.refresh.assert_called_once()
            client2._pygraphistry.refresh.assert_called_once()

    def test_client_does_not_use_global_pygraphistry(self):

        import graphistry.compute.chain_remote as cr_module
        import graphistry.compute.python_remote as pr_module

        with open(cr_module.__file__, 'r') as f:
            chain_remote_source = f.read()
            assert "from graphistry.pygraphistry import PyGraphistry" not in chain_remote_source
            assert "self._pygraphistry.refresh()" in chain_remote_source
            assert "self.session.api_token" in chain_remote_source

        with open(pr_module.__file__, 'r') as f:
            python_remote_source = f.read()
            assert "from graphistry.pygraphistry import PyGraphistry" not in python_remote_source
            assert "self._pygraphistry.refresh()" in python_remote_source
            assert "self.session.api_token" in python_remote_source
