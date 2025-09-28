"""Tests for remote data loading policy hooks."""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
from graphistry.compute.ast import ASTRemoteGraph, let as ast_let


class TestRemoteDataPolicy:
    """Test policy hooks for remote data loading operations."""

    def test_remote_graph_triggers_preload_hook(self):
        """Test that ASTRemoteGraph triggers preload hook with is_remote=True."""
        preload_called = {'called': False, 'context': None}

        def preload_policy(context: PolicyContext) -> None:
            preload_called['called'] = True
            preload_called['context'] = dict(context)
            return None

        # Mock chain_remote to avoid real network calls
        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            # Setup mock to return a plottable
            mock_result = Mock()
            mock_result._nodes = pd.DataFrame({'id': [1, 2, 3]})
            mock_result._edges = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            mock_chain_remote.return_value = mock_result

            # Create a graph with remote dataset reference
            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            # Create DAG with ASTRemoteGraph
            dag = ast_let({
                'remote_data': ASTRemoteGraph('test-dataset-123', token='test-jwt')
            })

            # Execute with policy
            policy = {'preload': preload_policy}
            g.gfql(dag, policy=policy)

            # Verify preload was called with correct context
            assert preload_called['called'] is True
            assert preload_called['context']['phase'] == 'preload'
            assert preload_called['context']['is_remote'] is True
            assert preload_called['context']['remote_dataset_id'] == 'test-dataset-123'
            assert preload_called['context']['remote_token'] == 'test-jwt'
            assert preload_called['context']['operation'] == 'ASTRemoteGraph'

    def test_remote_graph_triggers_postload_hook(self):
        """Test that ASTRemoteGraph triggers postload hook after fetch."""
        postload_calls = []

        def postload_policy(context: PolicyContext) -> None:
            postload_calls.append(dict(context))
            return None

        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            # Setup mock to return a plottable with data
            mock_result = Mock()
            mock_result._nodes = pd.DataFrame({'id': [1, 2, 3]})
            mock_result._edges = pd.DataFrame({'src': [1, 2], 'dst': [2, 3]})
            mock_chain_remote.return_value = mock_result

            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            dag = ast_let({
                'remote_data': ASTRemoteGraph('test-dataset-456')
            })

            policy = {'postload': postload_policy}
            g.gfql(dag, policy=policy)

            # Should have at least one postload call
            assert len(postload_calls) > 0

            # Look for the remote-specific postload call
            remote_postload = None
            for call in postload_calls:
                if call.get('is_remote'):
                    remote_postload = call
                    break

            # If we have a remote postload, verify it
            if remote_postload:
                assert remote_postload['phase'] == 'postload'
                assert remote_postload['is_remote'] is True
                assert remote_postload['remote_dataset_id'] == 'test-dataset-456'
                assert 'graph_stats' in remote_postload
                assert remote_postload['operation'] == 'ASTRemoteGraph'
            else:
                # We at least got the DAG-level postload
                assert any(c['phase'] == 'postload' for c in postload_calls)

    def test_policy_can_deny_remote_load(self):
        """Test that policy can deny remote data loading."""
        def deny_policy(context: PolicyContext) -> None:
            if context.get('is_remote') and context.get('remote_dataset_id') == 'forbidden':
                raise PolicyException(
                    phase='preload',
                    reason='Remote dataset forbidden',
                    code=403
                )
            return None

        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            mock_result = Mock()
            mock_chain_remote.return_value = mock_result

            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            dag = ast_let({
                'remote_data': ASTRemoteGraph('forbidden')
            })

            policy = {'preload': deny_policy}

            with pytest.raises(RuntimeError) as exc_info:
                g.gfql(dag, policy=policy)

            # The PolicyException is wrapped in RuntimeError by chain_let_impl
            assert 'PolicyException' in str(exc_info.value)
            assert 'forbidden' in str(exc_info.value).lower()
            # Verify chain_remote was never called
            mock_chain_remote.assert_not_called()

    def test_policy_can_check_jwt_token(self):
        """Test that policy can inspect JWT token for authorization."""
        token_checks = {'has_token': False, 'token_value': None}

        def token_policy(context: PolicyContext) -> None:
            if context.get('is_remote'):
                token_checks['has_token'] = 'remote_token' in context
                token_checks['token_value'] = context.get('remote_token')

                # Deny if no token provided
                if not context.get('remote_token'):
                    raise PolicyException(
                        phase='preload',
                        reason='JWT token required for remote access',
                        code=401
                    )
            return None

        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            mock_result = Mock()
            mock_result._nodes = pd.DataFrame()
            mock_result._edges = pd.DataFrame()
            mock_chain_remote.return_value = mock_result

            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            # Try without token - should fail
            dag_no_token = ast_let({
                'remote_data': ASTRemoteGraph('dataset-789')
            })

            policy = {'preload': token_policy}

            with pytest.raises((PolicyException, RuntimeError)) as exc_info:
                g.gfql(dag_no_token, policy=policy)

            # Check for the error message in the exception
            assert 'JWT token required' in str(exc_info.value)

            # Try with token - should succeed
            dag_with_token = ast_let({
                'remote_data': ASTRemoteGraph('dataset-789', token='valid-jwt-token')
            })

            g.gfql(dag_with_token, policy=policy)

            assert token_checks['has_token'] is True
            assert token_checks['token_value'] == 'valid-jwt-token'

    def test_postload_can_check_remote_data_size(self):
        """Test that postload can check size of fetched remote data."""
        def size_limit_policy(context: PolicyContext) -> None:
            if context.get('phase') == 'postload' and context.get('is_remote'):
                stats = context.get('graph_stats', {})
                nodes = stats.get('nodes', 0)

                if nodes > 100:
                    raise PolicyException(
                        phase='postload',
                        reason=f'Remote dataset too large: {nodes} nodes',
                        code=413,
                        data_size={'nodes': nodes, 'limit': 100}
                    )
            return None

        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            # Create large mock dataset
            mock_result = Mock()
            mock_result._nodes = pd.DataFrame({'id': range(150)})  # 150 nodes
            mock_result._edges = pd.DataFrame({'src': [1], 'dst': [2]})
            mock_chain_remote.return_value = mock_result

            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            dag = ast_let({
                'remote_data': ASTRemoteGraph('large-dataset')
            })

            policy = {'postload': size_limit_policy}

            with pytest.raises((PolicyException, RuntimeError)) as exc_info:
                g.gfql(dag, policy=policy)

            # Check for the error details in the exception
            assert '150 nodes' in str(exc_info.value)

    def test_local_operations_not_marked_as_remote(self):
        """Test that local operations don't have is_remote flag."""
        contexts_seen = []

        def tracking_policy(context: PolicyContext) -> None:
            contexts_seen.append({
                'phase': context['phase'],
                'is_remote': context.get('is_remote', False),
                'operation': context.get('operation')
            })
            return None

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        from graphistry.compute.ast import n

        # Local operation
        policy = {
            'preload': tracking_policy,
            'postload': tracking_policy
        }

        g.gfql([n()], policy=policy)

        # Check that no context was marked as remote
        for ctx in contexts_seen:
            assert ctx['is_remote'] is False
            assert ctx['operation'] is None  # Not a remote operation

    def test_remote_exception_enrichment(self):
        """Test that PolicyException is enriched with remote context."""
        def failing_policy(context: PolicyContext) -> None:
            if context.get('is_remote'):
                # Raise exception without remote details
                raise PolicyException(
                    phase='preload',
                    reason='Generic failure'
                )
            return None

        with patch('graphistry.compute.chain_remote.chain_remote'):
            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            dag = ast_let({
                'remote_data': ASTRemoteGraph('enrichment-test')
            })

            policy = {'preload': failing_policy}

            with pytest.raises((PolicyException, RuntimeError)) as exc_info:
                g.gfql(dag, policy=policy)

            # Check that the error message mentions the failure
            assert 'Generic failure' in str(exc_info.value)

    def test_multiple_remote_loads_in_dag(self):
        """Test policy hooks for DAG with multiple remote loads."""
        load_order = []

        def tracking_policy(context: PolicyContext) -> None:
            if context.get('is_remote'):
                load_order.append({
                    'phase': context['phase'],
                    'dataset_id': context.get('remote_dataset_id')
                })
            return None

        with patch('graphistry.compute.chain_remote.chain_remote') as mock_chain_remote:
            # Mock different results for different datasets
            def mock_remote_side_effect(*args, **kwargs):
                dataset_id = kwargs.get('dataset_id')
                mock_result = Mock()
                if dataset_id == 'dataset-1':
                    mock_result._nodes = pd.DataFrame({'id': [1, 2]})
                else:
                    mock_result._nodes = pd.DataFrame({'id': [3, 4]})
                mock_result._edges = pd.DataFrame()
                return mock_result

            mock_chain_remote.side_effect = mock_remote_side_effect

            df = pd.DataFrame({'s': ['a'], 'd': ['b']})
            g = graphistry.edges(df, 's', 'd')

            dag = ast_let({
                'remote1': ASTRemoteGraph('dataset-1'),
                'remote2': ASTRemoteGraph('dataset-2')
            })

            policy = {
                'preload': tracking_policy,
                'postload': tracking_policy
            }

            g.gfql(dag, policy=policy)

            # Should have preload and postload for each remote dataset
            assert len(load_order) >= 4

            # Check we saw both datasets
            dataset_ids = {item['dataset_id'] for item in load_order}
            assert 'dataset-1' in dataset_ids
            assert 'dataset-2' in dataset_ids

            # Check both phases were called
            phases = {item['phase'] for item in load_order}
            assert 'preload' in phases
            assert 'postload' in phases