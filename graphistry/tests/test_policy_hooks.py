"""Tests for GFQL policy hooks."""

import pytest
import pandas as pd
import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
from graphistry.compute.ast import n, e


class TestPolicyHooks:
    """Test basic policy hook functionality."""

    def test_no_policy_backward_compat(self):
        """Test that queries work without policy (backward compatibility)."""
        # Create simple graph
        df = pd.DataFrame({
            's': ['a', 'b', 'c'],
            'd': ['b', 'c', 'd']
        })
        g = graphistry.edges(df, 's', 'd')

        # Should work without policy
        result = g.gfql([n()])
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_preload_hook_called(self):
        """Test that preload hook is called."""
        hook_called = {'preload': False}

        def preload_policy(context: PolicyContext) -> None:
            hook_called['preload'] = True
            assert context['phase'] == 'preload'
            assert 'query' in context
            assert 'query_type' in context

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql([n()], policy={'preload': preload_policy})
        assert hook_called['preload'], "Preload hook should have been called"

    def test_postload_hook_called(self):
        """Test that postload hook is called."""
        hook_called = {'postload': False}

        def postload_policy(context: PolicyContext) -> None:
            hook_called['postload'] = True
            assert context['phase'] == 'postload'
            assert 'plottable' in context
            assert 'graph_stats' in context
            # Check stats were extracted
            from typing import Any, Dict
            stats: Dict[str, Any] = context.get('graph_stats', {})  # type: ignore[assignment]
            assert 'nodes' in stats or 'edges' in stats

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql([n()], policy={'postload': postload_policy})
        assert hook_called['postload'], "Postload hook should have been called"

    def test_precall_hook_called(self):
        """Test that precall hook is called for call operations."""
        from graphistry.compute.ast import call

        hook_called = {'precall': False}

        def precall_policy(context: PolicyContext) -> None:
            hook_called['precall'] = True
            assert context['phase'] == 'precall'
            assert 'call_op' in context
            assert 'call_params' in context
            assert context['call_op'] == 'hop'  # We're testing hop operation
            # Precall should not have execution_time
            assert 'execution_time' not in context or context['execution_time'] is None

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Test with hop operation
        g.gfql(call('hop', {'hops': 2}), policy={'precall': precall_policy})
        assert hook_called['precall'], "Precall hook should have been called"

    def test_postcall_hook_called(self):
        """Test that postcall hook is called after call operations."""
        from graphistry.compute.ast import call

        hook_called = {'postcall': False}

        def postcall_policy(context: PolicyContext) -> None:
            hook_called['postcall'] = True
            assert context['phase'] == 'postcall'
            assert 'call_op' in context
            assert 'call_params' in context
            assert context['call_op'] == 'hop'  # We're testing hop operation
            # Postcall should have execution_time and success
            assert 'execution_time' in context
            assert context['execution_time'] is not None
            assert context['execution_time'] > 0
            assert context['success'] is True

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Test with hop operation
        g.gfql(call('hop', {'hops': 2}), policy={'postcall': postcall_policy})
        assert hook_called['postcall'], "Postcall hook should have been called"

    def test_multiple_hooks(self):
        """Test that multiple hooks can be used together."""
        hooks_called = {'preload': False, 'postload': False}

        def preload_policy(context: PolicyContext) -> None:
            hooks_called['preload'] = True

        def postload_policy(context: PolicyContext) -> None:
            hooks_called['postload'] = True

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'preload': preload_policy,
                'postload': postload_policy
            }
        )

        assert hooks_called['preload'], "Preload hook should have been called"
        assert hooks_called['postload'], "Postload hook should have been called"

    def test_hook_order(self):
        """Test that hooks are called in correct order."""
        call_order = []

        def preload_policy(context: PolicyContext) -> None:
            call_order.append('preload')

        def postload_policy(context: PolicyContext) -> None:
            call_order.append('postload')

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'preload': preload_policy,
                'postload': postload_policy
            }
        )

        assert call_order == ['preload', 'postload'], f"Expected ['preload', 'postload'], got {call_order}"

    def test_prelet_hook_called(self):
        """Test that prelet hook is called for let() DAGs."""
        hook_called = {'prelet': False}

        def prelet_policy(context: PolicyContext) -> None:
            hook_called['prelet'] = True
            assert context['phase'] == 'prelet'
            assert context['hook'] == 'prelet'
            assert context['query_type'] == 'dag'
            assert 'plottable' in context
            assert 'graph_stats' in context

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Test with let() DAG
        g.gfql({'nodes': n()}, policy={'prelet': prelet_policy})
        assert hook_called['prelet'], "Prelet hook should have been called"

    def test_postlet_hook_called(self):
        """Test that postlet hook is called after let() DAG execution."""
        hook_called = {'postlet': False}

        def postlet_policy(context: PolicyContext) -> None:
            hook_called['postlet'] = True
            assert context['phase'] == 'postlet'
            assert context['hook'] == 'postlet'
            assert context['query_type'] == 'dag'
            assert 'plottable' in context
            assert 'graph_stats' in context
            assert 'success' in context
            assert context['success'] is True

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Test with let() DAG
        g.gfql({'nodes': n()}, policy={'postlet': postlet_policy})
        assert hook_called['postlet'], "Postlet hook should have been called"

    def test_prechain_hook_called(self):
        """Test that prechain hook is called for chain queries."""
        hook_called = {'prechain': False}

        def prechain_policy(context: PolicyContext) -> None:
            hook_called['prechain'] = True
            assert context['phase'] == 'prechain'
            assert context['hook'] == 'prechain'
            assert context['query_type'] == 'chain'
            assert 'plottable' in context
            assert 'graph_stats' in context

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Test with chain query
        g.gfql([n()], policy={'prechain': prechain_policy})
        assert hook_called['prechain'], "Prechain hook should have been called"

    def test_postchain_hook_called(self):
        """Test that postchain hook is called after chain execution."""
        hook_called = {'postchain': False}

        def postchain_policy(context: PolicyContext) -> None:
            hook_called['postchain'] = True
            assert context['phase'] == 'postchain'
            assert context['hook'] == 'postchain'
            assert context['query_type'] == 'chain'
            assert 'plottable' in context
            assert 'graph_stats' in context
            assert 'success' in context
            assert context['success'] is True

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Test with chain query
        g.gfql([n()], policy={'postchain': postchain_policy})
        assert hook_called['postchain'], "Postchain hook should have been called"

    def test_let_dag_hook_order(self):
        """Test that prelet and postlet fire in correct order for let() DAGs."""
        call_order = []

        def prelet_policy(context: PolicyContext) -> None:
            call_order.append('prelet')

        def postlet_policy(context: PolicyContext) -> None:
            call_order.append('postlet')

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            {'nodes': n()},
            policy={
                'prelet': prelet_policy,
                'postlet': postlet_policy
            }
        )

        assert call_order == ['prelet', 'postlet'], f"Expected ['prelet', 'postlet'], got {call_order}"

    def test_chain_hook_order(self):
        """Test that prechain and postchain fire in correct order for chains."""
        call_order = []

        def prechain_policy(context: PolicyContext) -> None:
            call_order.append('prechain')

        def postchain_policy(context: PolicyContext) -> None:
            call_order.append('postchain')

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'prechain': prechain_policy,
                'postchain': postchain_policy
            }
        )

        assert call_order == ['prechain', 'postchain'], f"Expected ['prechain', 'postchain'], got {call_order}"

    def test_full_hierarchy_hook_order(self):
        """Test that all hooks fire in correct order: preload → prelet → postlet → postload."""
        call_order = []

        def preload_policy(context: PolicyContext) -> None:
            call_order.append('preload')

        def prelet_policy(context: PolicyContext) -> None:
            call_order.append('prelet')

        def postlet_policy(context: PolicyContext) -> None:
            call_order.append('postlet')

        def postload_policy(context: PolicyContext) -> None:
            call_order.append('postload')

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            {'nodes': n()},
            policy={
                'preload': preload_policy,
                'prelet': prelet_policy,
                'postlet': postlet_policy,
                'postload': postload_policy
            }
        )

        assert call_order == ['preload', 'prelet', 'postlet', 'postload'], \
            f"Expected ['preload', 'prelet', 'postlet', 'postload'], got {call_order}"
