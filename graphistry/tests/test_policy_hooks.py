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
            stats = context.get('graph_stats', {})
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
