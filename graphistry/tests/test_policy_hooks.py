"""Tests for GFQL policy hooks."""

import pytest
import pandas as pd
from typing import Optional

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException,
    PolicyModification,
    validate_modification
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

        def preload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            hook_called['preload'] = True
            assert context['phase'] == 'preload'
            assert 'query' in context
            assert 'query_type' in context
            return None

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql([n()], policy={'preload': preload_policy})
        assert hook_called['preload'], "Preload hook should have been called"

    def test_postload_hook_called(self):
        """Test that postload hook is called."""
        hook_called = {'postload': False}

        def postload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            hook_called['postload'] = True
            assert context['phase'] == 'postload'
            assert 'plottable' in context
            assert 'graph_stats' in context
            # Check stats were extracted
            stats = context.get('graph_stats', {})
            assert 'nodes' in stats or 'edges' in stats
            return None

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql([n()], policy={'postload': postload_policy})
        assert hook_called['postload'], "Postload hook should have been called"

    def test_call_hook_called(self):
        """Test that call hook is called for call operations."""
        from graphistry.compute.ast import call

        hook_called = {'call': False}

        def call_policy(context: PolicyContext) -> Optional[PolicyModification]:
            hook_called['call'] = True
            assert context['phase'] == 'call'
            assert 'call_op' in context
            assert 'call_params' in context
            assert context['call_op'] == 'hop'  # We're testing hop operation
            return None

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Test with hop operation
        result = g.gfql(call('hop', {'hops': 2}), policy={'call': call_policy})
        assert hook_called['call'], "Call hook should have been called"

    def test_multiple_hooks(self):
        """Test that multiple hooks can be used together."""
        hooks_called = {'preload': False, 'postload': False}

        def preload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            hooks_called['preload'] = True
            return None

        def postload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            hooks_called['postload'] = True
            return None

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

        def preload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            call_order.append('preload')
            return None

        def postload_policy(context: PolicyContext) -> Optional[PolicyModification]:
            call_order.append('postload')
            return None

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
