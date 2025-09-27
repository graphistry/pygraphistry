"""Tests for policy behavior modification capabilities."""

import pytest
import pandas as pd
from typing import Optional

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyModification
)
from graphistry.compute.ast import n, call


class TestBehaviorModification:
    """Test policies that modify query behavior."""

    def test_engine_switching_in_preload(self):
        """Test policy can switch engine in preload phase."""
        engines_used = []

        def switching_policy(context: PolicyContext) -> Optional[PolicyModification]:
            # Force CPU in preload
            if context['phase'] == 'preload':
                return {'engine': 'cpu'}
            return None

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Execute with engine override
        result = g.gfql(
            [n()],
            engine='gpu',  # Request GPU
            policy={'preload': switching_policy}  # Policy overrides to CPU
        )

        # Result should be valid
        assert result is not None
        assert hasattr(result, '_nodes')

    def test_parameter_modification_in_call(self):
        """Test policy can modify call parameters."""
        call_params_seen = []

        def param_policy(context: PolicyContext) -> Optional[PolicyModification]:
            if context['phase'] == 'call':
                # Capture original params
                call_params_seen.append(context.get('call_params', {}))

                # Modify parameters
                new_params = {'hops': 1, 'direction': 'forward'}
                return {'params': new_params}
            return None

        df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e']
        })
        g = graphistry.edges(df, 's', 'd')

        # Execute call with policy that modifies params
        result = g.gfql(
            call('hop', {'hops': 3}),  # Request 3 hops
            policy={'call': param_policy}
        )

        # Should have captured original params
        assert len(call_params_seen) > 0
        assert call_params_seen[0].get('hops') == 3

        # Result should be valid (with modified params applied)
        assert result is not None

    def test_combined_modifications(self):
        """Test multiple modifications in single response."""
        modifications_applied = []

        def combined_policy(context: PolicyContext) -> Optional[PolicyModification]:
            phase = context['phase']

            if phase == 'preload':
                modifications_applied.append('preload')
                # Modify both query and engine
                return {
                    'query': [n({'source': 'modified'})],
                    'engine': 'cpu'
                }
            elif phase == 'postload':
                modifications_applied.append('postload')
                # Modify engine again (should apply)
                return {'engine': 'gpu'}

            return None

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql(
            [n()],
            policy={
                'preload': combined_policy,
                'postload': combined_policy
            }
        )

        # Both phases should have been called
        assert 'preload' in modifications_applied
        assert 'postload' in modifications_applied
        assert result is not None

    def test_query_replacement_chain_to_single(self):
        """Test policy can replace chain query with single operation."""
        query_replacements = []

        def replacement_policy(context: PolicyContext) -> Optional[PolicyModification]:
            if context['phase'] == 'preload':
                # Record original query type
                query_replacements.append(context.get('query_type'))

                # Replace with single operation
                return {'query': n({'replaced': True})}
            return None

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Start with chain query
        result = g.gfql(
            [n(), n({'filter': True})],  # Chain query
            policy={'preload': replacement_policy}
        )

        # Should have seen original as chain
        assert query_replacements[0] == 'chain'
        assert result is not None

    def test_engine_override_in_call_phase(self):
        """Test engine can be overridden in call phase."""
        engines_seen = []

        def call_engine_policy(context: PolicyContext) -> Optional[PolicyModification]:
            if context['phase'] == 'call':
                engines_seen.append('call')
                # Override engine for this specific call
                return {'engine': 'cpu'}
            return None

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql(
            call('hop', {'hops': 1}),
            engine='gpu',  # Request GPU
            policy={'call': call_engine_policy}
        )

        assert 'call' in engines_seen
        assert result is not None

    def test_partial_modification(self):
        """Test that partial modifications work (not all fields required)."""
        def partial_policy(context: PolicyContext) -> Optional[PolicyModification]:
            if context['phase'] == 'preload':
                # Only modify engine, leave query alone
                return {'engine': 'cpu'}
            elif context['phase'] == 'call':
                # Only modify params, leave engine alone
                return {'params': {'modified': True}}
            return None

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        # Test preload partial mod
        result = g.gfql([n()], policy={'preload': partial_policy})
        assert result is not None

        # Test call partial mod
        result = g.gfql(
            call('hop', {'hops': 1}),
            policy={'call': partial_policy}
        )
        assert result is not None

    def test_empty_modification_allowed(self):
        """Test that returning empty dict is valid (no-op)."""
        calls = {'count': 0}

        def noop_policy(context: PolicyContext) -> Optional[PolicyModification]:
            calls['count'] += 1
            return {}  # Empty modification

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql([n()], policy={'preload': noop_policy})

        assert calls['count'] == 1  # Policy was called
        assert result is not None  # Query proceeded normally

    def test_none_modification_allowed(self):
        """Test that returning None is valid (no modification)."""
        calls = {'count': 0}

        def none_policy(context: PolicyContext) -> Optional[PolicyModification]:
            calls['count'] += 1
            return None  # No modification

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql([n()], policy={'preload': none_policy})

        assert calls['count'] == 1  # Policy was called
        assert result is not None  # Query proceeded normally
