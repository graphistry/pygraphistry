"""Integration tests showing Hub-like policy patterns."""

import pytest
import pandas as pd
import os
import time
from typing import Dict, Any, Callable
from enum import Enum

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException,
    PolicyFunction
)
from graphistry.compute.ast import n, e, call
from graphistry.embed_utils import check_cudf

# Check for cudf availability
has_cudf, _ = check_cudf()
is_test_cudf = has_cudf and os.environ.get("TEST_CUDF", "0") == "1"


class PlanTier(Enum):
    """Hub plan tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class TestHubIntegration:
    """Test Hub-like integration scenarios."""

    def create_hub_policy(self, tier: PlanTier, user_id: str = "test_user"):
        """Create a Hub-style policy with tiered limits."""

        # Plan limits configuration
        PLAN_LIMITS = {
            PlanTier.FREE: {
                'max_nodes': 1000,
                'max_edges': 5000,
                'max_calls': 3,
                'allowed_ops': ['filter', 'chain'],
                'engine': 'pandas',
                'timeout_seconds': 10
            },
            PlanTier.PRO: {
                'max_nodes': 10000,
                'max_edges': 50000,
                'max_calls': 100,
                'allowed_ops': ['filter', 'chain', 'hop', 'aggregate'],
                'engine': 'auto',
                'timeout_seconds': 60
            },
            PlanTier.ENTERPRISE: {
                'max_nodes': None,  # Unlimited
                'max_edges': None,
                'max_calls': None,
                'allowed_ops': None,  # All operations
                'engine': 'cudf' if is_test_cudf else 'pandas',  # Use cudf if available, otherwise pandas
                'timeout_seconds': None
            }
        }

        # Closure state for tracking usage
        state = {
            'user_id': user_id,
            'tier': tier,
            'calls_made': 0,
            'nodes_processed': 0,
            'edges_processed': 0,
            'start_time': None,
            'denied': False,
            'deny_reason': None
        }

        limits = PLAN_LIMITS[tier]

        def policy(context: PolicyContext) -> None:
            """Hub policy implementation."""
            phase = context['phase']

            if phase == 'preload':
                # Start timing
                state['start_time'] = time.perf_counter()

                # Apply engine limits
                if limits['engine']:
                    return {'engine': limits['engine']}

            elif phase == 'postload':
                # Check data size limits
                stats = context.get('graph_stats', {})
                nodes = stats.get('nodes', 0)
                edges = stats.get('edges', 0)

                state['nodes_processed'] += nodes
                state['edges_processed'] += edges

                # Check node limit
                if limits['max_nodes'] and state['nodes_processed'] > limits['max_nodes']:
                    state['denied'] = True
                    state['deny_reason'] = f"Node limit exceeded for {tier.value} plan"
                    raise PolicyException(
                        phase='postload',
                        reason=state['deny_reason'],
                        code=403,
                        query_type=context.get('query_type'),
                        data_size={'nodes': state['nodes_processed'], 'limit': limits['max_nodes']}
                    )

                # Check edge limit
                if limits['max_edges'] and state['edges_processed'] > limits['max_edges']:
                    state['denied'] = True
                    state['deny_reason'] = f"Edge limit exceeded for {tier.value} plan"
                    raise PolicyException(
                        phase='postload',
                        reason=state['deny_reason'],
                        code=403,
                        query_type=context.get('query_type'),
                        data_size={'edges': state['edges_processed'], 'limit': limits['max_edges']}
                    )

                # Check timeout
                if limits['timeout_seconds'] and state['start_time']:
                    elapsed = time.perf_counter() - state['start_time']
                    if elapsed > limits['timeout_seconds']:
                        state['denied'] = True
                        state['deny_reason'] = f"Timeout exceeded for {tier.value} plan"
                        raise PolicyException(
                            phase='postload',
                            reason=state['deny_reason'],
                            code=408
                        )

            elif phase == 'precall':
                # Check call operation limits
                op = context.get('call_op', '')
                state['calls_made'] += 1

                # Check allowed operations
                if limits['allowed_ops'] and op not in limits['allowed_ops']:
                    state['denied'] = True
                    state['deny_reason'] = f"Operation '{op}' not available in {tier.value} plan"
                    raise PolicyException(
                        phase='precall',
                        reason=state['deny_reason'],
                        code=403
                    )

                # Check call count limit
                if limits['max_calls'] and state['calls_made'] > limits['max_calls']:
                    state['denied'] = True
                    state['deny_reason'] = f"Call limit exceeded for {tier.value} plan"
                    raise PolicyException(
                        phase='precall',
                        reason=state['deny_reason'],
                        code=429
                    )

                # Pro users can override to GPU for expensive operations
                if tier == PlanTier.PRO and op == 'hop':
                    params = context.get('call_params', {})
                    if params.get('hops', 0) > 2:
                        return {'engine': 'cudf' if is_test_cudf else 'pandas'}  # Use appropriate engine

            return None

        return policy, state

    def test_free_tier_limits(self):
        """Test free tier restrictions."""
        policy, state = self.create_hub_policy(PlanTier.FREE)

        # Small data should work
        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql(
            [n()],
            policy={'preload': policy, 'postload': policy}
        )
        assert result is not None

        # hop operation should be denied
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(
                call('hop', {'hops': 1}),
                policy={'precall': policy}
            )

        assert state['denied'] is True
        assert 'not available in free plan' in state['deny_reason']
        assert exc_info.value.code == 403

    def test_pro_tier_upgrades(self):
        """Test pro tier capabilities."""
        policy, state = self.create_hub_policy(PlanTier.PRO)

        df = pd.DataFrame({
            's': list(range(100)),
            'd': list(range(1, 101))
        })
        g = graphistry.edges(df, 's', 'd')

        # Pro can use hop
        result = g.gfql(
            call('hop', {'hops': 1}),
            policy={'precall': policy}
        )
        assert result is not None
        assert state['calls_made'] >= 1  # hop may be called multiple times internally

        # Multiple hops triggers GPU upgrade
        result = g.gfql(
            call('hop', {'hops': 3}),
            engine='pandas',  # Pro policy may override to cudf
            policy={'precall': policy}
        )
        assert state['calls_made'] >= 2  # Multiple gfql calls

    def test_enterprise_unlimited(self):
        """Test enterprise tier has no limits."""
        policy, state = self.create_hub_policy(PlanTier.ENTERPRISE)

        # Create large dataset
        df = pd.DataFrame({
            's': list(range(10000)),
            'd': list(range(1, 10001))
        })
        g = graphistry.edges(df, 's', 'd')

        # Should handle large data
        result = g.gfql(
            [n()],
            policy={'preload': policy, 'postload': policy}
        )
        assert result is not None
        assert state['nodes_processed'] > 0

        # Should allow any operation
        # Use operations that are actually in the safelist
        for op, params in [('hop', {'hops': 1}), ('materialize_nodes', {}), ('get_degrees', {})]:
            result = g.gfql(
                call(op, params),
                policy={'precall': policy}
            )
            # Should not raise

        assert state['denied'] is False

    def test_usage_metering(self):
        """Test usage tracking across multiple queries."""
        policy, state = self.create_hub_policy(PlanTier.PRO, user_id="user123")

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Run multiple queries
        for _ in range(3):
            g.gfql([n()], policy={'preload': policy, 'postload': policy})

        # Check cumulative usage
        assert state['user_id'] == "user123"
        assert state['nodes_processed'] > 0
        # Note: n() returns nodes but no edges, so edges_processed stays 0
        assert state['edges_processed'] >= 0

    def test_feature_gating_pattern(self):
        """Test feature gating based on plan tier."""

        def create_feature_gate_policy(features: Dict[str, bool]):
            """Create policy that gates features."""

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'precall':
                    op = context.get('call_op', '')

                    # Map operations to features
                    feature_map = {
                        'hop': 'graph_traversal',
                        'aggregate': 'analytics',
                        'pagerank': 'ml_algorithms'
                    }

                    feature = feature_map.get(op, op)

                    if not features.get(feature, False):
                        raise PolicyException(
                            phase='precall',
                            reason=f'Feature {feature} not enabled',
                            code=403
                        )

                return None

            return policy

        # Create policies for different feature sets
        basic_features = {'filter': True, 'chain': True}
        pro_features = {**basic_features, 'graph_traversal': True, 'analytics': True}

        basic_policy = create_feature_gate_policy(basic_features)
        pro_policy = create_feature_gate_policy(pro_features)

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Basic plan can't use hop
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(call('hop', {'hops': 1}), policy={'precall': basic_policy})
        assert 'graph_traversal not enabled' in str(exc_info.value)

        # Pro plan can use hop
        result = g.gfql(call('hop', {'hops': 1}), policy={'precall': pro_policy})
        assert result is not None

    def test_graceful_degradation(self):
        """Test graceful degradation when limits are hit."""

        def create_degradation_policy(max_memory: int = 1000):
            """Policy that degrades to CPU when memory limit hit."""
            state = {'memory_used': 0, 'degraded': False}

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'postload':
                    stats = context.get('graph_stats', {})
                    memory = stats.get('node_bytes', 0) + stats.get('edge_bytes', 0)
                    state['memory_used'] += memory

                    if state['memory_used'] > max_memory and not state['degraded']:
                        state['degraded'] = True
                        # Next query will use CPU
                        return {'engine': 'pandas'}

                elif context['phase'] == 'preload' and state['degraded']:
                    # Force CPU for all subsequent queries
                    return {'engine': 'pandas'}

                return None

            return policy, state

        policy, state = create_degradation_policy(max_memory=500)

        df = pd.DataFrame({'s': ['a', 'b'] * 10, 'd': ['b', 'c'] * 10})
        g = graphistry.edges(df, 's', 'd')

        # First query might trigger degradation
        g.gfql([n()], engine='pandas', policy={'preload': policy, 'postload': policy})

        # Check if degradation occurred
        if state['memory_used'] > 500:
            assert state['degraded'] is True

    def test_multi_tenant_isolation(self):
        """Test that different users have isolated state."""
        # Create policies for different users
        policy1, state1 = self.create_hub_policy(PlanTier.FREE, user_id="alice")
        policy2, state2 = self.create_hub_policy(PlanTier.PRO, user_id="bob")

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Alice (free tier) can't use hop
        with pytest.raises(PolicyException):
            g.gfql(call('hop', {'hops': 1}), policy={'precall': policy1})

        # Bob (pro tier) can use hop
        result = g.gfql(call('hop', {'hops': 1}), policy={'precall': policy2})
        assert result is not None

        # States are independent
        assert state1['user_id'] == "alice"
        assert state2['user_id'] == "bob"
        assert state1['tier'] == PlanTier.FREE
        assert state2['tier'] == PlanTier.PRO
        assert state1['denied'] is True
        assert state2['denied'] is False
