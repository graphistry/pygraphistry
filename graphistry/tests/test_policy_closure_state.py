"""Tests for closure-based state management in policies (Hub pattern)."""

import pytest
import pandas as pd
import time
from typing import Dict, Any

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
from graphistry.compute.ast import n, call


class TestClosureBasedState:
    """Test state management via closures as used by Hub."""

    def test_basic_state_tracking(self):
        """Test that closures can maintain state across calls."""
        def create_stateful_policy():
            state = {"call_count": 0, "phases_seen": []}

            def policy(context: PolicyContext) -> None:
                phase = context['phase']
                state["call_count"] += 1
                state["phases_seen"].append(phase)
                return None

            # Return both policy and state for testing
            return policy, state

        policy_func, state = create_stateful_policy()

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Execute with stateful policy
        g.gfql(
            [n()],
            policy={
                'preload': policy_func,
                'postload': policy_func
            }
        )

        # Check state was maintained
        assert state["call_count"] == 2
        assert 'preload' in state["phases_seen"]
        assert 'postload' in state["phases_seen"]

    def test_timing_via_closure(self):
        """Test performance tracking via closure (Hub pattern)."""
        def create_timing_policy():
            timings = {
                "start_time": None,
                "preload_time": None,
                "postload_time": None,
                "total_time": None
            }

            def policy(context: PolicyContext) -> None:
                phase = context['phase']

                if phase == 'preload':
                    timings["start_time"] = time.perf_counter()
                elif phase == 'postload':
                    if timings["start_time"]:
                        timings["postload_time"] = time.perf_counter()
                        timings["total_time"] = timings["postload_time"] - timings["start_time"]

                return None

            return policy, timings

        policy_func, timings = create_timing_policy()

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'preload': policy_func,
                'postload': policy_func
            }
        )

        # Should have captured timing
        assert timings["start_time"] is not None
        assert timings["postload_time"] is not None
        assert timings["total_time"] is not None
        assert timings["total_time"] > 0

    def test_rate_limiting_with_closure(self):
        """Test rate limiting using closure state."""
        def create_rate_limit_policy(max_calls: int = 3):
            state = {
                "call_count": 0,
                "denied": False
            }

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'precall':
                    state["call_count"] += 1

                    if state["call_count"] > max_calls:
                        state["denied"] = True
                        raise PolicyException(
                            phase='precall',
                            reason=f'Rate limit exceeded: {max_calls} calls',
                            code=429
                        )

                return None

            return policy, state

        # Set a low limit to test that the policy actually denies calls
        policy_func, state = create_rate_limit_policy(max_calls=1)

        df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e']
        })
        g = graphistry.edges(df, 's', 'd')

        # Use multiple chained hop calls to trigger rate limit
        # Each call will trigger precall once, so 2 calls > 1 max_calls
        with pytest.raises(PolicyException) as exc_info:
            g.gfql([call('hop', {'hops': 1}), call('hop', {'hops': 1})], policy={'precall': policy_func})

        assert exc_info.value.code == 429
        assert 'Rate limit' in exc_info.value.reason
        assert state["denied"] is True
        assert state["call_count"] > 1  # Should have tried multiple calls before failing

    def test_data_size_tracking(self):
        """Test tracking cumulative data sizes via closure."""
        def create_size_tracking_policy(max_bytes: int = 1000):
            state = {
                "total_nodes": 0,
                "total_edges": 0,
                "total_bytes": 0,
                "queries_processed": 0
            }

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'postload':
                    stats = context.get('graph_stats', {})

                    # Update cumulative stats
                    state["total_nodes"] += stats.get('nodes', 0)
                    state["total_edges"] += stats.get('edges', 0)
                    state["total_bytes"] += stats.get('node_bytes', 0) + stats.get('edge_bytes', 0)
                    state["queries_processed"] += 1

                    # Check limits
                    if state["total_bytes"] > max_bytes:
                        raise PolicyException(
                            phase='postload',
                            reason='Data size limit exceeded',
                            code=413,
                            data_size={
                                'total_bytes': state["total_bytes"],
                                'limit_bytes': max_bytes
                            }
                        )

                return None

            return policy, state

        policy_func, state = create_size_tracking_policy(max_bytes=10000)

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Process multiple queries
        g.gfql([n()], policy={'postload': policy_func})
        first_query_stats = state.copy()

        g.gfql([n()], policy={'postload': policy_func})

        # Stats should accumulate
        assert state["queries_processed"] == 2
        assert state["total_nodes"] >= first_query_stats["total_nodes"]
        assert state["total_edges"] >= first_query_stats["total_edges"]

    def test_feature_flag_via_closure(self):
        """Test feature flags managed via closure."""
        def create_feature_flag_policy(enabled_features: Dict[str, bool]):
            state = {
                "feature_checks": [],
                "denied_features": []
            }

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'precall':
                    op = context.get('call_op', '')
                    state["feature_checks"].append(op)

                    # Check if operation is enabled
                    if not enabled_features.get(op, True):
                        state["denied_features"].append(op)
                        raise PolicyException(
                            phase='precall',
                            reason=f'Feature not enabled: {op}',
                            code=403
                        )

                return None

            return policy, state

        # Create policy with hop disabled
        policy_func, state = create_feature_flag_policy({
            'hop': False,
            'filter': True
        })

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # hop should be denied
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(call('hop', {'hops': 1}), policy={'precall': policy_func})

        assert 'hop' in exc_info.value.reason
        assert 'hop' in state["denied_features"]

    def test_multiple_policies_with_shared_state(self):
        """Test multiple policy functions sharing state via closure."""
        def create_shared_state_policies():
            shared_state = {
                "preload_count": 0,
                "postload_count": 0,
                "call_count": 0
            }

            def preload_policy(context: PolicyContext) -> None:
                shared_state["preload_count"] += 1
                return None

            def postload_policy(context: PolicyContext) -> None:
                shared_state["postload_count"] += 1

                # Can see preload count
                if shared_state["preload_count"] == 0:
                    raise PolicyException('postload', 'Preload not called')

                return None

            def precall_policy(context: PolicyContext) -> None:
                shared_state["call_count"] += 1

                # Can see all counts
                total = sum(shared_state.values())
                if total > 10:
                    raise PolicyException('precall', 'Too many total operations')

                return None

            return {
                'preload': preload_policy,
                'postload': postload_policy,
                'precall': precall_policy
            }, shared_state

        policies, state = create_shared_state_policies()

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Execute with all policies
        g.gfql([n()], policy=policies)

        # All should have been called
        assert state["preload_count"] == 1
        assert state["postload_count"] == 1

        # Call operation (hop is called multiple times internally)
        g.gfql(call('hop', {'hops': 1}), policy=policies)
        assert state["call_count"] >= 1  # At least one call, likely more due to chain implementation

    def test_conditional_modification_via_state(self):
        """Test modifications that depend on accumulated state."""
        def create_adaptive_policy():
            state = {
                "slow_queries": 0,
                "forced_cpu": False
            }

            def policy(context: PolicyContext) -> None:
                if context['phase'] == 'preload':
                    # Force CPU if we've seen too many slow queries
                    if state["slow_queries"] >= 2 and not state["forced_cpu"]:
                        state["forced_cpu"] = True
                        return {'engine': 'pandas'}

                elif context['phase'] == 'postload':
                    # Simulate detecting a slow query
                    stats = context.get('graph_stats', {})
                    # Check nodes instead of edges since n() returns all nodes
                    if stats.get('nodes', 0) > 3:
                        state["slow_queries"] += 1

                return None

            return policy, state

        policy_func, state = create_adaptive_policy()

        df = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e']
        })
        g = graphistry.edges(df, 's', 'd')

        # First queries won't force CPU
        g.gfql([n()], policy={'preload': policy_func, 'postload': policy_func})
        assert state["forced_cpu"] is False

        # Second query increments slow count
        g.gfql([n()], policy={'preload': policy_func, 'postload': policy_func})

        # Third query should trigger CPU forcing
        g.gfql([n()], engine='pandas', policy={'preload': policy_func, 'postload': policy_func})
        assert state["forced_cpu"] is True  # Should have switched to CPU
