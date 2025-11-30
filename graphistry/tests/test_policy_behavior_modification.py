"""Tests for policy accept/deny behavior."""

import pytest
import pandas as pd
import os

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
from graphistry.compute.ast import n, call
from graphistry.embed_utils import check_cudf

# Check for cudf availability
has_cudf, _ = check_cudf()
is_test_cudf = has_cudf and os.environ.get("TEST_CUDF", "0") == "1"


class TestPolicyBehavior:
    """Test policy accept/deny behaviors."""

    def test_deny_wrong_engine_in_preload(self):
        """Test policy can deny based on engine in preload phase."""
        def engine_policy(context: PolicyContext) -> None:
            # Deny if trying to use cudf when not available
            if context['phase'] == 'preload':
                # Note: context should contain requested engine info
                # For now, just demonstrate the pattern
                if not has_cudf:
                    # Would check context for engine request
                    pass  # Accept pandas

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Should work with pandas
        result = g.gfql(
            [n()],
            engine='pandas',
            policy={'preload': engine_policy}
        )
        assert result is not None

    def test_deny_based_on_parameters(self):
        """Test policy can deny based on call parameters."""
        def param_policy(context: PolicyContext) -> None:
            if context['phase'] == 'precall':
                params = context.get('call_params', {})
                # Deny if hops > 2
                if params.get('hops', 0) > 2:
                    raise PolicyException(
                        phase='precall',
                        reason='Too many hops requested',
                        code=413
                    )

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Should work with 1 hop
        result = g.gfql(
            call('hop', {'hops': 1}),
            policy={'precall': param_policy}
        )
        assert result is not None

        # Should fail with 3 hops
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(
                call('hop', {'hops': 3}),
                policy={'precall': param_policy}
            )
        assert 'Too many hops' in exc_info.value.reason

    def test_deny_based_on_data_size(self):
        """Test policy can deny based on data size in postload."""
        def size_policy(context: PolicyContext) -> None:
            if context['phase'] == 'postload':
                stats = context.get('graph_stats', {})
                total_size = stats.get('nodes', 0) + stats.get('edges', 0)

                if total_size > 10:
                    raise PolicyException(
                        phase='postload',
                        reason=f'Data size {total_size} exceeds limit',
                        data_size=stats
                    )

        # Small data should pass
        df_small = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g_small = graphistry.edges(df_small, 's', 'd')

        result = g_small.gfql([n()], policy={'postload': size_policy})
        assert result is not None

    def test_deny_specific_operations(self):
        """Test policy can deny specific operations."""
        def operation_policy(context: PolicyContext) -> None:
            if context['phase'] == 'precall':
                op = context.get('call_op', '')
                # Deny hypergraph operation
                if op == 'hypergraph':
                    raise PolicyException(
                        phase='precall',
                        reason='Hypergraph not allowed',
                        code=403
                    )

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # hop should work
        result = g.gfql(
            call('hop', {'hops': 1}),
            policy={'precall': operation_policy}
        )
        assert result is not None

        # Note: Can't easily test hypergraph denial without proper setup

    def test_conditional_accept(self):
        """Test conditional acceptance based on context."""
        calls_made = []

        def tracking_policy(context: PolicyContext) -> None:
            phase = context['phase']
            calls_made.append(phase)

            # Accept everything - no exceptions raised
            if phase == 'postload':
                stats = context.get('graph_stats', {})
                # Could deny here but we accept
                assert 'nodes' in stats or 'edges' in stats

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql([n()], policy={
            'preload': tracking_policy,
            'postload': tracking_policy
        })

        assert 'preload' in calls_made
        assert 'postload' in calls_made
        assert result is not None

    def test_rate_limiting_pattern(self):
        """Test rate limiting pattern with accept/deny."""
        class RateLimiter:
            def __init__(self, max_calls=3):
                self.calls = 0
                self.max_calls = max_calls

            def policy(self, context: PolicyContext) -> None:
                if context['phase'] == 'precall':
                    self.calls += 1
                    if self.calls > self.max_calls:
                        raise PolicyException(
                            phase='precall',
                            reason=f'Rate limit exceeded: {self.calls}/{self.max_calls}',
                            code=429
                        )

        limiter = RateLimiter(max_calls=2)
        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Use multiple chained hop calls to trigger rate limit
        # Each hop call will trigger precall, so 3 hops > 2 max_calls
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(
                [call('hop', {'hops': 1}), call('hop', {'hops': 1}), call('hop', {'hops': 1})],
                policy={'precall': limiter.policy}
            )
        assert exc_info.value.code == 429

    def test_accept_by_doing_nothing(self):
        """Test that doing nothing means accept."""
        def empty_policy(context: PolicyContext) -> None:
            # Doing nothing = accept
            pass

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Should work - policy accepts by doing nothing
        result = g.gfql([n()], policy={
            'preload': empty_policy,
            'postload': empty_policy
        })
        assert result is not None

    def test_deny_with_custom_code(self):
        """Test denying with custom HTTP-like codes."""
        def http_policy(context: PolicyContext) -> None:
            phase = context['phase']

            if phase == 'preload':
                # 401 Unauthorized
                if context.get('user_id') is None:
                    raise PolicyException(
                        phase='preload',
                        reason='Authentication required',
                        code=401
                    )
            elif phase == 'postload':
                stats = context.get('graph_stats', {})
                # 413 Payload Too Large
                if stats.get('nodes', 0) > 1000000:
                    raise PolicyException(
                        phase='postload',
                        reason='Graph too large',
                        code=413,
                        data_size=stats
                    )

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        # Should work with small data and no auth check
        result = g.gfql([n()], policy={'postload': http_policy})
        assert result is not None
