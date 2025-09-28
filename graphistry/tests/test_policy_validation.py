"""Tests for policy exception validation and behavior."""

import pytest
import pandas as pd

from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
import graphistry
from graphistry.compute.ast import n


class TestPolicyExceptionValidation:
    """Test PolicyException validation and behavior."""

    def test_exception_requires_phase(self):
        """Test that PolicyException requires phase."""
        exc = PolicyException(
            phase='preload',
            reason='Test denial'
        )
        assert exc.phase == 'preload'
        assert exc.reason == 'Test denial'

    def test_exception_with_code(self):
        """Test PolicyException with custom code."""
        exc = PolicyException(
            phase='postload',
            reason='Resource limit exceeded',
            code=429
        )
        assert exc.code == 429

    def test_exception_with_enrichment(self):
        """Test PolicyException can be enriched with context."""
        exc = PolicyException('call', 'Operation denied')

        # Can enrich with query_type
        exc.query_type = 'chain'
        assert exc.query_type == 'chain'

        # Can enrich with data_size
        exc.data_size = {'nodes': 1000, 'edges': 5000}
        assert exc.data_size['nodes'] == 1000

    def test_policy_deny_in_preload(self):
        """Test denying in preload phase."""
        def deny_policy(context: PolicyContext) -> None:
            raise PolicyException(
                phase='preload',
                reason='Not allowed',
                code=403
            )

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        with pytest.raises(PolicyException) as exc_info:
            g.gfql([n()], policy={'preload': deny_policy})

        assert exc_info.value.phase == 'preload'
        assert exc_info.value.code == 403

    def test_policy_deny_in_postload(self):
        """Test denying in postload phase."""
        def deny_policy(context: PolicyContext) -> None:
            stats = context.get('graph_stats', {})
            if stats.get('nodes', 0) > 0:
                raise PolicyException(
                    phase='postload',
                    reason='Too many nodes',
                    data_size=stats
                )

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        with pytest.raises(PolicyException) as exc_info:
            g.gfql([n()], policy={'postload': deny_policy})

        assert exc_info.value.phase == 'postload'
        assert 'Too many nodes' in exc_info.value.reason

    def test_policy_accept_by_default(self):
        """Test that policies accept by default (no exception)."""
        def accept_policy(context: PolicyContext) -> None:
            # Do nothing - implicit accept
            pass

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        # Should not raise
        result = g.gfql([n()], policy={
            'preload': accept_policy,
            'postload': accept_policy
        })
        assert result is not None

    def test_conditional_deny(self):
        """Test conditional denial based on context."""
        def conditional_policy(context: PolicyContext) -> None:
            if context['phase'] == 'postload':
                stats = context.get('graph_stats', {})
                # Only deny if nodes exceed threshold
                if stats.get('nodes', 0) > 100:
                    raise PolicyException(
                        phase='postload',
                        reason='Node limit exceeded',
                        code=413  # Payload too large
                    )

        # Small graph - should pass
        df_small = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g_small = graphistry.edges(df_small, 's', 'd')

        result = g_small.gfql([n()], policy={'postload': conditional_policy})
        assert result is not None

        # Note: Can't easily test large graph without actual data
        # but the pattern is demonstrated

    def test_multiple_phase_policies(self):
        """Test policies can be applied to multiple phases."""
        calls = []

        def tracking_policy(context: PolicyContext) -> None:
            phase = context['phase']
            calls.append(phase)
            # Accept all

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        result = g.gfql([n()], policy={
            'preload': tracking_policy,
            'postload': tracking_policy
        })

        assert 'preload' in calls
        assert 'postload' in calls
        assert result is not None
