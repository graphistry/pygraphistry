"""Tests for PolicyException enrichment."""

import pytest
import pandas as pd
# No longer need Optional since policies return None

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException
)
from graphistry.compute.ast import n, e


class TestPolicyExceptions:
    """Test PolicyException with enriched error data."""

    def test_exception_basic_fields(self):
        """Test that PolicyException has all basic fields."""
        exc = PolicyException(
            phase='preload',
            reason='Test denial',
            code=403
        )

        assert exc.phase == 'preload'
        assert exc.reason == 'Test denial'
        assert exc.code == 403
        assert str(exc) == 'Policy denial in preload: Test denial'

    def test_exception_enrichment(self):
        """Test PolicyException with enriched data."""
        exc = PolicyException(
            phase='postload',
            reason='Dataset too large',
            code=403,
            query_type='chain',
            data_size={'nodes': 1000000, 'edges': 5000000}
        )

        assert exc.query_type == 'chain'
        assert exc.data_size == {'nodes': 1000000, 'edges': 5000000}

        # Test to_dict for JSON serialization
        exc_dict = exc.to_dict()
        assert exc_dict['code'] == 403
        assert exc_dict['phase'] == 'postload'
        assert exc_dict['reason'] == 'Dataset too large'
        assert exc_dict['query_type'] == 'chain'
        assert exc_dict['data_size'] == {'nodes': 1000000, 'edges': 5000000}

    def test_exception_from_preload_hook(self):
        """Test that exceptions from preload hook are propagated."""
        def denying_policy(context: PolicyContext) -> None:
            raise PolicyException(
                phase='preload',
                reason='Denied in test',
                query_type=context.get('query_type')
            )

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        with pytest.raises(PolicyException) as exc_info:
            g.gfql([n()], policy={'preload': denying_policy})

        exc = exc_info.value
        assert exc.phase == 'preload'
        assert exc.reason == 'Denied in test'
        assert exc.query_type in ['chain', 'dag', 'single']  # Should be set

    def test_exception_from_postload_hook(self):
        """Test that exceptions from postload hook include stats."""
        def denying_policy(context: PolicyContext) -> None:
            stats = context.get('graph_stats', {})
            raise PolicyException(
                phase='postload',
                reason='Too many nodes',
                query_type='chain',
                data_size=stats
            )

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        with pytest.raises(PolicyException) as exc_info:
            g.gfql([n()], policy={'postload': denying_policy})

        exc = exc_info.value
        assert exc.phase == 'postload'
        assert exc.reason == 'Too many nodes'
        assert exc.data_size is not None
        # Should have some stats
        if exc.data_size:
            assert 'nodes' in exc.data_size or 'edges' in exc.data_size

    def test_exception_from_call_hook(self):
        """Test that exceptions from call hook include operation details."""
        from graphistry.compute.ast import call

        def denying_policy(context: PolicyContext) -> None:
            raise PolicyException(
                phase='precall',
                reason=f"Operation {context.get('call_op')} not allowed",
                query_type='call'
            )

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        with pytest.raises(PolicyException) as exc_info:
            g.gfql(call('hop', {'hops': 2}), policy={'precall': denying_policy})

        exc = exc_info.value
        assert exc.phase == 'precall'
        assert 'hop' in exc.reason or 'not allowed' in exc.reason

    def test_exception_json_serializable(self):
        """Test that exception can be serialized to JSON."""
        import json

        exc = PolicyException(
            phase='postload',
            reason='Test',
            code=403,
            query_type='chain',
            data_size={'nodes': 100}
        )

        # Should be JSON serializable
        json_str = json.dumps(exc.to_dict())
        parsed = json.loads(json_str)

        assert parsed['phase'] == 'postload'
        assert parsed['reason'] == 'Test'
        assert parsed['code'] == 403
        assert parsed['query_type'] == 'chain'
        assert parsed['data_size']['nodes'] == 100

    def test_exception_accepts_all_phases(self):
        """Test that PolicyException accepts all 10 phases."""
        all_phases = [
            'preload', 'postload',
            'prelet', 'postlet',
            'prechain', 'postchain',
            'preletbinding', 'postletbinding',
            'precall', 'postcall'
        ]

        for phase in all_phases:
            exc = PolicyException(
                phase=phase,
                reason=f'Test {phase}',
                code=403
            )

            assert exc.phase == phase
            assert exc.reason == f'Test {phase}'
            assert str(exc) == f'Policy denial in {phase}: Test {phase}'

            # Verify to_dict works
            exc_dict = exc.to_dict()
            assert exc_dict['phase'] == phase
            assert exc_dict['reason'] == f'Test {phase}'
