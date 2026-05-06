"""Tests for recursion prevention in policy execution."""

import pytest
import pandas as pd
import os

import graphistry
from graphistry.compute.gfql.policy import (
    PolicyContext
)
from graphistry.compute.ast import n
from graphistry.embed_utils import check_cudf

# Check for cudf availability
has_cudf, _ = check_cudf()
is_test_cudf = has_cudf and os.environ.get("TEST_CUDF", "0") == "1"


class TestRecursionPrevention:
    """Test that recursion is prevented at depth 1."""

    def test_no_recursion_on_query_modification(self):
        """Test that modifying query doesn't trigger policy again."""
        call_count = {'count': 0}

        def modifying_policy(context: PolicyContext) -> None:
            call_count['count'] += 1

            # Should only be called at depth 0
            depth = context.get('_policy_depth', 0)
            assert depth == 0, f"Policy called at depth {depth}, should only be depth 0"

            if context['phase'] == 'preload':
                # Modify query - this should not trigger policy again
                return {'query': [n()]}  # Just get all nodes, don't filter on non-existent column
            return None

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Execute with policy
        result = g.gfql([n()], policy={'preload': modifying_policy})

        # Policy should only be called once
        assert call_count['count'] == 1, f"Policy called {call_count['count']} times, expected 1"
        assert result is not None

    def test_depth_tracking_in_context(self):
        """Test that _policy_depth is properly tracked in context."""
        depths_seen = []

        def tracking_policy(context: PolicyContext) -> None:
            depth = context.get('_policy_depth', -1)
            depths_seen.append(depth)
            return None

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql([n()], policy={'preload': tracking_policy})

        # Should only see depth 0
        assert depths_seen == [0], f"Expected depths [0], got {depths_seen}"

    def test_postload_not_called_recursively(self):
        """Test that postload hook is not called recursively."""
        postload_calls = {'count': 0}

        def preload_policy(context: PolicyContext) -> None:
            # Modify query
            return {'query': [n()]}  # Just get all nodes

        def postload_policy(context: PolicyContext) -> None:
            postload_calls['count'] += 1
            depth = context.get('_policy_depth', 0)

            # Should only be called at original depth
            assert depth == 0, f"Postload called at depth {depth}"
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

        # Postload should only be called once, even though query was modified
        assert postload_calls['count'] == 1, f"Postload called {postload_calls['count']} times"

    @pytest.mark.skipif(not is_test_cudf, reason="requires cudf for engine conversion")
    def test_complex_modification_no_recursion(self):
        """Test complex modifications don't cause recursion."""
        execution_log = []

        def complex_policy(context: PolicyContext) -> None:
            phase = context['phase']
            execution_log.append(phase)

            if phase == 'preload':
                # Modify both query and engine
                return {
                    'query': [n(), n()],  # Just get nodes twice
                    'engine': 'pandas'
                }
            elif phase == 'postload':
                # Try to modify engine in postload
                return {'engine': 'cudf'}

            return None

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'preload': complex_policy,
                'postload': complex_policy
            }
        )

        # Each phase should only be called once
        assert execution_log.count('preload') == 1
        assert execution_log.count('postload') == 1

    def test_malicious_policy_cannot_loop(self):
        """Test that a malicious policy cannot cause infinite loop."""
        call_count = {'count': 0}
        MAX_CALLS = 10  # Safety limit for test

        def malicious_policy(context: PolicyContext) -> None:
            call_count['count'] += 1

            # Safety check for test
            if call_count['count'] > MAX_CALLS:
                raise RuntimeError("Too many calls - loop detected!")

            # Always try to modify query
            return {'query': [n()]}  # Just get all nodes

        df = pd.DataFrame({'s': ['a'], 'd': ['b']})
        g = graphistry.edges(df, 's', 'd')

        # Should complete without infinite loop
        result = g.gfql([n()], policy={'preload': malicious_policy})

        # Policy should only be called once due to depth limit
        assert call_count['count'] == 1, f"Policy called {call_count['count']} times"
        assert result is not None
