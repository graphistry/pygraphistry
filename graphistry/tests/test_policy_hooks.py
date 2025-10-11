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

    def test_shortcuts_with_pre_post(self):
        """Test that shortcuts 'pre' and 'post' work in gfql()."""
        call_order = []

        def pre_handler(ctx: PolicyContext) -> None:
            call_order.append(f"pre:{ctx['phase']}")

        def post_handler(ctx: PolicyContext) -> None:
            call_order.append(f"post:{ctx['phase']}")

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'pre': pre_handler,
                'post': post_handler
            }
        )

        # Should have called all pre* and post* hooks
        pre_calls = [c for c in call_order if c.startswith('pre:')]
        post_calls = [c for c in call_order if c.startswith('post:')]

        # Should have preload and prechain
        assert 'pre:preload' in call_order
        assert 'pre:prechain' in call_order

        # Should have postload and postchain
        assert 'post:postload' in call_order
        assert 'post:postchain' in call_order

    def test_shortcuts_compose_with_full_names(self):
        """Test that shortcuts compose with full hook names."""
        call_order = []

        def pre_handler(ctx: PolicyContext) -> None:
            call_order.append('pre')

        def specific_preload(ctx: PolicyContext) -> None:
            call_order.append('preload')

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        g.gfql(
            [n()],
            policy={
                'pre': pre_handler,
                'preload': specific_preload
            }
        )

        # Should have both handlers called at preload
        # Order: general (pre) → specific (preload)
        preload_idx = None
        for i, call in enumerate(call_order):
            if call == 'pre':
                if i + 1 < len(call_order) and call_order[i + 1] == 'preload':
                    preload_idx = i
                    break

        assert preload_idx is not None, "Expected 'pre' followed by 'preload' in call order"

    def test_full_hierarchy_hook_order(self):
        """Test that all hooks fire in correct order for DAG queries.

        Note: When using dict convenience syntax, postload fires after loading data
        for the DAG binding, so we see: preload → prelet → postload → postlet → postload
        """
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

        # postload fires twice: once after loading DAG binding data, once after final execution
        assert call_order == ['preload', 'prelet', 'postload', 'postlet', 'postload'], \
            f"Expected ['preload', 'prelet', 'postload', 'postlet', 'postload'], got {call_order}"

    def test_shortcuts_with_dag_query(self):
        """Test that shortcuts work with let() DAG queries."""
        call_order = []

        def pre_handler(ctx: PolicyContext) -> None:
            call_order.append(f"pre:{ctx['phase']}")

        def post_handler(ctx: PolicyContext) -> None:
            call_order.append(f"post:{ctx['phase']}")

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Test with let() DAG using dict convenience
        g.gfql(
            {'nodes': n()},
            policy={
                'pre': pre_handler,
                'post': post_handler
            }
        )

        # Should have called prelet and postlet
        assert 'pre:prelet' in call_order, f"Expected prelet in {call_order}"
        assert 'post:postlet' in call_order, f"Expected postlet in {call_order}"

        # Should also have preload and postload
        assert 'pre:preload' in call_order
        assert 'post:postload' in call_order

    def test_shortcuts_with_call_operation(self):
        """Test that shortcuts work with call operations."""
        from graphistry.compute.ast import call

        call_order = []

        def pre_handler(ctx: PolicyContext) -> None:
            call_order.append('pre')

        def call_handler(ctx: PolicyContext) -> None:
            call_order.append('call')

        def post_handler(ctx: PolicyContext) -> None:
            call_order.append('post')

        df = pd.DataFrame({'s': ['a', 'b', 'c'], 'd': ['b', 'c', 'd']})
        g = graphistry.edges(df, 's', 'd')

        # Test with call operation using shortcuts
        g.gfql(
            call('hop', {'hops': 2}),
            policy={
                'pre': pre_handler,
                'call': call_handler,
                'post': post_handler
            }
        )

        # precall should have both 'pre' (general) and 'call' (scope)
        # They should appear in order in call_order
        pre_idx = call_order.index('pre')
        call_idx = call_order.index('call')
        assert pre_idx < call_idx, "pre should be called before call"

        # postcall should also have both
        post_idx = call_order.index('post')
        assert post_idx > call_idx, "post should be called after call"

    def test_three_orthogonal_policies_server_pattern(self):
        """Test server's multi-policy pattern with three orthogonal policies."""
        call_order = []

        # Policy 1: OpenTelemetry tracing (general pre/post)
        def create_span(ctx: PolicyContext) -> None:
            call_order.append('trace_start')

        def end_span(ctx: PolicyContext) -> None:
            call_order.append('trace_end')

        # Policy 2: Size checking (specific to postload)
        def check_size(ctx: PolicyContext) -> None:
            call_order.append('size_check')
            # In real usage, would check graph_stats

        # Policy 3: Rate limiting (specific to precall)
        def rate_limit(ctx: PolicyContext) -> None:
            call_order.append('rate_limit')
            # In real usage, would check rate limits

        df = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        g = graphistry.edges(df, 's', 'd')

        # Server pattern: combine orthogonal policies
        g.gfql(
            [n()],
            policy={
                'pre': create_span,      # Traces all operations
                'post': end_span,        # Ends all traces
                'postload': check_size,  # Specific size check
                'prechain': rate_limit   # Specific rate limit
            }
        )

        # Verify tracing happened
        assert 'trace_start' in call_order
        assert 'trace_end' in call_order

        # Verify specific policies happened
        assert 'size_check' in call_order
        assert 'rate_limit' in call_order

        # Verify ordering: preload (trace_start) → prechain (trace_start + rate_limit)
        # → postchain (trace_end) → postload (size_check + trace_end)

        # trace_start should appear before rate_limit
        first_trace_start_idx = call_order.index('trace_start')
        rate_limit_idx = call_order.index('rate_limit')
        assert first_trace_start_idx < rate_limit_idx

        # For postload, the composition order is: size_check (specific) → trace_end (general, reversed)
        # Find the last occurrence of trace_end (since trace_end appears multiple times)
        size_check_idx = call_order.index('size_check')
        trace_end_indices = [i for i, x in enumerate(call_order) if x == 'trace_end']
        last_trace_end_idx = trace_end_indices[-1]
        assert size_check_idx < last_trace_end_idx, \
            f"Expected size_check before final trace_end in postload composition"


class TestOpenTelemetryIntegration:
    """Integration tests for OpenTelemetry-style span tracing with complex nested queries."""

    def test_nested_query_span_hierarchy_success(self):
        """Test that spans form proper hierarchy for complex nested GFQL query (success case)."""
        # Mock OpenTelemetry-style span tracker
        class SpanTracker:
            def __init__(self):
                self.spans = []  # List of (action, span_name, parent_span_name)
                self.open_spans = []  # Stack of currently open span names

            def start_span(self, name: str) -> None:
                parent = self.open_spans[-1] if self.open_spans else None
                self.spans.append(('start', name, parent))
                self.open_spans.append(name)

            def end_span(self) -> None:
                if self.open_spans:
                    span_name = self.open_spans.pop()
                    self.spans.append(('end', span_name, None))

            def get_span_tree(self) -> str:
                """Get visual representation of span hierarchy."""
                lines = []
                for action, name, parent in self.spans:
                    if action == 'start':
                        indent = '  ' * len([s for s in self.spans[:self.spans.index((action, name, parent))]
                                           if s[0] == 'start' and s[1] not in
                                           [e[1] for e in self.spans[:self.spans.index((action, name, parent))]
                                            if e[0] == 'end']])
                        parent_info = f" (parent: {parent})" if parent else " (root)"
                        lines.append(f"{indent}→ {name}{parent_info}")
                    else:  # end
                        indent = '  ' * (len([s for s in self.spans[:self.spans.index((action, name, parent))]
                                            if s[0] == 'start' and s[1] not in
                                            [e[1] for e in self.spans[:self.spans.index((action, name, parent))]
                                             if e[0] == 'end']]) - 1)
                        lines.append(f"{indent}← {name}")
                return '\n'.join(lines)

        tracker = SpanTracker()

        # Create policy that mimics OpenTelemetry span creation/cleanup
        def create_span(ctx: PolicyContext) -> None:
            phase = ctx['phase']
            operation = ctx.get('operation_path', 'query')
            span_name = f"{phase}:{operation}"
            tracker.start_span(span_name)

        def end_span(ctx: PolicyContext) -> None:
            tracker.end_span()

        # Complex nested query: let() DAG with multiple bindings, chains, and refs
        from graphistry.compute.ast import let, ref

        # Create graph with node and edge attributes
        df_edges = pd.DataFrame({
            's': ['a', 'b', 'c', 'd'],
            'd': ['b', 'c', 'd', 'e'],
            'rel': ['knows', 'knows', 'works_with', 'knows']
        })
        df_nodes = pd.DataFrame({
            'id': ['a', 'b', 'c', 'd', 'e'],
            'type': ['person', 'person', 'person', 'person', 'org'],
            'active': [True, True, False, True, True]
        })
        g = graphistry.edges(df_edges, 's', 'd').nodes(df_nodes, 'id')

        # Execute complex nested query with multiple bindings and refs
        result = g.gfql(
            let({
                'base_nodes': n({'type': 'person'}),
                'filtered': ref('base_nodes', [e({'rel': 'knows'}), n({'active': True})]),
                'expanded': ref('filtered', [e(), n()])
            }),
            policy={
                'pre': create_span,
                'post': end_span
            }
        )

        # Verify result is valid
        assert result is not None

        # Verify all spans were closed (LIFO order)
        assert len(tracker.open_spans) == 0, f"Spans left open: {tracker.open_spans}"

        # Verify span structure
        span_tree = tracker.get_span_tree()
        print(f"\n=== Span Tree (Success Case) ===\n{span_tree}\n")

        # Check that spans were created and closed
        starts = [s for s in tracker.spans if s[0] == 'start']
        ends = [s for s in tracker.spans if s[0] == 'end']
        assert len(starts) == len(ends), f"Mismatch: {len(starts)} starts vs {len(ends)} ends"

        # Verify key phases are present in span names (pre hooks create spans)
        span_names = [s[1] for s in tracker.spans if s[0] == 'start']
        assert any('preload' in name for name in span_names), "Missing preload span"
        assert any('prelet' in name for name in span_names), "Missing prelet span"
        assert any('preletbinding' in name for name in span_names), "Missing preletbinding span"
        assert any('prechain' in name for name in span_names), "Missing prechain span"

        # Verify LIFO cleanup: each span should be closed before its parent
        for i, (action, name, _) in enumerate(tracker.spans):
            if action == 'end':
                # Find the corresponding start
                start_idx = None
                for j in range(i - 1, -1, -1):
                    if tracker.spans[j][0] == 'start' and tracker.spans[j][1] == name:
                        start_idx = j
                        break
                assert start_idx is not None, f"No start found for end: {name}"

                # Verify all children were closed before parent
                for j in range(start_idx + 1, i):
                    if tracker.spans[j][0] == 'start':
                        child_name = tracker.spans[j][1]
                        # Check if this child was closed
                        child_closed = any(
                            tracker.spans[k][0] == 'end' and tracker.spans[k][1] == child_name
                            for k in range(j + 1, i)
                        )
                        assert child_closed, f"Child span {child_name} not closed before parent {name}"

    def test_nested_query_span_hierarchy_with_exception(self):
        """Test that spans are properly closed even when exception occurs mid-execution."""
        # Mock span tracker
        class SpanTracker:
            def __init__(self):
                self.spans = []  # List of (action, span_name)
                self.open_spans = []  # Stack of currently open span names

            def start_span(self, name: str) -> None:
                self.spans.append(('start', name))
                self.open_spans.append(name)

            def end_span(self) -> None:
                if self.open_spans:
                    span_name = self.open_spans.pop()
                    self.spans.append(('end', span_name))

            def get_unclosed_spans(self):
                return self.open_spans.copy()

        tracker = SpanTracker()
        exception_fired = {'prelet': False}

        # Create policy that mimics OpenTelemetry
        def create_span(ctx: PolicyContext) -> None:
            phase = ctx['phase']
            operation = ctx.get('operation_path', 'query')
            span_name = f"{phase}:{operation}"
            tracker.start_span(span_name)

        def end_span(ctx: PolicyContext) -> None:
            tracker.end_span()

        # Policy that throws exception in prelet (after preload, before let execution)
        def throw_in_prelet(ctx: PolicyContext) -> None:
            if ctx['phase'] == 'prelet' and not exception_fired['prelet']:
                exception_fired['prelet'] = True
                # Record that we're about to throw
                raise PolicyException(
                    phase='prelet',
                    reason='Test exception during let execution',
                    code=500
                )

        # Create graph with node attributes
        from graphistry.compute.ast import let, ref

        df_edges = pd.DataFrame({'s': ['a', 'b'], 'd': ['b', 'c']})
        df_nodes = pd.DataFrame({'id': ['a', 'b', 'c'], 'type': ['person', 'person', 'person']})
        g = graphistry.edges(df_edges, 's', 'd').nodes(df_nodes, 'id')

        # Execute query that will fail in prelet
        with pytest.raises(PolicyException) as exc_info:
            g.gfql(
                let({
                    'nodes': n({'type': 'person'}),
                    'edges': ref('nodes', [e(), n()])
                }),
                policy={
                    'pre': create_span,
                    'prelet': throw_in_prelet,  # Throws after preload, before let execution
                    'post': end_span
                }
            )

        assert exc_info.value.phase == 'prelet'
        assert exc_info.value.reason == 'Test exception during let execution'

        # Print span tree for debugging
        span_tree_lines = []
        for action, name in tracker.spans:
            if action == 'start':
                span_tree_lines.append(f"  → {name}")
            else:
                span_tree_lines.append(f"  ← {name}")
        span_tree = '\n'.join(span_tree_lines)
        print(f"\n=== Span Tree (Exception Case) ===\n{span_tree}\n")

        # CRITICAL: Verify all spans were closed despite exception
        unclosed = tracker.get_unclosed_spans()
        assert len(unclosed) == 0, \
            f"Spans left unclosed after exception: {unclosed}. This indicates improper LIFO cleanup."

        # Verify partial execution: should have preload but prelet should have been interrupted
        span_names = [s[1] for s in tracker.spans if s[0] == 'start']
        assert any('preload' in name for name in span_names), "Should have preload span"
        # prelet creates the span but exception happens in prelet phase before let executes
        assert any('prelet' in name for name in span_names), "Should have prelet span (started before exception)"

        # Verify LIFO cleanup: every span that was opened should be closed
        starts = [s for s in tracker.spans if s[0] == 'start']
        ends = [s for s in tracker.spans if s[0] == 'end']
        assert len(starts) == len(ends), \
            f"LIFO violation: {len(starts)} spans opened but only {len(ends)} closed"

        # Verify cleanup order: last opened should be first closed (LIFO)
        start_order = [s[1] for s in tracker.spans if s[0] == 'start']
        end_order = [s[1] for s in tracker.spans if s[0] == 'end']
        expected_end_order = list(reversed(start_order))
        assert end_order == expected_end_order, \
            f"LIFO cleanup order violated.\nExpected: {expected_end_order}\nActual: {end_order}"
