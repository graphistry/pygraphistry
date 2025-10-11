"""Tests for GFQL policy shortcuts functionality."""

import pytest
from graphistry.compute.gfql.policy import expand_policy, debug_policy, PolicyContext


class TestExpandPolicyBasics:
    """Test basic expansion of shortcuts to hooks."""

    def test_pre_expands_to_all_pre_hooks(self):
        """Test that 'pre' shortcut expands to all 5 pre* hooks."""
        def handler(ctx): pass

        policy = {'pre': handler}
        expanded = expand_policy(policy)

        # Should expand to all pre* hooks
        assert 'preload' in expanded
        assert 'prelet' in expanded
        assert 'prechain' in expanded
        assert 'preletbinding' in expanded
        assert 'precall' in expanded

        # All should point to the same handler
        assert expanded['preload'] is handler
        assert expanded['prelet'] is handler
        assert expanded['prechain'] is handler
        assert expanded['preletbinding'] is handler
        assert expanded['precall'] is handler

        # Should not expand to post* hooks
        assert 'postload' not in expanded
        assert 'postlet' not in expanded

    def test_post_expands_to_all_post_hooks(self):
        """Test that 'post' shortcut expands to all 5 post* hooks."""
        def handler(ctx): pass

        policy = {'post': handler}
        expanded = expand_policy(policy)

        # Should expand to all post* hooks
        assert 'postload' in expanded
        assert 'postlet' in expanded
        assert 'postchain' in expanded
        assert 'postletbinding' in expanded
        assert 'postcall' in expanded

        # All should point to the same handler
        assert expanded['postload'] is handler
        assert expanded['postlet'] is handler
        assert expanded['postchain'] is handler
        assert expanded['postletbinding'] is handler
        assert expanded['postcall'] is handler

        # Should not expand to pre* hooks
        assert 'preload' not in expanded
        assert 'prelet' not in expanded

    def test_scope_shortcuts_expand_to_pre_and_post(self):
        """Test that scope shortcuts (load, let, chain, binding, call) expand to both pre and post."""
        def handler(ctx): pass

        # Test 'load'
        policy_load = {'load': handler}
        expanded_load = expand_policy(policy_load)
        assert 'preload' in expanded_load
        assert 'postload' in expanded_load
        assert expanded_load['preload'] is handler
        assert expanded_load['postload'] is handler

        # Test 'let'
        policy_let = {'let': handler}
        expanded_let = expand_policy(policy_let)
        assert 'prelet' in expanded_let
        assert 'postlet' in expanded_let

        # Test 'chain'
        policy_chain = {'chain': handler}
        expanded_chain = expand_policy(policy_chain)
        assert 'prechain' in expanded_chain
        assert 'postchain' in expanded_chain

        # Test 'binding'
        policy_binding = {'binding': handler}
        expanded_binding = expand_policy(policy_binding)
        assert 'preletbinding' in expanded_binding
        assert 'postletbinding' in expanded_binding

        # Test 'call'
        policy_call = {'call': handler}
        expanded_call = expand_policy(policy_call)
        assert 'precall' in expanded_call
        assert 'postcall' in expanded_call

    def test_empty_policy_returns_empty_dict(self):
        """Test that empty policy returns empty dict."""
        assert expand_policy({}) == {}
        assert expand_policy(None) == {}  # type: ignore

    def test_full_hook_names_work_as_specific_overrides(self):
        """Test that full hook names work and override shortcuts."""
        def handler(ctx): pass

        policy = {'preload': handler, 'postcall': handler}
        expanded = expand_policy(policy)

        # Full names work because they're checked as specific_key in expansion
        assert 'preload' in expanded
        assert 'postcall' in expanded
        assert expanded['preload'] is handler
        assert expanded['postcall'] is handler

    def test_unknown_keys_ignored(self):
        """Test that unknown keys are silently ignored."""
        def handler(ctx): pass

        policy = {'unknown_key': handler, 'invalid': handler}
        expanded = expand_policy(policy)

        # Should return empty dict (no valid shortcuts)
        assert expanded == {}


class TestComposition:
    """Test composition when multiple shortcuts apply to same hook."""

    def test_general_and_scope_compose_at_pre_hooks(self):
        """Test that 'pre' and 'call' compose at precall."""
        call_order = []

        def pre_handler(ctx):
            call_order.append('pre')

        def call_handler(ctx):
            call_order.append('call')

        policy = {'pre': pre_handler, 'call': call_handler}
        expanded = expand_policy(policy)

        # precall should have both handlers
        assert 'precall' in expanded
        expanded['precall']({})  # Call composed function

        # Should execute in order: general (pre) → scope (call)
        assert call_order == ['pre', 'call']

    def test_three_level_composition_at_precall(self):
        """Test that pre + call + precall all compose at precall."""
        call_order = []

        def pre_handler(ctx):
            call_order.append('pre')

        def call_handler(ctx):
            call_order.append('call')

        def precall_handler(ctx):
            call_order.append('precall')

        policy = {'pre': pre_handler, 'call': call_handler, 'precall': precall_handler}
        expanded = expand_policy(policy)

        # precall should have all three
        assert 'precall' in expanded
        expanded['precall']({})

        # Should execute: general → scope → specific
        assert call_order == ['pre', 'call', 'precall']

    def test_post_hooks_compose_in_reverse(self):
        """Test that post hooks execute in reverse (LIFO) order."""
        call_order = []

        def post_handler(ctx):
            call_order.append('post')

        def call_handler(ctx):
            call_order.append('call')

        policy = {'post': post_handler, 'call': call_handler}
        expanded = expand_policy(policy)

        # postcall should have both handlers
        assert 'postcall' in expanded
        expanded['postcall']({})

        # Should execute in LIFO order: scope (call) → general (post)
        assert call_order == ['call', 'post']

    def test_three_level_post_composition_reversed(self):
        """Test that post hooks with 3 levels execute in full reverse."""
        call_order = []

        def post_handler(ctx):
            call_order.append('post')

        def call_handler(ctx):
            call_order.append('call')

        def postcall_handler(ctx):
            call_order.append('postcall')

        policy = {'post': post_handler, 'call': call_handler, 'postcall': postcall_handler}
        expanded = expand_policy(policy)

        assert 'postcall' in expanded
        expanded['postcall']({})

        # Should execute: specific → scope → general
        assert call_order == ['postcall', 'call', 'post']

    def test_no_composition_when_only_one_applies(self):
        """Test that single handlers don't get composed unnecessarily."""
        def handler(ctx): pass

        policy = {'pre': handler}
        expanded = expand_policy(policy)

        # preload should just be the handler, not a composed function
        assert expanded['preload'] is handler


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_policy_with_none_value(self):
        """Test that None values are handled gracefully."""
        policy = {'pre': None}  # type: ignore
        expanded = expand_policy(policy)

        # Should expand, but calling the hook will fail
        assert 'preload' in expanded

    def test_empty_string_key(self):
        """Test that empty string keys don't cause issues."""
        def handler(ctx): pass

        policy = {'': handler}
        expanded = expand_policy(policy)

        # Empty string is not a valid shortcut
        assert expanded == {}

    def test_mixed_shortcuts_and_full_names(self):
        """Test behavior when shortcuts and full names are mixed."""
        call_order = []

        def handler1(ctx):
            call_order.append('handler1')

        def handler2(ctx):
            call_order.append('handler2')

        # This policy has both shortcut and full name for preload
        # They should compose: 'pre' (general) + 'preload' (specific)
        policy = {'pre': handler1, 'preload': handler2}
        expanded = expand_policy(policy)

        # Both should be composed
        assert 'preload' in expanded
        call_order.clear()
        expanded['preload']({})
        assert call_order == ['handler1', 'handler2']  # general → specific

    def test_idempotency(self):
        """Test that calling expand_policy multiple times is safe."""
        def handler(ctx): pass

        policy = {'pre': handler, 'post': handler}
        expanded1 = expand_policy(policy)
        expanded2 = expand_policy(policy)

        # Should produce same keys
        assert set(expanded1.keys()) == set(expanded2.keys())

    def test_policy_with_callable_class(self):
        """Test that callable classes work as handlers."""
        class CallableHandler:
            def __call__(self, ctx):
                pass

        handler = CallableHandler()
        policy = {'pre': handler}
        expanded = expand_policy(policy)

        assert 'preload' in expanded
        assert expanded['preload'] is handler


class TestDebugPolicy:
    """Test debug_policy() helper function."""

    def test_debug_policy_shows_expansion(self, capsys):
        """Test that debug_policy prints expansion info."""
        def auth(ctx): pass
        def rate_limit(ctx): pass

        policy = {'pre': auth, 'call': rate_limit}
        result = debug_policy(policy, verbose=True)

        # Check output was printed
        captured = capsys.readouterr()
        assert 'preload' in captured.out
        assert 'precall' in captured.out
        assert 'auth' in captured.out
        assert 'rate_limit' in captured.out

        # Check return value structure
        assert isinstance(result, dict)
        assert 'preload' in result
        assert 'precall' in result

        # Check tuple structure: (handler_name, source_key)
        assert result['preload'] == [('auth', 'pre')]
        assert result['precall'] == [('auth', 'pre'), ('rate_limit', 'call')]

    def test_debug_policy_shows_reversed_marker(self, capsys):
        """Test that debug_policy shows '← reversed' for post hooks."""
        def handler1(ctx): pass
        def handler2(ctx): pass

        policy = {'post': handler1, 'call': handler2}
        debug_policy(policy, verbose=True)

        captured = capsys.readouterr()
        # Should show reversed marker for postcall (has multiple handlers)
        assert '← reversed' in captured.out

    def test_debug_policy_verbose_false(self):
        """Test that verbose=False doesn't print."""
        def auth(ctx): pass

        policy = {'pre': auth}
        result = debug_policy(policy, verbose=False)

        # Should return data but not print
        assert isinstance(result, dict)
        assert 'preload' in result

    def test_debug_policy_empty_policy(self):
        """Test debug_policy with empty policy."""
        result = debug_policy({}, verbose=True)
        assert result == {}

    def test_debug_policy_shows_composition_order(self):
        """Test that debug_policy shows correct composition order."""
        def pre_handler(ctx): pass
        def post_handler(ctx): pass
        def call_handler(ctx): pass
        def precall_handler(ctx): pass
        def postcall_handler(ctx): pass

        policy = {
            'pre': pre_handler,
            'post': post_handler,
            'call': call_handler,
            'precall': precall_handler,
            'postcall': postcall_handler
        }
        result = debug_policy(policy, verbose=False)

        # precall should show all three in order: general → scope → specific
        assert len(result['precall']) == 3
        assert result['precall'][0] == ('pre_handler', 'pre')
        assert result['precall'][1] == ('call_handler', 'call')
        assert result['precall'][2] == ('precall_handler', 'precall')

        # postcall should show reversed order: specific → scope → general
        assert len(result['postcall']) == 3
        assert result['postcall'][0] == ('postcall_handler', 'postcall')  # Reversed!
        assert result['postcall'][1] == ('call_handler', 'call')
        assert result['postcall'][2] == ('post_handler', 'post')


class TestExceptionHandling:
    """Test exception handling in composed handlers."""

    def test_exception_in_first_handler_stops_execution(self):
        """Exception in first handler should prevent later handlers from running."""
        call_order = []

        def failing_handler(ctx):
            call_order.append('failing')
            raise ValueError("Handler failed")

        def second_handler(ctx):
            call_order.append('second')

        policy = {'pre': failing_handler, 'preload': second_handler}
        expanded = expand_policy(policy)

        # Call the composed function
        with pytest.raises(ValueError, match="Handler failed"):
            expanded['preload']({})

        # Only first handler should have been called
        assert call_order == ['failing']

    def test_policy_exception_in_composed_handler_has_clear_traceback(self):
        """PolicyException in composed handler should have clear traceback."""
        from graphistry.compute.gfql.policy import PolicyException

        def failing_policy(ctx):
            raise PolicyException(
                phase='preload',
                reason='Test policy failure',
                code=403
            )

        def second_handler(ctx):
            pass

        policy = {'pre': failing_policy, 'preload': second_handler}
        expanded = expand_policy(policy)

        # Should raise PolicyException with clear message
        with pytest.raises(PolicyException) as exc_info:
            expanded['preload']({})

        assert exc_info.value.reason == 'Test policy failure'
        assert exc_info.value.code == 403
        assert exc_info.value.phase == 'preload'

    def test_non_callable_handler_fails_at_runtime(self):
        """Non-callable handler should fail at runtime with clear error."""
        policy = {'pre': "not a function"}  # type: ignore
        expanded = expand_policy(policy)

        # Expansion succeeds (we don't validate callability)
        assert 'preload' in expanded

        # But calling the handler should fail
        with pytest.raises(TypeError):
            expanded['preload']({})


class TestRealWorldPatterns:
    """Test real-world usage patterns."""

    def test_opentelemetry_pattern(self):
        """Test typical OpenTelemetry pattern with shortcuts."""
        def create_span(ctx): pass
        def end_span(ctx): pass

        policy = {'pre': create_span, 'post': end_span}
        expanded = expand_policy(policy)

        # Should cover all 10 hooks
        assert len(expanded) == 10
        assert all(hook in expanded for hook in [
            'preload', 'prelet', 'prechain', 'preletbinding', 'precall',
            'postload', 'postlet', 'postchain', 'postletbinding', 'postcall'
        ])

    def test_server_multi_policy_pattern(self):
        """Test server's multi-policy pattern with composition."""
        call_order = []

        def create_span(ctx):
            call_order.append('trace')

        def end_span(ctx):
            call_order.append('trace_end')

        def check_size(ctx):
            call_order.append('size')

        def rate_limit(ctx):
            call_order.append('rate')

        policy = {
            'pre': create_span,
            'post': end_span,
            'postload': check_size,
            'precall': rate_limit
        }
        expanded = expand_policy(policy)

        # Test precall composition (trace + rate)
        call_order.clear()
        expanded['precall']({})
        assert call_order == ['trace', 'rate']

        # Test postload composition (post + postload)
        # postload is checked as specific key, so both 'post' and 'postload' apply
        # For post hooks, they execute in reverse: specific → general
        call_order.clear()
        expanded['postload']({})
        assert call_order == ['size', 'trace_end']  # Reversed: postload → post

    def test_selective_override_pattern(self):
        """Test selective override of specific hooks while using shortcuts."""
        def default_pre(ctx): pass
        def default_post(ctx): pass
        def special_precall(ctx): pass

        # Use shortcuts for most, specific override for precall
        policy = {
            'pre': default_pre,
            'post': default_post,
            'precall': special_precall  # Override just precall
        }
        expanded = expand_policy(policy)

        # precall should compose default_pre + special_precall
        assert 'precall' in expanded

        # Other pre* hooks should just have default_pre
        assert expanded['preload'] is default_pre
        assert expanded['prelet'] is default_pre
