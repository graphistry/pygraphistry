from __future__ import annotations

from typing import Callable, cast

import pytest

from graphistry.compute.gfql.policy import (
    PolicyContext,
    PolicyException,
    debug_policy,
    expand_policy,
    format_policy_expansion,
)

Hook = Callable[[PolicyContext], None]

PRE_HOOKS = ("preload", "prelet", "prechain", "preletbinding", "precall")
POST_HOOKS = ("postload", "postlet", "postchain", "postletbinding", "postcall")
SCOPE_HOOKS = {
    "load": ("preload", "postload"),
    "let": ("prelet", "postlet"),
    "chain": ("prechain", "postchain"),
    "binding": ("preletbinding", "postletbinding"),
    "call": ("precall", "postcall"),
}


def _ctx() -> PolicyContext:
    return cast(PolicyContext, {})


def _handler(name: str, calls: list[str] | None = None) -> Hook:
    def record(ctx: PolicyContext) -> None:
        if calls is not None:
            calls.append(name)

    record.__name__ = name
    return record


@pytest.mark.parametrize(
    "shortcut,expected_hooks",
    [
        ("pre", PRE_HOOKS),
        ("post", POST_HOOKS),
        *SCOPE_HOOKS.items(),
    ],
)
def test_shortcuts_expand_to_expected_hooks(
    shortcut: str, expected_hooks: tuple[str, ...]
) -> None:
    handler = _handler("handler")
    expanded = expand_policy({shortcut: handler})

    assert set(expanded) == set(expected_hooks)
    assert all(expanded[hook] is handler for hook in expected_hooks)


def test_empty_unknown_and_invalid_shortcuts_are_ignored() -> None:
    handler = _handler("handler")

    assert expand_policy({}) == {}
    assert expand_policy(None) == {}  # type: ignore[arg-type]
    assert expand_policy({"": handler, "unknown": handler}) == {}


def test_direct_and_specific_hooks_survive_shortcut_expansion() -> None:
    handler = _handler("handler")
    expanded = expand_policy(
        {
            "pre": handler,
            "post": handler,
            "preload": handler,
            "postcall": handler,
            "precompile": handler,
            "postcompile": handler,
        }
    )

    assert set(expanded) == {*PRE_HOOKS, *POST_HOOKS, "precompile", "postcompile"}
    assert expanded["precompile"] is handler
    assert expanded["postcompile"] is handler
    assert all(hook in expanded for hook in ("preload", "postcall"))

def test_callable_object_handler_expands_by_identity() -> None:
    class CallableHandler:
        def __call__(self, ctx: PolicyContext) -> None:
            pass

    handler = CallableHandler()
    assert expand_policy({"pre": handler})["preload"] is handler

@pytest.mark.parametrize(
    "policy_keys,hook,expected_order",
    [
        (("pre", "call"), "precall", ["pre", "call"]),
        (("pre", "call", "precall"), "precall", ["pre", "call", "precall"]),
        (("post", "call"), "postcall", ["call", "post"]),
        (("post", "call", "postcall"), "postcall", ["postcall", "call", "post"]),
        (("pre", "preload"), "preload", ["pre", "preload"]),
        (("post", "postload"), "postload", ["postload", "post"]),
    ],
)
def test_composed_hooks_execute_in_policy_order(
    policy_keys: tuple[str, ...], hook: str, expected_order: list[str]
) -> None:
    calls: list[str] = []
    policy = {key: _handler(key, calls) for key in policy_keys}

    expanded = expand_policy(policy)
    expanded[hook](_ctx())

    assert calls == expected_order


def test_debug_policy_reports_sources_order_and_compile_hooks() -> None:
    policy = {
        "pre": _handler("auth"),
        "post": _handler("cleanup"),
        "call": _handler("rate_limit"),
        "precall": _handler("validate"),
        "postcall": _handler("audit"),
        "precompile": _handler("precompile_hook"),
        "postcompile": _handler("postcompile_hook"),
    }

    debug_info = debug_policy(policy)

    assert debug_policy({}) == {}
    assert debug_info["preload"] == [("auth", "pre")]
    assert debug_info["precall"] == [
        ("auth", "pre"),
        ("rate_limit", "call"),
        ("validate", "precall"),
    ]
    assert debug_info["postcall"] == [
        ("audit", "postcall"),
        ("rate_limit", "call"),
        ("cleanup", "post"),
    ]
    assert debug_info["precompile"] == [("precompile_hook", "precompile")]
    assert debug_info["postcompile"] == [("postcompile_hook", "postcompile")]

def test_format_policy_expansion_renders_human_readable_order() -> None:
    output = format_policy_expansion(
        {
            "pre": _handler("auth"),
            "post": _handler("cleanup"),
            "call": _handler("rate_limit"),
        }
    )

    assert isinstance(output, str)
    assert "preload" in output
    assert "auth" in output
    assert "postcall" in output
    assert "reversed" in output
    assert format_policy_expansion({}) == "Policy Expansion: (empty policy)"

def test_composed_hook_stops_after_first_exception() -> None:
    calls: list[str] = []

    def failing_handler(ctx: PolicyContext) -> None:
        calls.append("failing")
        raise ValueError("Handler failed")

    policy = {"pre": failing_handler, "preload": _handler("second", calls)}
    expanded = expand_policy(policy)

    with pytest.raises(ValueError, match="Handler failed"):
        expanded["preload"](_ctx())

    assert calls == ["failing"]


def test_policy_exception_from_composed_hook_keeps_structured_fields() -> None:
    def failing_policy(ctx: PolicyContext) -> None:
        raise PolicyException(phase="preload", reason="Test policy failure", code=403)

    expanded = expand_policy({"pre": failing_policy, "preload": _handler("second")})

    with pytest.raises(PolicyException) as exc_info:
        expanded["preload"](_ctx())

    assert exc_info.value.reason == "Test policy failure"
    assert exc_info.value.code == 403
    assert exc_info.value.phase == "preload"


def test_non_callable_handler_fails_when_executed() -> None:
    expanded = expand_policy({"pre": "not a function"})  # type: ignore[dict-item]

    with pytest.raises(TypeError):
        expanded["preload"](_ctx())


def test_common_tracing_and_selective_override_patterns() -> None:
    calls: list[str] = []
    policy = {
        "pre": _handler("trace_start", calls),
        "post": _handler("trace_end", calls),
        "postload": _handler("size", calls),
        "precall": _handler("rate", calls),
    }

    expanded = expand_policy(policy)

    assert set(expanded) == {*PRE_HOOKS, *POST_HOOKS}
    expanded["precall"](_ctx())
    assert calls == ["trace_start", "rate"]

    calls.clear()
    expanded["postload"](_ctx())
    assert calls == ["size", "trace_end"]
