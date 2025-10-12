"""Policy shortcuts for reducing boilerplate in common patterns.

This module provides convenience shortcuts for policy hooks, enabling concise
policy definitions that expand to multiple hooks automatically.

Key Features:
    - Shortcuts: 'pre', 'post', and scope names ('load', 'let', 'chain', 'binding', 'call')
    - Automatic composition when multiple shortcuts apply to same hook
    - Predictable ordering: forward for pre hooks, reversed for post hooks (LIFO cleanup)
    - Debug helper for visibility into expansion

Example:
    >>> from graphistry.compute.gfql.policy import expand_policy
    >>>
    >>> # Concise OpenTelemetry policy
    >>> policy = {'pre': create_span, 'post': end_span}
    >>>
    >>> # Expands to all 10 hooks automatically
    >>> expanded = expand_policy(policy)
    >>> # -> {'preload': create_span, 'postload': end_span, 'prelet': create_span, ...}

Composition Example:
    >>> # Multiple shortcuts can compose at same hook
    >>> policy = {
    ...     'pre': auth_check,       # All pre* hooks
    ...     'call': rate_limit,      # precall + postcall
    ...     'precall': validate_op   # Just precall
    ... }
    >>>
    >>> # precall will run: auth_check → rate_limit → validate_op
    >>> # postcall will run: rate_limit → auth_check (reversed, LIFO)
"""

from typing import Dict, List, Tuple, cast
from .types import PolicyFunction, PolicyContext, PolicyDict, Phase, ShortcutKey, GeneralShortcut, ScopeShortcut

__all__ = ['expand_policy', 'debug_policy', 'format_policy_expansion', 'HandlerInfo', 'HookExpansionMap']

# Type aliases for debug output
HandlerInfo = Tuple[str, ShortcutKey]  # (handler_name, source_key) e.g., ('auth', 'pre')
HookExpansionMap = Dict[Phase, List[HandlerInfo]]  # hook_name -> list of handlers with sources

# Expansion mapping: hook_name → (general_key, scope_key, specific_key)
# Keys are Phase literals, values are tuples of ShortcutKey literals
_EXPANSION_MAP: Dict[Phase, Tuple[GeneralShortcut, ScopeShortcut, Phase]] = {
    'preload': ('pre', 'load', 'preload'),
    'postload': ('post', 'load', 'postload'),
    'prelet': ('pre', 'let', 'prelet'),
    'postlet': ('post', 'let', 'postlet'),
    'prechain': ('pre', 'chain', 'prechain'),
    'postchain': ('post', 'chain', 'postchain'),
    'preletbinding': ('pre', 'binding', 'preletbinding'),
    'postletbinding': ('post', 'binding', 'postletbinding'),
    'precall': ('pre', 'call', 'precall'),
    'postcall': ('post', 'call', 'postcall')
}


def expand_policy(policy: Dict[str, PolicyFunction]) -> PolicyDict:
    """Expand shorthand policy keys to full hook names with composition.

    This function transforms a policy dictionary with shortcuts (like 'pre', 'post')
    into a full policy dictionary with only concrete hook names (like 'preload',
    'postload', etc.).

    Shorthand Keys:
        'pre'  - Expands to all pre* hooks (preload, prelet, prechain, preletbinding, precall)
        'post' - Expands to all post* hooks (postload, postlet, postchain, postletbinding, postcall)
        'load' - Expands to both preload and postload
        'let' - Expands to both prelet and postlet
        'chain' - Expands to both prechain and postchain
        'binding' - Expands to both preletbinding and postletbinding
        'call' - Expands to both precall and postcall

    Composition Rules:
        When multiple shortcuts map to the same hook, their handlers compose:
        - Pre hooks: Execute in order from general → scope → specific
        - Post hooks: Execute in reverse order (specific → scope → general) for LIFO cleanup

    Override Behavior:
        Full hook names (like 'preload') always override shortcuts. This allows selective
        replacement while still using shortcuts for other hooks.

    Args:
        policy: Policy dictionary with shortcuts and/or full hook names (accepts any string keys)

    Returns:
        PolicyDict: Expanded policy dictionary with only Phase hook names (no shortcuts)

    Example:
        >>> policy = {'pre': auth, 'call': rate_limit, 'precall': validate}
        >>> expanded = expand_policy(policy)
        >>> # precall will run: auth → rate_limit → validate (composed)
        >>> # postcall will run: rate_limit → auth (reversed for LIFO)

    Note:
        This function is idempotent - calling it multiple times on the same
        policy (or on already-expanded policy) is safe and produces the same result.
    """
    if not policy:
        return {}

    expanded: PolicyDict = {}

    # Expand shortcuts to hooks with composition
    for hook_name, (general_key, scope_key, specific_key) in _EXPANSION_MAP.items():
        # Collect handlers from shortcuts in specificity order
        handlers = [policy[k] for k in [general_key, scope_key, specific_key] if k in policy]

        if not handlers:
            continue

        # For post hooks, reverse for LIFO cleanup order
        if hook_name.startswith('post'):
            handlers = handlers[::-1]

        # Single handler or compose multiple
        expanded[hook_name] = handlers[0] if len(handlers) == 1 else _compose(*handlers)

    return expanded


def _compose(*fns: PolicyFunction) -> PolicyFunction:
    """Compose multiple policy functions into a single function.

    Creates a new function that calls all provided functions in sequence,
    passing the same context to each.

    Args:
        *fns: Policy functions to compose

    Returns:
        Composed function that calls all fns in order
    """
    def composed(ctx: PolicyContext) -> None:
        for fn in fns:
            fn(ctx)
    return composed


def debug_policy(policy: Dict[str, PolicyFunction]) -> HookExpansionMap:
    """Analyze how shortcuts expand to hooks (returns data structure).

    This function helps understand how shortcuts will expand by returning which
    handlers will run for each hook, and in what order.

    Args:
        policy: Policy dict with shortcuts (before expansion)

    Returns:
        HookExpansionMap: Dict mapping hook names to list of HandlerInfo tuples.
        Each HandlerInfo is (handler_name, source_key) showing which handler
        came from which shortcut key.

    Example:
        >>> def auth(ctx): pass
        >>> def rate_limit(ctx): pass
        >>>
        >>> policy = {'pre': auth, 'call': rate_limit}
        >>> info = debug_policy(policy)
        >>> info['precall']
        [('auth', 'pre'), ('rate_limit', 'call')]
        >>>
        >>> # For formatted output, use format_policy_expansion()
        >>> print(format_policy_expansion(policy))

    Note:
        For human-readable output, use format_policy_expansion() instead.
        This function returns the raw data structure for programmatic access.
    """
    if not policy:
        return {}

    debug_info: HookExpansionMap = {}

    # _EXPANSION_MAP has Phase-typed keys, so hook_name is Phase
    for hook_name, (general_key, scope_key, specific_key) in _EXPANSION_MAP.items():
        # Collect (source_key, handler) pairs in specificity order
        sources: List[Tuple[ShortcutKey, PolicyFunction]] = [
            (cast(ShortcutKey, k), policy[k])
            for k in [general_key, scope_key, specific_key]
            if k in policy
        ]

        if not sources:
            continue

        # For post hooks, reverse to show execution order
        if hook_name.startswith('post'):
            sources = sources[::-1]

        # Store handler name and source key - both are properly typed now
        debug_info[hook_name] = [(fn.__name__, key) for key, fn in sources]

    return debug_info


def format_policy_expansion(policy: Dict[str, PolicyFunction]) -> str:
    """Generate formatted string showing how shortcuts expand to hooks.

    This function provides a human-readable visualization of policy expansion,
    showing which handlers will run for each hook and in what order.

    Args:
        policy: Policy dict with shortcuts (before expansion)

    Returns:
        Multi-line string showing the expansion

    Example:
        >>> def auth(ctx): pass
        >>> def rate_limit(ctx): pass
        >>>
        >>> policy = {'pre': auth, 'call': rate_limit}
        >>> print(format_policy_expansion(policy))
        preload         [auth (from 'pre')]
        prelet          [auth (from 'pre')]
        prechain        [auth (from 'pre')]
        preletbinding   [auth (from 'pre')]
        precall         [auth (from 'pre'), rate_limit (from 'call')]
        postcall        [rate_limit (from 'call'), auth (from 'pre')] ← reversed
        postload        [auth (from 'pre')]
        postlet         [auth (from 'pre')]
        postchain       [auth (from 'pre')]
        postletbinding  [auth (from 'pre')]

    Note:
        The "← reversed" marker indicates that post hooks execute in LIFO order
        (like try/finally blocks) for proper cleanup semantics.
    """
    debug_info = debug_policy(policy)

    if not debug_info:
        return "Policy Expansion: (empty policy)"

    lines = []
    for hook_name, handlers in debug_info.items():
        handlers_str = ', '.join([f"{fn_name} (from '{key}')" for fn_name, key in handlers])
        reverse_marker = " ← reversed" if hook_name.startswith('post') and len(handlers) > 1 else ""
        lines.append(f"{hook_name:15} [{handlers_str}]{reverse_marker}")

    return '\n'.join(lines)
