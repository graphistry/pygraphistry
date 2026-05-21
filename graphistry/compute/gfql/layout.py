"""Helpers for detecting GFQL layout call chains."""

from typing import FrozenSet, Iterable, Mapping, Optional, Tuple

from graphistry.compute.gfql.call.validation import _LAYOUT_CALL_KINDS


LAYOUT_FUNCTION_NAMES: FrozenSet[str] = frozenset(_LAYOUT_CALL_KINDS)
LAYOUT_KINDS: FrozenSet[str] = frozenset(_LAYOUT_CALL_KINDS.values())
RADIAL_LAYOUT_KINDS: FrozenSet[str] = frozenset({
    'ring_continuous',
    'ring_categorical',
    'time_ring',
})
RADIAL_LAYOUT_FUNCTION_NAMES: FrozenSet[str] = frozenset(
    function
    for function, kind in _LAYOUT_CALL_KINDS.items()
    if kind in RADIAL_LAYOUT_KINDS
)


def _iter_chain_steps(chain: object) -> Iterable[object]:
    if isinstance(chain, str):
        return ()

    if isinstance(chain, Mapping):
        steps = chain.get('chain')
        if isinstance(steps, list):
            return steps
        return (chain,)

    if isinstance(chain, (list, tuple)):
        return chain

    steps = getattr(chain, 'chain', None)
    if isinstance(steps, list):
        return steps

    return (chain,)


def _step_function(step: object) -> Optional[str]:
    if isinstance(step, Mapping):
        function = step.get('function')
    else:
        function = getattr(step, 'function', None)
    return function if isinstance(function, str) else None


def _layout_call_kind(function: object) -> Optional[str]:
    """Return the layout kind for a GFQL ``call()`` function name, if any."""
    return _LAYOUT_CALL_KINDS.get(function) if isinstance(function, str) else None


def is_layout_kind(chain: object) -> Optional[str]:
    """Return the first layout kind in a GFQL chain, or ``None``.

    Accepts a ``Chain`` object, a list/tuple of GFQL steps, a Chain wire dict,
    or a direct ``Call`` object/dict. Opaque strings are not parsed.
    """
    pending: Tuple[object, ...] = tuple(_iter_chain_steps(chain))

    while pending:
        step, pending = pending[0], pending[1:]
        kind = _layout_call_kind(_step_function(step))
        if kind is not None:
            return kind

        if not isinstance(step, str) and (
            isinstance(step, Mapping) and isinstance(step.get('chain'), list)
            or isinstance(getattr(step, 'chain', None), list)
        ):
            pending = tuple(_iter_chain_steps(step)) + pending

    return None


def is_layout_chain(chain: object) -> bool:
    """Return whether any operation in a GFQL chain is a layout helper."""
    return is_layout_kind(chain) is not None
