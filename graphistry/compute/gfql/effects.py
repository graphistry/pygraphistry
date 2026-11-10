"""Execution-scoped ledger of side effects, so engine fallback never repeats them.

An ``engine='auto'`` attempt that declines with ``NotImplementedError`` re-runs the whole query
on another engine. Operations with external side effects (model provider calls) record
themselves here first; an attempt whose ledger grew raises instead of re-running.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, List, Optional

from typing_extensions import Literal

from graphistry.compute.exceptions import ErrorCode, GFQLCallExecutionError


_LEDGER: "ContextVar[Optional[List[str]]]" = ContextVar("gfql_effect_ledger", default=None)


@contextmanager
def effect_scope() -> Iterator[List[str]]:
    """Open an effect ledger for one engine attempt, reusing an enclosing one if present."""
    current = _LEDGER.get()
    if current is not None:
        yield current
        return
    ledger: List[str] = []
    token = _LEDGER.set(ledger)
    try:
        yield ledger
    finally:
        _LEDGER.reset(token)


def record_effect(name: str) -> None:
    """Record a side effect in the current ledger; a no-op outside any ``effect_scope``."""
    ledger = _LEDGER.get()
    if ledger is not None:
        ledger.append(name)


FallbackEngine = Literal["polars", "polars-gpu"]


def raise_if_effects_since(ledger: List[str], start: int, error: NotImplementedError, engine: FallbackEngine) -> None:
    """Refuse an engine fallback once the declined attempt has already caused side effects."""
    effects = ledger[start:]
    if not effects:
        return
    names = ", ".join(sorted(set(effects)))
    raise GFQLCallExecutionError(
        ErrorCode.E306,
        f"The query already ran side-effecting steps ({names}) before a step the {engine} engine cannot run, "
        "so engine='auto' will not re-run it on another engine",
        field="engine",
        value="auto",
        suggestion="Rerun with engine='pandas' (or 'cudf'), or remove the unsupported step",
        engine=engine,
        original_exception=type(error).__name__,
    ) from error
