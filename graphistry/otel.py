from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable


def otel_traced(*_args: Any, **_kwargs: Any):
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        return func

    return decorator


def otel_detail_enabled() -> bool:
    return False


def otel_enabled() -> bool:
    return False


@contextmanager
def otel_span(*_args: Any, **_kwargs: Any):
    yield
