"""Optional OpenTelemetry helpers for Graphistry."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Iterator, Optional, Tuple
import os
import sys

_OTEL_ENV = "GRAPHISTRY_OTEL"
_OTEL_DETAIL_ENV = "GRAPHISTRY_OTEL_DETAIL"

_otel_enabled_override: Optional[bool] = None
_otel_detail_override: Optional[bool] = None


def _env_enabled(name: str) -> bool:
    value = os.environ.get(name, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def otel_enabled() -> bool:
    if _otel_enabled_override is not None:
        return _otel_enabled_override
    return _env_enabled(_OTEL_ENV)


def otel_detail_enabled() -> bool:
    if _otel_detail_override is not None:
        return _otel_detail_override
    return _env_enabled(_OTEL_DETAIL_ENV)


def otel(
    enabled: Optional[bool] = None,
    detail: Optional[bool] = None,
    reset: bool = False,
) -> Tuple[bool, bool]:
    """Get/set OpenTelemetry enablement for Graphistry spans."""
    global _otel_enabled_override, _otel_detail_override
    if reset:
        _otel_enabled_override = None
        _otel_detail_override = None
    if enabled is not None:
        _otel_enabled_override = bool(enabled)
    if detail is not None:
        _otel_detail_override = bool(detail)
    return otel_enabled(), otel_detail_enabled()


def _get_tracer() -> Optional[Any]:
    if not otel_enabled():
        return None
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:
        return None
    return trace.get_tracer("graphistry")


@contextmanager
def otel_span(name: str, attrs: Optional[Dict[str, Any]] = None) -> Iterator[Optional[Any]]:
    """Create an OpenTelemetry span if tracing is enabled."""
    tracer = _get_tracer()
    if tracer is None:
        yield None
        return
    with tracer.start_as_current_span(name) as span:
        if attrs:
            for key, value in attrs.items():
                try:
                    span.set_attribute(key, value)
                except Exception:
                    continue
        yield span


class OTelScope:
    def __init__(self, name: str, attrs: Optional[Dict[str, Any]] = None) -> None:
        self._cm = otel_span(name, attrs=attrs)
        self.span = self._cm.__enter__()

    def close(self) -> None:
        exc_type, exc_val, exc_tb = sys.exc_info()
        self._cm.__exit__(exc_type, exc_val, exc_tb)


def otel_scope(name: str, attrs: Optional[Dict[str, Any]] = None) -> OTelScope:
    return OTelScope(name, attrs=attrs)


def otel_traced(
    name: str,
    attrs_fn: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator for wrapping a function in an optional OTel span."""
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs = attrs_fn(*args, **kwargs) if attrs_fn and otel_enabled() else None
            with otel_span(name, attrs=attrs):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def inject_trace_headers(headers: Dict[str, str]) -> Dict[str, str]:
    """Inject W3C trace context headers into an outgoing request."""
    if not otel_enabled():
        return headers
    try:
        from opentelemetry.propagate import inject  # type: ignore
    except Exception:
        return headers
    try:
        inject(headers)
    except Exception:
        return headers
    return headers
