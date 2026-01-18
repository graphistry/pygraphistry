"""Optional OpenTelemetry helpers for GFQL execution."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional
import os

_OTEL_ENV = "GRAPHISTRY_DF_EXECUTOR_OTEL"
_OTEL_DETAIL_ENV = "GRAPHISTRY_DF_EXECUTOR_OTEL_DETAIL"


def _otel_enabled() -> bool:
    value = os.environ.get(_OTEL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def otel_enabled() -> bool:
    return _otel_enabled()


def otel_detail_enabled() -> bool:
    value = os.environ.get(_OTEL_DETAIL_ENV, "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def _get_tracer() -> Optional[Any]:
    if not _otel_enabled():
        return None
    try:
        from opentelemetry import trace  # type: ignore
    except Exception:
        return None
    return trace.get_tracer("graphistry.gfql")


@contextmanager
def otel_span(name: str, attrs: Optional[Dict[str, Any]] = None) -> Iterator[Optional[Any]]:
    """Create an OpenTelemetry span if tracing is enabled.

    This is a no-op unless GRAPHISTRY_DF_EXECUTOR_OTEL is truthy and
    opentelemetry is installed.
    """
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
