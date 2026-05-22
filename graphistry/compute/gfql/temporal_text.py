from __future__ import annotations

from typing import Any

# Compatibility shim for the historical temporal_text import path.
# Domain-specific implementation lives under graphistry.compute.gfql.temporal.
from graphistry.compute.gfql.temporal import (
    constructors,
    durations,
    folding,
    rendering,
    truncation,
    values,
)

_EXPORT_MODULES = (constructors, values, durations, truncation, rendering, folding)
__all__ = tuple(
    name for module in _EXPORT_MODULES for name in vars(module) if not name.startswith("__")
) + ("parse_day_time_duration_nanoseconds",)


def parse_day_time_duration_nanoseconds(text: str) -> int | None:
    parsed = durations.parse_temporal_sort_duration_components(text)
    if parsed is None:
        return None
    month_shift, nanosecond_shift = parsed
    if month_shift != 0:
        return None
    return nanosecond_shift


def __getattr__(name: str) -> Any:
    for module in _EXPORT_MODULES:
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
