"""Operator capability registry for logical operators."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decomposable(str, Enum):
    NONE = "none"
    PARTIAL = "partial"
    FULL = "full"


class Monotonicity(str, Enum):
    UNKNOWN = "unknown"
    INCREASING = "increasing"
    DECREASING = "decreasing"
    NON_MONOTONIC = "non_monotonic"


@dataclass(frozen=True)
class OpCapability:
    deterministic: bool = True
    side_effects: bool = False
    streaming: bool = True
    row_preserving: bool = True
    decomposable: Decomposable = Decomposable.NONE
    gpu_capable: bool = False
    js_transpilable: bool = False
    out_of_core_safe: bool = False
    graph_aware: bool = False
    graph_shape_change: bool = False
    supports_predicate_pushdown: bool = False
    monotonicity: Monotonicity = Monotonicity.UNKNOWN
