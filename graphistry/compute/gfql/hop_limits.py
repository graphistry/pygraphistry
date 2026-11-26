"""Experimental resource limits for GFQL hop traversal."""

from dataclasses import dataclass
from math import isfinite
from threading import Event
from typing import Optional


class HopLimitError(RuntimeError):
    """A hop reached a resource limit before it could return a complete result."""


@dataclass(frozen=True)
class HopLimits:
    """Experimental cooperative limits for a finite hop.

    max_nodes counts input seed rows before deduplication, node matches per lookup
    and retained node identities. max_edges counts candidates per hop before
    destination validation and retained edge identities. A scan may return one
    additional row to detect overflow. max_result_bytes caps the sum of Polars'
    estimated sizes for the materialized output frames. It is checked after each
    property collection. Timeout and cancellation are checked at operation
    boundaries. These limits do not cap scan I/O or process RSS, or interrupt a
    running scan.

    Currently supported for CPU Polars with direct local Parquet scans. Pass
    consistent schemas and keep the files immutable throughout the call. GFQL
    accepts LazyFrames and does not verify provenance.
    """

    max_nodes: int = 100_000
    max_edges: int = 100_000
    max_result_bytes: int = 64 * 1024 * 1024
    max_hops: int = 16
    timeout_seconds: float = 30.0
    cancel_event: Optional[Event] = None

    def __post_init__(self) -> None:
        for name, limit in (
            ("max_nodes", self.max_nodes), ("max_edges", self.max_edges),
            ("max_result_bytes", self.max_result_bytes), ("max_hops", self.max_hops),
        ):
            if type(limit) is not int or limit < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.timeout_seconds, bool) or not isfinite(self.timeout_seconds)
                or self.timeout_seconds <= 0):
            raise ValueError("timeout_seconds must be positive and finite")
        if self.cancel_event is not None and not isinstance(self.cancel_event, Event):
            raise TypeError("cancel_event must be a threading.Event")
