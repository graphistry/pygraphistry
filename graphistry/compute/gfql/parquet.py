"""Options and errors for the opt-in CPU Parquet hop path."""

from dataclasses import dataclass
from math import isfinite
from threading import Event
from typing import Optional


class ParquetHopLimitError(RuntimeError):
    """A Parquet hop reached a resource limit before it could return a complete result."""


@dataclass(frozen=True)
class ParquetHopLimits:
    """Cooperative limits for a finite CPU hop over immutable local Parquet scans.

    max_nodes counts input seed rows before deduplication, node matches per lookup
    and retained node identities. max_edges counts candidates per hop before
    destination validation and retained edge identities. A scan may return one
    additional row to detect overflow. max_result_bytes caps the sum of Polars'
    estimated sizes for the materialized output frames. It is checked after each
    property collection. Timeout and cancellation are checked at operation
    boundaries. These limits do not cap process RSS or interrupt a running scan.

    Pass direct local Parquet scans with consistent schemas and immutable files
    throughout the call. GFQL accepts LazyFrames and does not verify provenance.
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
