"""Options and errors for the opt-in CPU Parquet hop path."""

from dataclasses import dataclass
from math import isfinite
from threading import Event
from typing import Optional


class ParquetHopLimitError(RuntimeError):
    """A Parquet hop reached a resource limit before it could return a complete result."""


@dataclass(frozen=True)
class ParquetHopLimits:
    """Resource limits for a finite CPU hop over immutable local Parquet scans.

    The node limit also caps the number of input seeds. The edge limit caps both
    cumulative output edges and the number of candidate edges collected per hop.
    The byte limit is checked on materialized output frames. Timeout and cancellation
    are checked between Polars operations. These limits do not cap process RSS or
    interrupt a running Polars scan.
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
        if not isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
