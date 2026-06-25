"""Native Polars execution engine for GFQL (Phase 1: chain/hop traversal).

Dedicated, dispatched implementation (see plans/gfql-polars-engine): keeps the
production pandas/cuDF hop/chain internals untouched. Correctness is gated by
differential parity tests (pandas == polars).
"""

from .hop import hop_polars
from .chain import chain_polars

__all__ = ["hop_polars", "chain_polars"]
