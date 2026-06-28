"""Lazy / deferred GFQL execution framework.

The lazy engines build a *deferred plan* and execute it once on a chosen
**execution target** (CPU / GPU now; remote backends later). This package is the
shared framework — the execution-target abstraction + collect-once helpers — that
each lazy backend (``lazy/engine/polars``, future ``lazy/engine/duckdb``) plugs
into. Per-backend *lowering* lives under ``lazy/engine/<backend>/``; only the
target/collect framework is shared here (the lowering itself is NOT shared across
backends — polars builds ``pl.LazyFrame``, DuckDB would build relations).

Design (from the GPU benchmark + architecture review):
- ``engine='polars'`` and ``engine='polars-gpu'`` are ONE lazy polars engine with
  two targets (CPU vs GPU ``.collect()``), not two engines.
- The win requires building ONE plan and collecting ONCE (transfer-once, fused) —
  per-op eager collect loses on GPU (repeated H2D). So lazy backends accumulate a
  plan and call :func:`collect` / :func:`collect_all` at the materialization
  boundary, on the active target.
"""
from __future__ import annotations

import contextvars
from enum import Enum
from typing import Any, List, Optional


class ExecutionTarget(Enum):
    CPU = "cpu"
    GPU = "gpu"  # polars GPU backend (cudf_polars)


_TARGET: "contextvars.ContextVar[ExecutionTarget]" = contextvars.ContextVar(
    "gfql_lazy_target", default=ExecutionTarget.CPU
)


def active_target() -> ExecutionTarget:
    return _TARGET.get()


class target_mode:
    """Context manager selecting the execution target for an enclosed lazy run."""

    def __init__(self, target: ExecutionTarget) -> None:
        self._target = target
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "target_mode":
        self._token = _TARGET.set(self._target)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _TARGET.reset(self._token)


import os as _os

# CPU collect engine. The polars STREAMING executor benchmarks faster + more stable
# than the default collect on big traversal joins (isolated 80M-edge 2-hop semijoin
# 1669→1040 ms, ~1.6×; end-to-end chain dilutes to ~1.04–1.11× as the forward/backward/
# combine overhead is unaffected), parity-identical. Opt-in (default off) because
# small/interactive sizes REGRESS (~0.86× at 100K) from streaming overhead.
_CPU_STREAMING = _os.environ.get("GFQL_POLARS_CPU_STREAMING", "0") == "1"


def _engine_for(target: ExecutionTarget) -> Any:
    """Polars collect engine for a target. ``None`` = default (CPU streaming/in-mem).

    GPU uses the cudf-polars IN-MEMORY executor (`executor="in-memory"`), not the
    default streaming `engine="gpu"` (`DefaultSingletonEngine`). GFQL results fit
    in device memory — the regime the in-memory engine is built for — and it is
    both FASTER (semijoin 1.33×, antijoin 2.58×, unique 1.49× @10M) and STABLE
    (the streaming executor spiked bimodally to ~1 s on the same semijoin; in-memory
    holds ~30 ms). `raise_on_fail=False` keeps any GPU-incapable node on CPU **in
    Polars** — NOT a pandas bridge (still honest/native; see NO-CHEATING). For
    larger-than-device-memory inputs the in-memory engine would OOM rather than
    stream — acceptable here (gfql graphs in scope fit), revisit if that changes."""
    if target == ExecutionTarget.GPU:
        import polars as pl
        return pl.GPUEngine(executor="in-memory", raise_on_fail=False)
    return None


def collect(lf: Any) -> Any:
    """Collect one polars LazyFrame on the active target (CPU/GPU)."""
    eng = _engine_for(active_target())
    if eng is not None:
        return lf.collect(engine=eng)
    return lf.collect(engine="streaming") if _CPU_STREAMING else lf.collect()


def collect_all(lfs: List[Any]) -> List[Any]:
    """Collect several LazyFrames in ONE pass on the active target.

    Sharing the plan means common subplans (e.g. the edge table loaded once) are
    materialized/transferred a single time — the whole point of going lazy on GPU.
    Falls back to per-frame collect if the installed polars lacks ``collect_all``."""
    import polars as pl
    eng = _engine_for(active_target())
    if hasattr(pl, "collect_all"):
        try:
            if eng is not None:
                return pl.collect_all(lfs, engine=eng)
            return pl.collect_all(lfs, engine="streaming") if _CPU_STREAMING else pl.collect_all(lfs)
        except TypeError:
            # older signature without engine= — collect individually on target
            pass
    return [collect(lf) for lf in lfs]
