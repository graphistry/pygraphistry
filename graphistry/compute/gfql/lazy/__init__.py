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


def _engine_for(target: ExecutionTarget) -> Any:
    """Polars collect engine for a target. ``None`` = default (CPU streaming/in-mem).

    GPU uses ``raise_on_fail=False`` so any GPU-incapable node stays on CPU **in
    Polars** — NOT a pandas bridge (still honest/native; see NO-CHEATING)."""
    if target == ExecutionTarget.GPU:
        import polars as pl
        return pl.GPUEngine(raise_on_fail=False)
    return None


def collect(lf: Any) -> Any:
    """Collect one polars LazyFrame on the active target (CPU/GPU)."""
    eng = _engine_for(active_target())
    return lf.collect(engine=eng) if eng is not None else lf.collect()


def collect_all(lfs: List[Any]) -> List[Any]:
    """Collect several LazyFrames in ONE pass on the active target.

    Sharing the plan means common subplans (e.g. the edge table loaded once) are
    materialized/transferred a single time — the whole point of going lazy on GPU.
    Falls back to per-frame collect if the installed polars lacks ``collect_all``."""
    import polars as pl
    eng = _engine_for(active_target())
    if hasattr(pl, "collect_all"):
        try:
            return pl.collect_all(lfs, engine=eng) if eng is not None else pl.collect_all(lfs)
        except TypeError:
            # older signature without engine= — collect individually on target
            pass
    return [collect(lf) for lf in lfs]
