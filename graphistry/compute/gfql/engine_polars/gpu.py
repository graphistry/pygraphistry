"""GPU execution mode for the native Polars engine (``Engine.POLARS_GPU``).

The Polars engine is eager (operates on ``pl.DataFrame``). GPU mode runs the
SAME native ops but materializes the heavy ones on GPU via the cudf_polars
backend: ``LazyFrame.collect(engine=pl.GPUEngine(raise_on_fail=False))``.

``raise_on_fail=False`` keeps any GPU-incapable node on CPU **in Polars** — NOT a
pandas bridge (still honest, still native Polars; see plan.md NO-CHEATING).

GPU intent is carried by a context var set at the dispatch boundary (so the
engine internals don't need a ``gpu`` parameter threaded through every call).
When GPU is inactive, helpers run the ordinary eager path verbatim, so the
``engine='polars'`` (CPU) behavior is byte-for-byte unchanged.
"""
from __future__ import annotations

import contextvars
from typing import Any, Optional

_GPU_ACTIVE: "contextvars.ContextVar[bool]" = contextvars.ContextVar("gfql_polars_gpu_active", default=False)


def gpu_active() -> bool:
    return _GPU_ACTIVE.get()


class gpu_mode:
    """Context manager: activate GPU collection for the enclosed Polars engine run."""

    def __init__(self, active: bool = True) -> None:
        self._active = active
        self._token: Optional[contextvars.Token] = None

    def __enter__(self) -> "gpu_mode":
        self._token = _GPU_ACTIVE.set(self._active)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._token is not None:
            _GPU_ACTIVE.reset(self._token)


def _gpu_engine() -> Any:
    import polars as pl
    return pl.GPUEngine(raise_on_fail=False)


def collect(lf: Any) -> Any:
    """Collect a polars LazyFrame — on GPU when GPU mode is active, else CPU."""
    if gpu_active():
        return lf.collect(engine=_gpu_engine())
    return lf.collect()


def join(left: Any, right: Any, **kwargs: Any) -> Any:
    """Eager-signature join that runs on GPU (lazy + GPU collect) when active.

    When GPU is inactive this is exactly ``left.join(right, **kwargs)`` (eager),
    so CPU behavior is unchanged. ``left``/``right`` are eager ``pl.DataFrame``.
    """
    if gpu_active():
        return left.lazy().join(right.lazy(), **kwargs).collect(engine=_gpu_engine())
    return left.join(right, **kwargs)
