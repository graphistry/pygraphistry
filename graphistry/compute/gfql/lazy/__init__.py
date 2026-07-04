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
from typing import TYPE_CHECKING, Any, List, Optional
from typing_extensions import Literal

if TYPE_CHECKING:
    import polars as pl


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

# --- polars execution config -----------------------------------------------------
# Two knobs. Each resolves THREE ways, checked in order and LIVE (at call time, not
# frozen at import): (1) a Python override (set_*), None = unset; (2) the env var;
# (3) the default. So both env and Python settings take effect without a re-import.
# NOTE: these overrides are PROCESS-GLOBAL (plain module state), unlike the per-collect
# execution TARGET (_TARGET is a ContextVar because target_mode sets it transiently). They are
# meant as a global default set once at startup, not a per-request/per-thread override — a
# concurrent set_cpu_streaming() affects all in-flight GFQL queries in the process.
_cpu_streaming_override: Optional[bool] = None
_gpu_executor_override: "Optional[GpuExecutor]" = None

#: Valid cudf-polars GPU collect executors (public; see :func:`set_gpu_executor`).
GPU_EXECUTORS = ("in-memory", "streaming")
GpuExecutor = Literal["in-memory", "streaming"]


def cpu_streaming() -> bool:
    """Does the CPU polars collect use the STREAMING executor? (default off)

    The streaming executor benchmarks faster + more stable than the default collect on
    big traversal joins (isolated 80M-edge 2-hop semijoin 1669→1040 ms, ~1.6×; end-to-end
    chain dilutes to ~1.04–1.11× as forward/backward/combine overhead is unaffected),
    parity-identical. Opt-in (default off) because small/interactive sizes REGRESS
    (~0.86× at 100K) from streaming overhead.
    Resolution: :func:`set_cpu_streaming` override > ``$GFQL_POLARS_CPU_STREAMING`` (``"1"``) > ``False``.
    """
    if _cpu_streaming_override is not None:
        return _cpu_streaming_override
    return _os.environ.get("GFQL_POLARS_CPU_STREAMING", "0") == "1"


def set_cpu_streaming(value: Optional[bool]) -> None:
    """Enable/disable the CPU streaming-collect from Python. ``None`` resets to env/default."""
    global _cpu_streaming_override
    _cpu_streaming_override = None if value is None else bool(value)


def gpu_executor() -> GpuExecutor:
    """cudf-polars GPU collect executor: ``'in-memory'`` (default) or ``'streaming'``.

    'in-memory' is fast + stable for results that fit device memory (the GFQL regime —
    see :func:`_engine_for`). 'streaming' is the opt-in escape hatch for larger-than-device
    results (in-memory would OOM); slower/less stable on small work. These are the ONLY two
    accepted values (``GPU_EXECUTORS``); a size-aware per-query 'auto' is a possible future
    addition but is NOT selectable today (``set_gpu_executor('auto')`` raises).
    Resolution: :func:`set_gpu_executor` override > ``$GFQL_POLARS_GPU_EXECUTOR`` > ``'in-memory'``
    (an invalid env value also resolves to 'in-memory').
    """
    if _gpu_executor_override is not None:
        return _gpu_executor_override
    raw = _os.environ.get("GFQL_POLARS_GPU_EXECUTOR", "in-memory").strip().lower()
    return "streaming" if raw == "streaming" else "in-memory"


def set_gpu_executor(value: Optional[str]) -> None:
    """Select the GPU collect executor from Python (``'in-memory'`` | ``'streaming'``).

    ``None`` resets to env/default; an invalid value raises ``ValueError`` (the Python
    setter is strict, unlike the env path which falls back to 'in-memory')."""
    global _gpu_executor_override
    if value is None:
        _gpu_executor_override = None
        return
    v = value.strip().lower()
    if v not in GPU_EXECUTORS:
        raise ValueError(f"gpu_executor must be one of {GPU_EXECUTORS}, got {value!r}")
    _gpu_executor_override = "streaming" if v == "streaming" else "in-memory"


# --- off-engine call() modality policy -------------------------------------------
# A GFQL ``call()`` op may run a Plottable-method ANALYTIC (umap / hypergraph /
# compute_cugraph / compute_igraph / layout_* / collapse / ...) that has NO native
# polars implementation and never will — it runs eagerly on pandas/cuDF. Under
# ``engine='polars'|'polars-gpu'`` this knob decides what happens:
#   'auto'   (default): transparently BRIDGE — run the analytic off-engine on
#            pandas (polars) / cuDF (polars-gpu), coerce the result back to polars
#            losslessly (Arrow), warn once per (process, function).
#   'strict': DECLINE with NotImplementedError (the parity-or-NIE behavior) — for
#            benchmark integrity (no hidden modality switch) or a hard memory ceiling.
# This is DELIBERATELY distinct from CHAIN traversal/filter/row ops, which stay
# parity-or-NIE (a bridge there would hide a missing impl + cheat a benchmark). It
# mirrors the existing ``GRAPHISTRY_CUDF_SAME_PATH_MODE`` auto/strict precedent.
_call_mode_override: "Optional[CallMode]" = None

#: Valid off-engine call() modality modes (public; see :func:`set_call_mode`).
CALL_MODES = ("auto", "strict")
CallMode = Literal["auto", "strict"]


def call_mode() -> CallMode:
    """Off-engine ``call()`` analytic policy under a polars engine: ``'auto'`` | ``'strict'``.

    ``'auto'`` (default) bridges an analytic with no native polars impl (umap / hypergraph /
    compute_cugraph / ...) to pandas/cuDF, runs it, and coerces the result back to polars
    losslessly (warn once per function). ``'strict'`` declines with ``NotImplementedError``
    (no hidden modality switch — for benchmarking / a hard memory ceiling).
    Resolution: :func:`set_call_mode` override > ``$GFQL_POLARS_CALL_MODE`` > ``'auto'``
    (an invalid env value also resolves to 'auto').
    """
    if _call_mode_override is not None:
        return _call_mode_override
    raw = _os.environ.get("GFQL_POLARS_CALL_MODE", "auto").strip().lower()
    return "strict" if raw == "strict" else "auto"


def set_call_mode(value: Optional[str]) -> None:
    """Select the off-engine call() policy from Python (``'auto'`` | ``'strict'``).

    ``None`` resets to env/default; an invalid value raises ``ValueError`` (the Python
    setter is strict, unlike the env path which falls back to 'auto')."""
    global _call_mode_override
    if value is None:
        _call_mode_override = None
        return
    v = value.strip().lower()
    if v not in CALL_MODES:
        raise ValueError(f"call_mode must be one of {CALL_MODES}, got {value!r}")
    _call_mode_override = "strict" if v == "strict" else "auto"


# Public surface of the lazy framework. Explicit (vs a dedicated ``types.py``) because the
# quasi-public types (ExecutionTarget/GpuExecutor/CallMode + their tuples) are few and live
# co-located with their validators/accessors here; a per-engine types module is deferred until
# a 2nd lazy backend (duckdb) needs to share ExecutionTarget (mirrors the lazy-restructure defer).
__all__ = [
    "ExecutionTarget", "active_target", "target_mode", "collect", "collect_all",
    "GpuExecutor", "GPU_EXECUTORS", "gpu_executor", "set_gpu_executor",
    "CallMode", "CALL_MODES", "call_mode", "set_call_mode",
    "cpu_streaming", "set_cpu_streaming",
]


def _engine_for(target: ExecutionTarget) -> "Optional[pl.GPUEngine]":
    """Polars collect engine for a target. ``None`` = default (CPU streaming/in-mem).

    GPU uses the cudf-polars IN-MEMORY executor (`executor="in-memory"`), not the
    default streaming `engine="gpu"` (`DefaultSingletonEngine`). GFQL results fit
    in device memory — the regime the in-memory engine is built for — and it is
    both FASTER (semijoin 1.33×, antijoin 2.58×, unique 1.49× @10M) and STABLE
    (the streaming executor spiked bimodally to ~1 s on the same semijoin; in-memory
    holds ~30 ms). ``raise_on_fail=True`` is the NO-CHEATING contract for the GPU
    target: if any node of the plan is not GPU-executable we RAISE — we never
    silently run it on CPU and report it as a GPU result. (``raise_on_fail=False``
    looked honest — fallback stays *in Polars*, not a pandas bridge — but it makes
    ``engine='polars-gpu'`` indistinguishable from ``engine='polars'`` whenever the
    plan isn't fully GPU-capable, which silently mislabels CPU work as GPU. A hard
    raise forces the truth: ``polars-gpu`` is GPU-or-error; use ``polars`` for CPU.)
    For larger-than-device-memory inputs the in-memory engine would OOM rather than
    stream — acceptable here (gfql graphs in scope fit), revisit if that changes."""
    if target == ExecutionTarget.GPU:
        import polars as pl
        # (The RAPIDS/cudf_polars-not-installed check lives at the chain dispatch, pre-coercion,
        # so the user-facing engine='polars-gpu' always gets a clean install error there. Here we
        # only build the engine; a genuine not-GPU-capable plan is reported via _gpu_raise.)
        # Executor is in-memory by default; GFQL_POLARS_GPU_EXECUTOR=streaming opts into the
        # streaming executor for larger-than-device-memory results (see gpu_executor()).
        executor = gpu_executor()
        return pl.GPUEngine(executor=executor, raise_on_fail=True)
    return None


def _gpu_raise(exc: Exception) -> "NotImplementedError":
    """Translate a cudf-polars GPU-execution failure into the NO-CHEATING error.

    With ``raise_on_fail=True`` the GPU engine raises when a plan node is not
    GPU-executable. We surface that as a clear NotImplementedError (chained from the
    polars error) instead of a cryptic ComputeError, pointing at the CPU engine."""
    return NotImplementedError(
        "GFQL engine='polars-gpu': this query plan is not fully GPU-executable on the "
        "installed cudf-polars (NO-CHEATING: we raise rather than silently run it on CPU "
        "and call it a GPU result). Rerun with engine='polars' for native CPU execution. "
        f"Underlying GPU error: {type(exc).__name__}: {exc}"
    )


def collect(lf: "pl.LazyFrame") -> "pl.DataFrame":
    """Collect one polars LazyFrame on the active target (CPU/GPU)."""
    eng = _engine_for(active_target())
    if eng is not None:
        try:
            return lf.collect(engine=eng)  # pragma: no cover  # GPU-target collect (no GPU in CI)
        except NotImplementedError:
            raise
        except Exception as ex:  # pragma: no cover  # GPU-target collect (no GPU in CI)
            raise _gpu_raise(ex) from ex
    return lf.collect(engine="streaming") if cpu_streaming() else lf.collect()


def collect_all(lfs: "List[pl.LazyFrame]") -> "List[pl.DataFrame]":
    """Collect several LazyFrames in ONE pass on the active target.

    Sharing the plan means common subplans (e.g. the edge table loaded once) are
    materialized/transferred a single time — the whole point of going lazy on GPU.
    Falls back to per-frame collect if the installed polars lacks ``collect_all``."""
    import polars as pl
    eng = _engine_for(active_target())
    if hasattr(pl, "collect_all"):
        try:
            if eng is not None:
                return pl.collect_all(lfs, engine=eng)  # pragma: no cover  # GPU-target collect (no GPU in CI)
            return pl.collect_all(lfs, engine="streaming") if cpu_streaming() else pl.collect_all(lfs)
        except TypeError:
            # older signature without engine= — collect individually on target
            pass
        except NotImplementedError:  # pragma: no cover  # GPU-target collect (no GPU in CI)
            raise
        except Exception as ex:  # pragma: no cover  # GPU-target collect (no GPU in CI)
            if eng is not None:
                raise _gpu_raise(ex) from ex
            raise
    return [collect(lf) for lf in lfs]
