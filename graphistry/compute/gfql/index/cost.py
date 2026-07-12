"""Planner cost helpers for GFQL physical indexes."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

from graphistry.Engine import Engine

# Index-vs-scan crossover fraction (of distinct source keys): past this frontier a
# full scan is cheaper than per-seed index probes, so `use` falls back to scan and
# never loses to the un-indexed path. Engine-aware because vectorized-scan engines
# (polars/cudf/GPU) scan far faster than pandas, so their crossover is much smaller
# (see test_index_cost_gate_engine_aware). Measured N=1e5 deg8: pandas ~0.5,
# polars ~0.02. GPU values provisional (dgx-gated) and conservatively grouped with
# polars.
_COST_GATE_FRAC = {Engine.PANDAS: 0.5}
_COST_GATE_FRAC_DEFAULT = 0.02
_COST_GATE_FRAC_OVERRIDES: Dict[Engine, float] = {}

# Absolute small-frontier floor: at or below this many seed rows the planner NEVER
# bails to scan on the frac gate. The frac gate scales with distinct-key cardinality,
# so on small / low-cardinality edge slices (e.g. per-edge-type homogeneous frames:
# n_keys <= 1/0.02 = 50 at the polars/cudf frac) even a single-node seed trips it and
# the hop scans O(E) despite a resident O(degree) index. A frontier of <= K seeds
# costs at most K searchsorted probes plus a gather of exactly the rows the scan
# would emit, so the index cannot meaningfully lose there — the gate's bulk-hop
# protection only matters once the frontier is genuinely large. Constant (not a
# function of n_keys) on purpose: scaling with n_keys is the frac gate's job; the
# floor bounds absolute per-hop probe work in the regime where frac*n_keys collapses
# below a handful of seeds. Uniform across engines (pandas's 0.5 frac only overlaps
# the floor when n_keys <= 2*K — micro slices where either path is trivially fast).
# Purely a routing heuristic: index and scan return identical results.
_COST_GATE_MIN_FRONTIER_DEFAULT = 16


def cost_gate_min_frontier() -> int:
    """Return the absolute small-frontier floor for the index-vs-scan cost gate.

    Frontiers of at most this many seeds always take a resident index, regardless
    of the frac gate. ``0`` disables the floor (pure frac gating). Env-overridable
    via ``GFQL_INDEX_COST_GATE_MIN_FRONTIER`` for benchmark/diagnostic tuning.
    """
    raw = os.environ.get("GFQL_INDEX_COST_GATE_MIN_FRONTIER")
    if raw is None or raw == "":
        return _COST_GATE_MIN_FRONTIER_DEFAULT
    try:
        val = int(raw)
    except ValueError as ex:
        raise ValueError(
            f"Invalid GFQL_INDEX_COST_GATE_MIN_FRONTIER={raw!r}: "
            f"expected a non-negative integer"
        ) from ex
    if val < 0:
        raise ValueError(
            f"Invalid GFQL_INDEX_COST_GATE_MIN_FRONTIER={raw!r}: "
            f"expected a non-negative integer"
        )
    return val


def _validate_cost_gate_frac(frac: float) -> float:
    if not 0.0 < frac <= 1.0:
        raise ValueError(f"GFQL index cost gate fraction must be in (0, 1], got {frac!r}")
    return frac


def set_cost_gate_frac(engine: Engine, frac: Optional[float]) -> None:
    """Override the index-vs-scan crossover for one process.

    Passing ``None`` clears the code-level override for ``engine``. This is meant
    for benchmark/diagnostic tuning; defaults remain stable and documented.
    """
    if frac is None:
        _COST_GATE_FRAC_OVERRIDES.pop(engine, None)
        return
    _COST_GATE_FRAC_OVERRIDES[engine] = _validate_cost_gate_frac(float(frac))


def reset_cost_gate_frac(engine: Optional[Engine] = None) -> None:
    """Clear code-level cost-gate overrides."""
    if engine is None:
        _COST_GATE_FRAC_OVERRIDES.clear()
    else:
        _COST_GATE_FRAC_OVERRIDES.pop(engine, None)


def _env_cost_gate_frac(engine: Engine) -> Optional[float]:
    names = (
        f"GFQL_INDEX_COST_GATE_FRAC_{engine.value.upper().replace('-', '_')}",
        "GFQL_INDEX_COST_GATE_FRAC",
    )
    for name in names:
        raw = os.environ.get(name)
        if raw is None or raw == "":
            continue
        try:
            return _validate_cost_gate_frac(float(raw))
        except ValueError as ex:
            raise ValueError(f"Invalid {name}={raw!r}: {ex}") from ex
    return None


def cost_gate_frac(engine: Engine) -> float:
    """Return the index-vs-scan crossover fraction for an engine.

    Precedence: code override via ``set_cost_gate_frac`` > engine/global env var
    > baked default. Env vars are ``GFQL_INDEX_COST_GATE_FRAC`` and per-engine
    ``GFQL_INDEX_COST_GATE_FRAC_PANDAS`` / ``..._POLARS_GPU`` style names.
    """
    if engine in _COST_GATE_FRAC_OVERRIDES:
        return _COST_GATE_FRAC_OVERRIDES[engine]
    env = _env_cost_gate_frac(engine)
    if env is not None:
        return env
    return _COST_GATE_FRAC.get(engine, _COST_GATE_FRAC_DEFAULT)


def seed_id_array(nodes: Any, node_col: str) -> Optional[Any]:
    """Seed ids of the frontier as a host numpy array.

    Device-to-host copy is fine here because this is only used for diagnostic
    planner-cost reporting under ``gfql_explain``. Returns ``None`` on any failure.
    """
    try:
        col = nodes[node_col]
        if hasattr(col, "to_numpy"):
            return col.to_numpy()
        arr = getattr(col, "values", col)
        return arr.get() if hasattr(arr, "get") else arr
    except Exception:
        return None


def seed_deg_sum(idx: Any, seed_ids: Any) -> Optional[int]:
    """Sum resident seed degree from CSR offsets for explain diagnostics."""
    try:
        import numpy as np
        ks = idx.keys_sorted
        ks = ks.get() if hasattr(ks, "get") else np.asarray(ks)
        off = idx.group_offsets
        off = off.get() if hasattr(off, "get") else np.asarray(off)
        seed_ids = np.asarray(seed_ids)
        pos = np.searchsorted(ks, seed_ids)
        valid = pos < ks.size
        pos_v = pos[valid]
        seed_v = seed_ids[valid]
        hit = pos_v[ks[pos_v] == seed_v]
        return int((off[hit + 1] - off[hit]).sum())
    except Exception:
        return None
