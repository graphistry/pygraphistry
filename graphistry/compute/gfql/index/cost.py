"""Planner cost helpers for GFQL physical indexes."""
from __future__ import annotations

from typing import Any, Optional

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


def cost_gate_frac(engine: Engine) -> float:
    """Return the index-vs-scan crossover fraction for an engine."""
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
