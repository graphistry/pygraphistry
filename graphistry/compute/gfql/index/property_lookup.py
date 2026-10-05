"""Candidate gathers for property predicates; canonical filtering stays authoritative."""
from __future__ import annotations

from numbers import Integral
from typing import Mapping, Optional, Tuple

import numpy as np

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.Plottable import Plottable
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.typing import ArrayLike, DataFrameT
from .api import _record, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, take_rows
from .lookup import lookup_prop_rows, prop_match_count
from .registry import ColStatsRole, NodePropIndex


def property_candidate_positions(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> Optional[ArrayLike]:
    """Gather the most selective live property index, retaining input row order.

    Unsupported predicates, stale indexes, or expensive gathers leave the frame
    unchanged. Callers must apply the entire canonical filter to these candidates.
    """
    policy = get_index_policy(g)
    registry = get_registry(g)
    if policy == "off" or not filter_dict or not registry.property_indexes(role):
        return None
    xp, _ = array_namespace(engine)
    best: Optional[Tuple[str, NodePropIndex, ArrayLike, int]] = None
    for column in sorted(registry.property_indexes(role)):
        if column not in filter_dict:
            continue
        predicate = filter_dict[column]
        members = predicate.options if isinstance(predicate, IsIn) else (
            predicate if isinstance(predicate, (list, tuple)) else [predicate]
        )
        if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in members):
            continue
        index = registry.get_property_valid(role, column, frame, engine)
        if index is None and engine in POLARS_ENGINES:
            other = Engine.POLARS_GPU if engine == Engine.POLARS else Engine.POLARS
            index = registry.get_property_valid(role, column, frame, other)
        if index is None:
            continue
        bounds = np.iinfo(index.keys_sorted.dtype)
        # Bounds prove this cast is lossless, even for mixed signed/unsigned keys.
        values = xp.unique(xp.asarray(
            [int(value) for value in members if isinstance(value, Integral) and bounds.min <= int(value) <= bounds.max],
            dtype=index.keys_sorted.dtype,
        ))
        count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
    if best is None:
        return None
    column, index, values, count = best
    use_index = policy == "force" or count < cost_gate_frac(engine) * len(frame)
    _record({
        "op": "property_lookup", "role": role, "column": column,
        "engine": engine.value, "policy": policy, "est_result_rows": count,
        "path": "index" if use_index else "scan",
        "decision_code": "index_selected" if use_index else "scan_cost",
        "decision_reason": "property candidates gathered" if use_index else "property gather cost exceeds scan",
    })
    if not use_index:
        return None
    return xp.sort(lookup_prop_rows(index, values, xp))


def property_candidate_frame(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> DataFrameT:
    """Candidate frame in original row order, or the original frame on a decline."""
    positions = property_candidate_positions(g, role, frame, filter_dict, engine)
    return frame if positions is None else take_rows(frame, positions, engine)
