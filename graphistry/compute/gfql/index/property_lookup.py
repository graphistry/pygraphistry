"""Candidate gathers for property predicates; canonical filtering stays authoritative."""
from __future__ import annotations

from typing import Mapping, Optional, Tuple

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.Plottable import Plottable
from graphistry.compute.typing import ArrayLike, DataFrameT
from .api import _record, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, as_eager_polars_frame, take_rows
from .lookup import lookup_prop_rows, prop_match_count
from .property_keys import property_query_values
from .registry import ColStatsRole, GfqlIndexRegistry, NodePropIndex


def property_candidate_positions(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> Optional[ArrayLike]:
    """Gather the most selective live property index, retaining input row order.

    Unsupported predicates and stale indexes decline to canonical filtering.
    Callers must apply the entire canonical filter to these candidates.
    """
    return property_candidate_positions_from_registry(
        get_registry(g), role, frame, filter_dict, engine, get_index_policy(g),
    )


def property_candidate_positions_from_registry(
    registry: GfqlIndexRegistry, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine, policy: str,
    *, record_decision: bool = True,
) -> Optional[ArrayLike]:
    """Shared selector for graph filtering and specialized node-seed consumers."""
    if policy == "off" or not filter_dict or not registry.property_indexes(role):
        return None
    xp, _ = array_namespace(engine)
    best: Optional[Tuple[str, NodePropIndex, ArrayLike, int]] = None
    for column in sorted(registry.property_indexes(role)):
        if column not in filter_dict:
            continue
        index = registry.get_property_valid(role, column, frame, engine)
        if index is None and engine in POLARS_ENGINES:
            other = Engine.POLARS_GPU if engine == Engine.POLARS else Engine.POLARS
            index = registry.get_property_valid(role, column, frame, other)
        if index is None:
            continue
        values = property_query_values(index, filter_dict[column], xp)
        if values is None:
            continue
        count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
    if best is None:
        return None
    column, index, values, count = best
    use_index = policy == "force" or count < cost_gate_frac(engine) * len(frame)
    if use_index:
        if engine in POLARS_ENGINES:
            from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_expr_by_dict_polars
            eager = as_eager_polars_frame(frame)
            if eager is None:
                return None
            filter_expr_by_dict_polars(eager, dict(filter_dict))
        else:
            from graphistry.compute.filter_by_dict import _prepare_filter_dict
            _prepare_filter_dict(frame, filter_dict)
    if record_decision:
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
