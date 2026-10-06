"""Candidate gathers for property predicates; canonical filtering stays authoritative."""
from __future__ import annotations

from numbers import Integral
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import polars as pl

import numpy as np

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.Plottable import Plottable
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.typing import ArrayLike, DataFrameT
from .api import _record, _trace_active, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, as_eager_polars_frame, take_rows
from .lookup import lookup_prop_rows, prop_match_count
from .registry import ColStatsRole, NodePropIndex


def _empty_gather_changes_temporal_filter(
    frame: "pl.DataFrame", filter_dict: Mapping[str, object],
) -> bool:
    """An empty eager frame skips canonical scalar temporal-string parsing."""
    from graphistry.compute.filter_by_dict import resolve_filter_column_or_absent
    from graphistry.compute.gfql.lazy.engine.polars.predicates import _dtype_is_temporal
    from graphistry.compute.gfql.strictness import absent_column_matches

    needs_original_frame = False
    schema = frame.schema
    for column, value in filter_dict.items():
        resolved = resolve_filter_column_or_absent(frame, column, value)
        if resolved is None:
            if not absent_column_matches(value):
                return False  # Canonical filtering stops before later predicates.
            continue
        resolved_column, resolved_value = resolved
        needs_original_frame |= isinstance(resolved_value, str) and _dtype_is_temporal(schema.get(resolved_column))
    return needs_original_frame


def property_candidate_positions(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> Optional[ArrayLike]:
    """Gather the most selective live property index, retaining input row order.

    Unsupported predicates and stale indexes decline to canonical filtering.
    Callers must apply the entire canonical filter to these candidates.
    """
    policy = get_index_policy(g)
    registry = get_registry(g)
    if policy == "off" or not filter_dict or not registry.property_indexes(role):
        return None
    xp, _ = array_namespace(engine)
    best: Optional[Tuple[str, NodePropIndex, ArrayLike, int]] = None
    single_polars_threshold: Optional[float] = None
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
        if engine == Engine.POLARS and len(filter_dict) == 1 and policy != "force":
            # Polars uses the same crossover for every property encoding. If
            # even the smallest bucket is dense, canonical scanning avoids a
            # redundant probe. Tracing still costs the actual requested key;
            # force and multi-predicate selection keep their existing behavior.
            single_polars_threshold = cost_gate_frac(engine) * len(frame)
            if (index.min_group_count > 0
                    and index.min_group_count >= single_polars_threshold and not _trace_active()):
                return None
        count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
    if best is None:
        return None
    column, index, values, count = best
    use_index = policy == "force" or count < (
        single_polars_threshold if single_polars_threshold is not None
        else cost_gate_frac(engine, kind="node_prop" if role == "nodes" else "edge_prop") * len(frame)
    )
    semantic_decline = False
    if use_index:
        if engine in POLARS_ENGINES:
            # Nonempty gathers retain the canonical filter's schema and eager-height behavior.
            if count == 0:
                from graphistry.compute.gfql.lazy.engine.polars.predicates import filter_expr_by_dict_polars
                eager = as_eager_polars_frame(frame)
                if eager is None:
                    return None
                filter_expr_by_dict_polars(eager, dict(filter_dict))
                if eager.height > 0 and _empty_gather_changes_temporal_filter(eager, filter_dict):
                    use_index = False
                    semantic_decline = True
        else:
            from graphistry.compute.filter_by_dict import _prepare_filter_dict
            native_integer_scalar = (
                engine == Engine.PANDAS and len(filter_dict) == 1
                and type(filter_dict[column]) is int and frame[column].dtype.kind in "iu"
            )
            # A live native integer index already proves this single real column
            # and exact integer literal need no type/label rewriting validation.
            if not native_integer_scalar:
                _prepare_filter_dict(frame, filter_dict)
    if _trace_active():
        _record({
            "op": "property_lookup", "role": role, "column": column,
            "engine": engine.value, "policy": policy, "est_result_rows": count,
            "path": "index" if use_index else "scan",
            "decision_code": "index_path_unavailable" if semantic_decline else "index_selected" if use_index else "scan_cost",
            "decision_reason": "empty property candidates would change canonical temporal filtering" if semantic_decline else "property candidates gathered" if use_index else "property gather cost exceeds scan",
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
