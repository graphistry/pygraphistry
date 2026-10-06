"""Candidate gathers for property predicates; canonical filtering stays authoritative."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import polars as pl

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.Plottable import Plottable
from graphistry.compute.typing import ArrayLike, DataFrameT
from .api import _record, _trace_active, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, as_eager_polars_frame, take_rows
from .lookup import lookup_prop_rows, prop_match_count
from .property_keys import property_query_values, uncovered_property_column, valid_property_index
from .registry import ColStatsRole, GfqlIndexRegistry, NodePropIndex
from .types import IndexKind


def _record_uncovered_property(
    role: ColStatsRole, column: str, index_kind: IndexKind, engine: Engine, policy: str,
) -> None:
    if _trace_active():
        _record({
            "op": "property_lookup", "role": role, "column": column,
            "index_kind": index_kind, "engine": engine.value, "policy": policy,
            "path": "scan", "decision_code": "not_index_coverable",
            "decision_reason": "property predicate has no supported index encoding",
        })


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
    return property_candidate_positions_from_registry(
        get_registry(g), role, frame, filter_dict, engine, get_index_policy(g),
        binding_column=g._node if role == "nodes" else None,
    )


def property_candidate_positions_from_registry(
    registry: GfqlIndexRegistry, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine, policy: str,
    *, record_decision: bool = True, binding_column: Optional[str] = None,
) -> Optional[ArrayLike]:
    """Shared selector for graph filtering and specialized node-seed consumers."""
    if policy == "off" or not filter_dict:
        return None
    index_kind: IndexKind = "node_prop" if role == "nodes" else "edge_prop"
    indexes = registry.property_indexes(role)
    if not indexes and not (record_decision and _trace_active()):
        return None
    if not indexes:
        if record_decision and _trace_active():
            column = uncovered_property_column(frame, filter_dict, engine, binding_column=binding_column)
            if column is not None:
                _record_uncovered_property(role, column, index_kind, engine, policy)
        return None
    xp, _ = array_namespace(engine)
    best: Optional[Tuple[str, NodePropIndex, ArrayLike, int]] = None
    for column in sorted(indexes):
        if column not in filter_dict:
            continue
        index = valid_property_index(registry, role, frame, column, engine)
        if index is None:
            continue
        values = property_query_values(index, filter_dict[column], xp)
        if values is None:
            continue
        count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
    if best is None:
        if record_decision and _trace_active():
            column = uncovered_property_column(frame, filter_dict, engine, registry=registry, role=role, binding_column=binding_column)
            if column is not None:
                _record_uncovered_property(role, column, index_kind, engine, policy)
        return None
    column, index, values, count = best
    use_index = policy == "force" or count < cost_gate_frac(engine, kind=None if index.string_keys is not None else "node_prop" if role == "nodes" else "edge_prop") * len(frame)
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
            _prepare_filter_dict(frame, filter_dict)
    if record_decision and _trace_active():
        _record({
            "op": "property_lookup", "role": role, "column": column,
            "index_kind": index_kind, "engine": engine.value, "policy": policy, "est_result_rows": count,
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
