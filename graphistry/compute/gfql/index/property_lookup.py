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
from .property_keys import property_query_values, string_literals_are_utf8
from .registry import ColStatsRole, GfqlIndexRegistry, NodePropIndex


def _empty_gather_changes_scalar_filter(
    frame: "pl.DataFrame", filter_dict: Mapping[str, object],
) -> bool:
    """An empty eager frame can skip canonical scalar validation."""
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
        needs_original_frame |= not string_literals_are_utf8(resolved_value)
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
    single_polars_threshold: Optional[float] = None
    for column in sorted(registry.property_indexes(role)):
        if column not in filter_dict:
            continue
        index = registry.get_property_valid(role, column, frame, engine)
        if index is None and engine in POLARS_ENGINES:
            other = Engine.POLARS_GPU if engine == Engine.POLARS else Engine.POLARS
            index = registry.get_property_valid(role, column, frame, other)
        if index is None:
            continue
        # Exact native text scalars always have a defined dictionary lookup.
        # Other predicates must prove admission before cost configuration so
        # unsupported/coercing values retain canonical error ordering.
        native_text_scalar = (
            engine == Engine.POLARS and index.string_keys is not None
            and type(filter_dict[column]) is str
            and string_literals_are_utf8(filter_dict[column])
        )
        values = None if native_text_scalar else property_query_values(index, filter_dict[column], xp)
        if values is None and not native_text_scalar:
            continue
        if engine == Engine.POLARS and len(filter_dict) == 1 and policy != "force":
            # Polars uses the same crossover for every property encoding. If
            # even the smallest bucket is dense, canonical scanning avoids a
            # redundant probe. Tracing still costs the actual requested key;
            # force and multi-predicate selection keep their existing behavior.
            single_polars_threshold = cost_gate_frac(engine) * len(frame)
            if (index.min_group_count > 0
                    and index.min_group_count >= single_polars_threshold and not _trace_active()):
                return None
        if values is None:
            values = property_query_values(index, filter_dict[column], xp)
            if values is None:
                continue
        count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
    if best is None:
        return None
    column, index, values, count = best
    use_index = policy == "force" or count < (
        single_polars_threshold if single_polars_threshold is not None
        else cost_gate_frac(engine, kind=None if index.string_keys is not None else "node_prop" if role == "nodes" else "edge_prop") * len(frame)
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
                if eager.height > 0 and _empty_gather_changes_scalar_filter(eager, filter_dict):
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
    if record_decision and _trace_active():
        _record({
            "op": "property_lookup", "role": role, "column": column,
            "engine": engine.value, "policy": policy, "est_result_rows": count,
            "path": "index" if use_index else "scan",
            "decision_code": "index_path_unavailable" if semantic_decline else "index_selected" if use_index else "scan_cost",
            "decision_reason": "empty property candidates would change canonical scalar filtering" if semantic_decline else "property candidates gathered" if use_index else "property gather cost exceeds scan",
        })
    if not use_index:
        return None
    return xp.sort(lookup_prop_rows(index, values, xp))


def property_candidate_frame(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> DataFrameT:
    """Candidate frame in original row order, or the original frame on a decline."""
    # A live index proves that every stored bucket is beyond the scan crossover.
    # For an exact native scalar, scan the original columns directly. Keep off,
    # force, tracing, stale indexes, and coercing predicates on their usual path.
    registry, policy = get_registry(g), get_index_policy(g)
    if policy == "off" or not filter_dict or not registry.property_indexes(role) and not _trace_active():
        return frame
    if engine == Engine.POLARS and filter_dict and len(filter_dict) == 1 and policy == "use" and registry.property_indexes(role) and not _trace_active():
        from graphistry.compute.filter_by_dict import _filter_native_property_scalar, _supports_native_property_scalar
        column, value = next(iter(filter_dict.items()))
        # Reserve this shortcut for repeated buckets. Singleton dictionaries
        # retain canonical selection, avoiding duplicate validation and dtype
        # dispatch on selective lookups; small frames keep the same scan result.
        stored = registry.property_indexes(role).get(column)
        if stored is not None and stored.min_group_count > 1:
            index = registry.get_property_valid(role, column, frame, engine)
            if index is None:
                index = registry.get_property_valid(role, column, frame, Engine.POLARS_GPU)
            if index is not None and _supports_native_property_scalar(frame, column, value):
                if index.min_group_count >= cost_gate_frac(engine) * len(frame):
                    return _filter_native_property_scalar(frame, column, value)
    positions = property_candidate_positions(g, role, frame, filter_dict, engine)
    return frame if positions is None else take_rows(frame, positions, engine)
