"""Candidate gathers for property predicates; canonical filtering stays authoritative."""
from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Optional, Tuple

if TYPE_CHECKING:
    import polars as pl

import pandas as pd

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.Plottable import Plottable
from graphistry.compute.typing import ArrayLike, DataFrameT
from .api import _record, _trace_active, get_index_policy, get_registry
from .cost import cost_gate_frac
from .engine_arrays import array_namespace, as_eager_polars_frame, take_rows
from .lookup import _csr_hit_positions, _csr_single_group_bounds, lookup_prop_rows, prop_match_count
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
    best_scalar_rows: Optional[ArrayLike] = None
    best_groups: Optional[ArrayLike] = None
    best_group_sizes: Optional[ArrayLike] = None
    for column in sorted(registry.property_indexes(role)):
        if column not in filter_dict:
            continue
        index = registry.get_property_valid(role, column, frame, engine)
        if index is None and engine in POLARS_ENGINES:
            other = Engine.POLARS_GPU if engine == Engine.POLARS else Engine.POLARS
            index = registry.get_property_valid(role, column, frame, other)
        if index is None:
            continue
        # Unsupported literals must prove admission before cost configuration.
        native_text_scalar = (
            engine in (Engine.POLARS, Engine.CUDF) and index.string_keys is not None
            and type(filter_dict[column]) is str
            and string_literals_are_utf8(filter_dict[column])
        )
        values = None if native_text_scalar else property_query_values(index, filter_dict[column], xp)
        if values is None and not native_text_scalar:
            continue
        if engine in (Engine.POLARS, Engine.CUDF) and len(filter_dict) == 1 and policy != "force":
            # Tracing costs the requested key; force bypasses this density decline.
            single_polars_threshold = cost_gate_frac(engine) * len(frame)
            if (index.min_group_count > 0
                    and index.min_group_count >= single_polars_threshold and not _trace_active()):
                return None
        if values is None:
            values = property_query_values(index, filter_dict[column], xp)
            if values is None:
                continue
        scalar_rows = None
        groups: Optional[ArrayLike] = None
        group_sizes: Optional[ArrayLike] = None
        if engine == Engine.CUDF and int(values.shape[0]) <= 1:
            groups = values if index.string_keys is not None else _csr_hit_positions(index.keys_sorted, values, xp)
            if int(groups.shape[0]) == 0:
                scalar_rows = index.row_positions[:0]
            else:
                start, end = _csr_single_group_bounds(index, groups)
                scalar_rows = index.row_positions[start:end]
            count = int(scalar_rows.shape[0])
        elif engine == Engine.CUDF and index.string_keys is None:
            from .lookup import _csr_group_sizes
            groups = _csr_hit_positions(index.keys_sorted, values, xp)
            group_sizes = _csr_group_sizes(index, groups)
            count = int(group_sizes.sum())
        else:
            count = prop_match_count(index, values, xp)
        if best is None or count < best[3]:
            best = column, index, values, count
            best_scalar_rows = scalar_rows
            best_groups = groups if scalar_rows is None else None
            best_group_sizes = group_sizes
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
            native_scalar = (
                engine in (Engine.PANDAS, Engine.CUDF) and len(filter_dict) == 1
                and type(filter_dict[column]) is int and frame[column].dtype.kind in "iu"
            )
            if (not native_scalar and engine == Engine.PANDAS and len(filter_dict) == 1
                    and type(filter_dict[column]) is str and type(frame) is pd.DataFrame
                    and isinstance(frame[column].dtype, pd.CategoricalDtype)):
                array = frame[column].array
                native_scalar = type(array) is pd.Categorical and type(array.categories) is pd.Index
            if (not native_scalar and engine == Engine.CUDF and len(filter_dict) == 1
                    and type(filter_dict[column]) is str and index.string_keys is not None
                    and string_literals_are_utf8(filter_dict[column])):
                native_scalar = True
            # A live native index proves these exact literals need no label rewriting.
            if not native_scalar:
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
    rows = best_scalar_rows if best_scalar_rows is not None else (
        lookup_prop_rows(index, values, xp, group_positions=best_groups, group_sizes=best_group_sizes, match_count=count)
        if best_groups is not None else lookup_prop_rows(index, values, xp)
    )
    return rows if engine == Engine.CUDF and int(rows.shape[0]) <= 1 else xp.sort(rows)


def property_candidate_frame(
    g: Plottable, role: ColStatsRole, frame: DataFrameT,
    filter_dict: Optional[Mapping[str, object]], engine: Engine,
) -> DataFrameT:
    """Candidate frame in original row order, or the original frame on a decline."""
    # Off, force, tracing, stale indexes, and coercing literals retain canonical selection.
    registry, policy = get_registry(g), get_index_policy(g)
    if policy == "off" or not filter_dict or not registry.property_indexes(role) and not _trace_active():
        return frame
    if engine == Engine.POLARS and filter_dict and len(filter_dict) == 1 and policy == "use" and registry.property_indexes(role) and not _trace_active():
        from graphistry.compute.filter_by_dict import _filter_native_property_scalar, _supports_native_property_scalar
        column, value = next(iter(filter_dict.items()))
        # Singleton dictionaries retain canonical selection.
        stored = registry.property_indexes(role).get(column)
        if stored is not None and stored.min_group_count > 1 and as_eager_polars_frame(frame) is not None:
            try:
                dense_scan = stored.min_group_count >= cost_gate_frac(engine) * len(frame)
            except ValueError:
                dense_scan = False  # The selector validates cost only after admitting the literal.
            if dense_scan:
                index = registry.get_property_valid(role, column, frame, engine)
                if index is None:
                    index = registry.get_property_valid(role, column, frame, Engine.POLARS_GPU)
                if index is not None and _supports_native_property_scalar(frame, column, value):
                    return _filter_native_property_scalar(frame, column, value)
    positions = property_candidate_positions(g, role, frame, filter_dict, engine)
    return frame if positions is None else take_rows(frame, positions, engine)
