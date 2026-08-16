"""openCypher ungrouped-aggregate identity row.

An aggregate with NO grouping keys always yields exactly one row, so a stage that
empties the row stream must still return the aggregate identities (count/sum -> 0,
collect -> [], min/max/avg -> null) rather than an empty frame. Every lowering path
that can end in an ungrouped aggregate feeds its compiled row steps through
``ungrouped_aggregate_identity_row`` here; the result becomes the compiled query's
``empty_result_row``, applied at runtime only when the real result is empty.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import pandas as pd

from graphistry.compute.ast import ASTCall, ASTObject


def aggregate_identity_value(func: str) -> Any:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    """openCypher aggregate identity over an empty row stream."""
    if func in {"count", "count_distinct", "sum"}:
        return 0
    if func in {"collect", "collect_distinct"}:
        return []
    return None


def identity_row_after_paging(
    row: Optional[Dict[str, Any]],  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    *,
    skip_value: Optional[int],
    limit_value: Optional[int],
) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    """Terminal SKIP/LIMIT pages the synthesized identity row like any other row:
    ``SKIP >= 1`` and ``LIMIT <= 0`` drop it, everything else keeps it."""
    if row is None:
        return None
    if skip_value is not None and skip_value >= 1:
        return None
    if limit_value is not None and limit_value <= 0:
        return None
    return row


_IDENTITY_ROW_REPLAY_CALLS = frozenset(
    {"select", "with_", "return_", "order_by", "skip", "limit", "distinct", "drop_cols", "group_by"}
)


def _ungrouped_aggregate_identity_seed(step: ASTObject) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    if not isinstance(step, ASTCall):
        return None
    if step.function == "count_table":
        alias = step.params.get("alias")
        return {alias: 0} if isinstance(alias, str) else None
    if step.function != "group_by" or step.params.get("key_prefixes"):
        return None
    keys = step.params.get("keys")
    if not (
        isinstance(keys, (list, tuple))
        and len(keys) == 1
        and isinstance(keys[0], str)
        and keys[0].startswith("__cypher_group__")
    ):
        return None
    aggregations = step.params.get("aggregations")
    if not isinstance(aggregations, (list, tuple)) or not aggregations:
        return None
    row: Dict[str, Any] = {keys[0]: 1}
    for aggregation in aggregations:
        if not (isinstance(aggregation, (list, tuple)) and len(aggregation) >= 2):
            return None
        output_name, runtime_func = aggregation[0], aggregation[1]
        if not (isinstance(output_name, str) and isinstance(runtime_func, str)):
            return None
        row[output_name] = aggregate_identity_value(runtime_func)
    return row


def _identity_row_passthrough_projection(
    row: Mapping[str, Any],  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    params: Mapping[str, Any],  # hygiene-ok: explicit-any -- untyped ASTCall params mapping
) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    items = params.get("items")
    if not isinstance(items, (list, tuple)):
        return None
    out: Dict[str, Any] = dict(row) if params.get("extend") else {}
    for entry in items:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 2):
            return None
        output_name, source = entry
        if not (isinstance(output_name, str) and isinstance(source, str) and source in row):
            return None
        out[output_name] = row[source]
    return out


def _identity_row_scalar(value: Any) -> Any:  # hygiene-ok: explicit-any -- heterogeneous replayed row values
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (str, bytes, list, tuple, dict)):
        return value
    item = getattr(value, "item", None)
    if callable(item) and getattr(value, "ndim", 0) == 0:
        try:
            return _identity_row_scalar(item())
        except (ValueError, AttributeError):
            return value
    return value


def _replay_identity_row(
    row: Mapping[str, Any],  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    steps: Sequence[ASTObject],
) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    from graphistry.compute.gfql.cypher.lowering import _SyntheticRowGraph
    from graphistry.compute.gfql.row.pipeline import execute_row_pipeline_call

    graph: Any = _SyntheticRowGraph(pd.DataFrame([dict(row)]))
    for step in steps:
        if not isinstance(step, ASTCall) or step.function not in _IDENTITY_ROW_REPLAY_CALLS:
            return None
        try:
            graph = execute_row_pipeline_call(cast(Any, graph), step.function, dict(step.params))  # hygiene-ok: cast -- _SyntheticRowGraph stands in for Plottable in the row pipeline
        except Exception:  # noqa: BLE001 -- compile-time probe; decline rather than fail the query
            return None
    out = graph._nodes
    if out is None or len(out) == 0:
        return None
    return {str(key): _identity_row_scalar(value) for key, value in out.iloc[0].to_dict().items()}


def ungrouped_aggregate_identity_row(
    row_steps: Sequence[ASTObject],
) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    """openCypher: an aggregate with no grouping keys ALWAYS yields exactly one row,
    so a later stage that empties the stream still returns the identities
    (count/sum -> 0, collect -> [], else null).

    Synthesizes that row at the last ungrouped aggregate step and applies the
    compiled suffix (trailing projections, post-aggregate expressions, ORDER BY,
    SKIP/LIMIT) to it. Declines whenever the suffix can filter or reshape rows on
    data invisible at compile time -- notably a post-aggregate WHERE, whose outcome
    depends on the real aggregate value.
    """
    pivot: Optional[int] = None
    seed: Optional[Dict[str, Any]] = None
    # Prefer the EARLIEST seed whose whole suffix is replayable: a later ungrouped
    # aggregate then aggregates the replayed identity row (count(c) over an empty
    # stream is 1, not 0); pivoting at the last seed there fabricates its value.
    for idx in range(len(row_steps)):
        candidate = _ungrouped_aggregate_identity_seed(row_steps[idx])
        if candidate is None:
            continue
        if pivot is None:
            pivot, seed = idx, candidate
            continue
        if all(
            isinstance(step, ASTCall) and step.function in _IDENTITY_ROW_REPLAY_CALLS
            for step in row_steps[pivot + 1:]
        ):
            break
        pivot, seed = idx, candidate
    if pivot is None or seed is None:
        return None

    current: Dict[str, Any] = seed
    suffix = row_steps[pivot + 1:]
    for idx, step in enumerate(suffix):
        if not isinstance(step, ASTCall):
            return None
        if step.function in {"order_by", "distinct"}:
            continue
        if step.function in {"skip", "limit"}:
            value = step.params.get("value")
            if not isinstance(value, int) or isinstance(value, bool):
                return None
            paged = identity_row_after_paging(
                current,
                skip_value=value if step.function == "skip" else None,
                limit_value=value if step.function == "limit" else None,
            )
            if paged is None:
                return None
            continue
        if step.function in {"select", "with_", "return_"}:
            projected = _identity_row_passthrough_projection(current, step.params)
            if projected is not None:
                current = projected
                continue
        return _identity_row_without_temps(_replay_identity_row(current, suffix[idx:]))
    return _identity_row_without_temps(current)


def _identity_row_without_temps(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:  # hygiene-ok: explicit-any -- heterogeneous Cypher identity values (0 / [] / None)
    if row is None:
        return None
    out = {
        key: value
        for key, value in row.items()
        if not key.startswith(("__cypher_group__", "__cypher_agg__", "__cypher_postagg__"))
    }
    return out or None


