"""Which reentry outputs an unmatched prefix row's null-fill can reproduce, and with what."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graphistry.compute.ast import ASTCall, ASTObject
from graphistry.compute.gfql.agg_types import (
    CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS,
    CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS,
    CypherEmptyGroupValue,
)
from graphistry.compute.gfql.cypher.reentry.naming import (
    is_reentry_hidden_column_reference,
    reentry_hidden_column_output_name,
)
from graphistry.compute.gfql.identifiers import identifier_tokens, is_bare_identifier

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.lowering import CompiledCypherQuery


__all__ = [
    "CarriedOutputSources",
    "CARRIED_OUTPUTS_NOT_REPRODUCIBLE",
    "carried_output_sources",
    "carried_output_source_column",
    "optional_reentry_aggregate_fill_values",
]


@dataclass(frozen=True)
class CarriedOutputSources:
    """Prefix-frame column behind each result output that reads the carried alias."""

    columns: Mapping[str, str]
    every_output_reproducible: bool


#: At least one carried-alias output has no prefix-frame column behind it, so an unmatched
#: prefix row cannot be given its own values and the null-fill must decline typed.
CARRIED_OUTPUTS_NOT_REPRODUCIBLE = CarriedOutputSources(columns={}, every_output_reproducible=False)

_EVERY_CARRIED_OUTPUT_TRIVIALLY_REPRODUCIBLE = CarriedOutputSources(
    columns={}, every_output_reproducible=True
)


def _output_reads_carried_alias(
    src: str, *, alias_names: Set[str], scalar_columns: Set[str]
) -> bool:
    tokens = identifier_tokens(src)
    return bool(tokens & alias_names or tokens & scalar_columns)


def carried_output_source_column(
    src: str, *, alias_names: Set[str], scalar_columns: Set[str]
) -> Optional[str]:
    """Prefix-frame column an output copies verbatim, or None when it cannot be reproduced."""
    if src in scalar_columns:
        return src
    parts = src.split(".")
    if len(parts) != 2 or parts[0] not in alias_names:
        return None
    carried_output = reentry_hidden_column_output_name(parts[1])
    if carried_output is not None and carried_output in scalar_columns:
        return carried_output
    if is_bare_identifier(parts[1]):
        return src
    return None


def _projection_items_and_grouping(
    ops: Sequence[ASTObject],
) -> Tuple[Optional[List[object]], bool]:
    """The last projection stage's items, and whether a group_by follows it."""
    items: Optional[List[object]] = None
    grouped = False
    for op in ops:
        if not isinstance(op, ASTCall):
            continue
        if op.function in ("select", "return_", "with_") and op.params.get("items"):
            items = list(op.params["items"])
            grouped = False
        elif op.function == "group_by":
            grouped = True
    return items, grouped


def carried_output_sources(compiled_query: "CompiledCypherQuery") -> CarriedOutputSources:
    """Which result outputs the unmatched-prefix-row null-fill can reproduce."""
    plan = compiled_query.reentry_plan
    if plan is None:
        return _EVERY_CARRIED_OUTPUT_TRIVIALLY_REPRODUCIBLE
    alias_names = {plan.reentry_alias_name} | {a.output_name for a in plan.aliases}
    scalar_columns = set(plan.scalar_columns)
    ops = list(compiled_query.chain.chain) if compiled_query.chain is not None else []
    items, grouped = _projection_items_and_grouping(ops)
    if items is None or grouped:
        return _EVERY_CARRIED_OUTPUT_TRIVIALLY_REPRODUCIBLE
    columns: Dict[str, str] = {}
    for item in items:
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)):
            continue
        name, src = item[0], item[1]
        if not isinstance(src, str):
            continue
        if not _output_reads_carried_alias(
            src, alias_names=alias_names, scalar_columns=scalar_columns
        ):
            continue
        source_column = carried_output_source_column(
            src, alias_names=alias_names, scalar_columns=scalar_columns
        )
        if source_column is None:
            return CARRIED_OUTPUTS_NOT_REPRODUCIBLE
        columns[name] = source_column
    return CarriedOutputSources(columns=columns, every_output_reproducible=True)


def optional_reentry_aggregate_fill_values(
    compiled_query: "CompiledCypherQuery",
) -> Dict[str, CypherEmptyGroupValue]:
    """Cypher empty-group value per aggregate output on an unmatched prefix row's null-extended row."""
    plan = compiled_query.reentry_plan
    carried_scalar_columns = set(plan.scalar_columns) if plan is not None else set()
    reentry_alias = plan.reentry_alias_name if plan is not None else None

    def source_is_carried_rather_than_suffix_bound(source: str) -> bool:
        base = source.split(".", 1)[0]
        return (
            is_reentry_hidden_column_reference(source)
            or base in carried_scalar_columns
            or base == reentry_alias
        )

    ops = list(compiled_query.chain.chain) if compiled_query.chain is not None else []
    with_map: Dict[str, object] = {}
    fills: Dict[str, CypherEmptyGroupValue] = {}
    for op in ops:
        if not isinstance(op, ASTCall):
            continue
        if op.function == "with_":
            for item in op.params.get("items") or []:
                if isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str):
                    with_map[item[0]] = item[1]
        elif op.function == "group_by":
            fills = {}
            for agg in op.params.get("aggregations") or []:
                if not isinstance(agg, (list, tuple)) or len(agg) not in (2, 3):
                    continue
                alias = str(agg[0])
                func = str(agg[1]).lower()
                expr = agg[2] if len(agg) == 3 else None
                if func == "count" and (expr is None or expr == "*"):
                    fills[alias] = 1
                    continue
                source: object = with_map.get(expr, expr) if isinstance(expr, str) else expr
                if not isinstance(source, str):
                    continue
                if source_is_carried_rather_than_suffix_bound(source):
                    continue
                if func in CYPHER_ZERO_EMPTY_GROUP_AGGREGATIONS:
                    fills[alias] = 0
                elif func in CYPHER_EMPTY_LIST_EMPTY_GROUP_AGGREGATIONS:
                    fills[alias] = []
    return fills
