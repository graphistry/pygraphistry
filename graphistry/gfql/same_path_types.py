"""Shared data structures for same-path WHERE comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence


ComparisonOp = Literal[
    "==",
    "!=",
    "<",
    "<=",
    ">",
    ">=",
]


@dataclass(frozen=True)
class StepColumnRef:
    alias: str
    column: str


@dataclass(frozen=True)
class WhereComparison:
    left: StepColumnRef
    op: ComparisonOp
    right: StepColumnRef


def col(alias: str, column: str) -> StepColumnRef:
    return StepColumnRef(alias, column)


def compare(
    left: StepColumnRef, op: ComparisonOp, right: StepColumnRef
) -> WhereComparison:
    return WhereComparison(left, op, right)


def parse_column_ref(ref: str) -> StepColumnRef:
    if "." not in ref:
        raise ValueError(f"Column reference '{ref}' must be alias.column")
    alias, column = ref.split(".", 1)
    if not alias or not column:
        raise ValueError(f"Invalid column reference '{ref}'")
    return StepColumnRef(alias, column)


def parse_where_json(
    where_json: Optional[Sequence[Dict[str, Dict[str, str]]]]
) -> List[WhereComparison]:
    if not where_json:
        return []
    clauses: List[WhereComparison] = []
    for entry in where_json:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid WHERE clause: {entry}")
        op_name, payload = next(iter(entry.items()))
        if op_name not in {"eq", "neq", "gt", "lt", "ge", "le"}:
            raise ValueError(f"Unsupported WHERE operator '{op_name}'")
        op_map: Dict[str, ComparisonOp] = {
            "eq": "==",
            "neq": "!=",
            "gt": ">",
            "lt": "<",
            "ge": ">=",
            "le": "<=",
        }
        left = parse_column_ref(payload["left"])
        right = parse_column_ref(payload["right"])
        clauses.append(WhereComparison(left, op_map[op_name], right))
    return clauses


def where_to_json(where: Sequence[WhereComparison]) -> List[Dict[str, Dict[str, str]]]:
    result: List[Dict[str, Dict[str, str]]] = []
    op_map: Dict[str, str] = {
        "==": "eq",
        "!=": "neq",
        ">": "gt",
        "<": "lt",
        ">=": "ge",
        "<=": "le",
    }
    for clause in where:
        op_name = op_map.get(clause.op)
        if not op_name:
            continue
        result.append(
            {
                op_name: {
                    "left": f"{clause.left.alias}.{clause.left.column}",
                    "right": f"{clause.right.alias}.{clause.right.column}",
                }
            }
        )
    return result
