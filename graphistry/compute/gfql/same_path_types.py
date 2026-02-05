from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence

from graphistry.compute.typing import DataFrameT, DomainT

ComparisonOp = Literal["==", "!=", "<", "<=", ">", ">="]
_WHERE_OP_MAP: Dict[str, ComparisonOp] = {
    "eq": "==",
    "neq": "!=",
    "gt": ">",
    "lt": "<",
    "ge": ">=",
    "le": "<=",
}
_WHERE_OP_REV: Dict[ComparisonOp, str] = {value: key for key, value in _WHERE_OP_MAP.items()}


@dataclass(frozen=True)
class StepColumnRef:
    alias: str
    column: str


@dataclass(frozen=True)
class WhereComparison:
    left: StepColumnRef
    op: ComparisonOp
    right: StepColumnRef


col = StepColumnRef
compare = WhereComparison


def parse_column_ref(ref: str) -> StepColumnRef:
    if "." not in ref:
        raise ValueError(f"Column reference '{ref}' must be alias.column")
    alias, column = ref.split(".", 1)
    if not alias or not column:
        raise ValueError(f"Invalid column reference '{ref}'")
    return StepColumnRef(alias, column)


def parse_where_json(where_json: Any) -> List[WhereComparison]:
    if where_json is None:
        return []
    if not isinstance(where_json, (list, tuple)):
        raise ValueError(f"WHERE clauses must be a list, got {type(where_json).__name__}")
    clauses: List[WhereComparison] = []
    for entry in where_json:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Invalid WHERE clause: {entry}")
        op_name, payload = next(iter(entry.items()))
        if op_name not in {"eq", "neq", "gt", "lt", "ge", "le"}:
            raise ValueError(f"Unsupported WHERE operator '{op_name}'")
        if not isinstance(payload, dict):
            raise ValueError(
                f"WHERE clause payload must be a dict, got {type(payload).__name__}"
            )
        if "left" not in payload or "right" not in payload:
            raise ValueError(
                "WHERE clause must have 'left' and 'right' keys, got "
                f"{list(payload.keys())}"
            )
        if not isinstance(payload["left"], str) or not isinstance(payload["right"], str):
            raise ValueError("WHERE clause 'left' and 'right' must be strings")
        left = parse_column_ref(payload["left"])
        right = parse_column_ref(payload["right"])
        clauses.append(WhereComparison(left, _WHERE_OP_MAP[op_name], right))
    return clauses


def where_to_json(where: Sequence[WhereComparison]) -> List[Dict[str, Dict[str, str]]]:
    result: List[Dict[str, Dict[str, str]]] = []
    for clause in where:
        op_name = _WHERE_OP_REV.get(clause.op)
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


@dataclass(frozen=True)
class PathState:
    allowed_nodes: Mapping[int, DomainT]
    allowed_edges: Mapping[int, DomainT]
    pruned_edges: Mapping[int, DataFrameT]

    @classmethod
    def from_mutable(
        cls,
        allowed_nodes: Dict[int, DomainT],
        allowed_edges: Dict[int, DomainT],
        pruned_edges: Optional[Dict[int, DataFrameT]] = None,
    ) -> "PathState":
        return cls(
            allowed_nodes=MappingProxyType(dict(allowed_nodes)),
            allowed_edges=MappingProxyType(dict(allowed_edges)),
            pruned_edges=MappingProxyType(pruned_edges or {}),
        )

    def to_mutable(self) -> tuple:
        return dict(self.allowed_nodes), dict(self.allowed_edges)
