"""Shared data structures for same-path WHERE comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, FrozenSet, List, Literal, Mapping, Optional, Sequence, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.compute.typing import DataFrameT


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
    where_json: Any
) -> List[WhereComparison]:
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
            raise ValueError(f"WHERE clause payload must be a dict, got {type(payload).__name__}")
        if "left" not in payload or "right" not in payload:
            raise ValueError(f"WHERE clause must have 'left' and 'right' keys, got {list(payload.keys())}")
        if not isinstance(payload["left"], str) or not isinstance(payload["right"], str):
            raise ValueError(f"WHERE clause 'left' and 'right' must be strings")
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


# ---------------------------------------------------------------------------
# Immutable PathState for Yannakakis execution
# ---------------------------------------------------------------------------

IdSet = FrozenSet[Any]


def _mp(d: Dict) -> MappingProxyType:
    """Wrap dict in MappingProxyType for true immutability."""
    return MappingProxyType(d)


def _update_map(m: Mapping, k: Any, v: Any) -> MappingProxyType:
    """Return new MappingProxyType with key updated."""
    d = dict(m)
    d[k] = v
    return _mp(d)


@dataclass(frozen=True, slots=True)
class PathState:
    """Immutable state for same-path execution.

    Contains allowed node/edge IDs per step index and pruned edge DataFrames.
    All fields are truly immutable (MappingProxyType + frozenset).

    Used by the Yannakakis-style semi-join executor for WHERE clause evaluation.
    All state transitions create new PathState instances (functional style).
    """

    allowed_nodes: Mapping[int, IdSet]
    allowed_edges: Mapping[int, IdSet]
    pruned_edges: Mapping[int, Any]  # edge_idx -> filtered DataFrame

    @classmethod
    def empty(cls) -> "PathState":
        """Create empty PathState."""
        return cls(
            allowed_nodes=_mp({}),
            allowed_edges=_mp({}),
            pruned_edges=_mp({}),
        )

    @classmethod
    def from_mutable(
        cls,
        allowed_nodes: Dict[int, Set[Any]],
        allowed_edges: Dict[int, Set[Any]],
        pruned_edges: Optional[Dict[int, Any]] = None,
    ) -> "PathState":
        """Create PathState from mutable dicts."""
        return cls(
            allowed_nodes=_mp({k: frozenset(v) for k, v in allowed_nodes.items()}),
            allowed_edges=_mp({k: frozenset(v) for k, v in allowed_edges.items()}),
            pruned_edges=_mp(pruned_edges or {}),
        )

    def to_mutable(self) -> tuple:
        """Convert to mutable dicts for local processing.

        Returns:
            (allowed_nodes: Dict[int, Set], allowed_edges: Dict[int, Set])
        """
        return (
            {k: set(v) for k, v in self.allowed_nodes.items()},
            {k: set(v) for k, v in self.allowed_edges.items()},
        )

    def restrict_nodes(self, idx: int, keep: IdSet) -> "PathState":
        """Return new PathState with node set at idx intersected with keep."""
        cur = self.allowed_nodes.get(idx, frozenset())
        new = cur & keep if cur else keep
        if new is cur:
            return self
        return PathState(
            allowed_nodes=_update_map(self.allowed_nodes, idx, new),
            allowed_edges=self.allowed_edges,
            pruned_edges=self.pruned_edges,
        )

    def set_nodes(self, idx: int, nodes: IdSet) -> "PathState":
        """Return new PathState with node set at idx replaced."""
        return PathState(
            allowed_nodes=_update_map(self.allowed_nodes, idx, nodes),
            allowed_edges=self.allowed_edges,
            pruned_edges=self.pruned_edges,
        )

    def restrict_edges(self, idx: int, keep: IdSet) -> "PathState":
        """Return new PathState with edge set at idx intersected with keep."""
        cur = self.allowed_edges.get(idx, frozenset())
        new = cur & keep if cur else keep
        if new is cur:
            return self
        return PathState(
            allowed_nodes=self.allowed_nodes,
            allowed_edges=_update_map(self.allowed_edges, idx, new),
            pruned_edges=self.pruned_edges,
        )

    def set_edges(self, idx: int, edges: IdSet) -> "PathState":
        """Return new PathState with edge set at idx replaced."""
        return PathState(
            allowed_nodes=self.allowed_nodes,
            allowed_edges=_update_map(self.allowed_edges, idx, edges),
            pruned_edges=self.pruned_edges,
        )

    def with_pruned_edges(self, edge_idx: int, df: Any) -> "PathState":
        """Return new PathState with pruned edges DataFrame at edge_idx."""
        return PathState(
            allowed_nodes=self.allowed_nodes,
            allowed_edges=self.allowed_edges,
            pruned_edges=_update_map(self.pruned_edges, edge_idx, df),
        )

    def sync_to_mutable(
        self,
        mutable_nodes: Dict[int, Set[Any]],
        mutable_edges: Dict[int, Set[Any]],
    ) -> None:
        """Sync this immutable state back to mutable dicts.

        Clears and updates the mutable dicts in-place.
        """
        mutable_nodes.clear()
        mutable_nodes.update({k: set(v) for k, v in self.allowed_nodes.items()})
        mutable_edges.clear()
        mutable_edges.update({k: set(v) for k, v in self.allowed_edges.items()})

    def sync_pruned_to_forward_steps(self, forward_steps: List[Any]) -> None:
        """Sync pruned_edges back to forward_steps (mutates forward_steps)."""
        for edge_idx, df in self.pruned_edges.items():
            forward_steps[edge_idx]._edges = df
