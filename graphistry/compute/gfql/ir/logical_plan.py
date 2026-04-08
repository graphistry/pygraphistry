"""Logical plan operator dataclasses for compiler IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional, Tuple

from graphistry.compute.gfql.ir.types import BoundPredicate, LogicalType


@dataclass(frozen=True)
class RowSchema:
    """Row-oriented type map."""

    columns: Dict[str, LogicalType] = field(default_factory=dict)


@dataclass(frozen=True)
class LogicalPlan:
    """Marker base for logical operators."""

    op_id: int = 0
    output_schema: RowSchema = field(default_factory=RowSchema)


@dataclass(frozen=True)
class NodeScan(LogicalPlan):
    label: str = ""


@dataclass(frozen=True)
class EdgeScan(LogicalPlan):
    edge_type: Optional[str] = None


@dataclass(frozen=True)
class IndexScan(LogicalPlan):
    label: str = ""
    index: Any = None
    predicate: BoundPredicate = field(default_factory=BoundPredicate)
    residual_predicates: List[BoundPredicate] = field(default_factory=list)


@dataclass(frozen=True)
class PatternMatch(LogicalPlan):
    pattern: Any = None
    input: Optional[LogicalPlan] = None
    predicates: List[BoundPredicate] = field(default_factory=list)
    optional: bool = False
    arm_id: Optional[str] = None


@dataclass(frozen=True)
class PathProjection(LogicalPlan):
    path_var: str = ""
    hop_count_col: str = ""
    input: Optional[LogicalPlan] = None


@dataclass(frozen=True)
class Filter(LogicalPlan):
    input: Optional[LogicalPlan] = None
    predicate: BoundPredicate = field(default_factory=BoundPredicate)


@dataclass(frozen=True)
class Project(LogicalPlan):
    input: Optional[LogicalPlan] = None
    expressions: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Aggregate(LogicalPlan):
    input: Optional[LogicalPlan] = None
    group_keys: List[Any] = field(default_factory=list)
    aggregates: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Distinct(LogicalPlan):
    input: Optional[LogicalPlan] = None


@dataclass(frozen=True)
class OrderBy(LogicalPlan):
    input: Optional[LogicalPlan] = None
    sort_keys: List[Any] = field(default_factory=list)


@dataclass(frozen=True)
class Limit(LogicalPlan):
    input: Optional[LogicalPlan] = None
    count: int = 0
    offset: int = 0


@dataclass(frozen=True)
class Skip(LogicalPlan):
    input: Optional[LogicalPlan] = None
    count: int = 0


@dataclass(frozen=True)
class Unwind(LogicalPlan):
    input: Optional[LogicalPlan] = None
    list_expr: Any = None
    variable: str = ""


@dataclass(frozen=True)
class Union(LogicalPlan):
    left: Optional[LogicalPlan] = None
    right: Optional[LogicalPlan] = None
    distinct: bool = True


@dataclass(frozen=True)
class Join(LogicalPlan):
    left: Optional[LogicalPlan] = None
    right: Optional[LogicalPlan] = None
    condition: Any = None
    join_type: str = "inner"


@dataclass(frozen=True)
class GraphToRows(LogicalPlan):
    variant: str = ""
    input: Optional[LogicalPlan] = None


@dataclass(frozen=True)
class RowsToGraph(LogicalPlan):
    input: Optional[LogicalPlan] = None
    node_id_col: str = ""
    src_col: str = ""
    dst_col: str = ""


@dataclass(frozen=True)
class Apply(LogicalPlan):
    input: Optional[LogicalPlan] = None
    subquery: Optional[LogicalPlan] = None
    correlation_vars: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SemiApply(LogicalPlan):
    input: Optional[LogicalPlan] = None
    subquery: Optional[LogicalPlan] = None
    correlation_vars: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class AntiSemiApply(LogicalPlan):
    input: Optional[LogicalPlan] = None
    subquery: Optional[LogicalPlan] = None
    correlation_vars: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ProcedureOutputColumn:
    source_name: str
    output_name: str


@dataclass(frozen=True)
class ProcedureCall(LogicalPlan):
    procedure: str = ""
    backend: Literal["degree", "cugraph", "igraph", "networkx"] = "degree"
    algorithm: Optional[str] = None
    call_function: Optional[str] = None
    result_kind: Literal["rows", "graph"] = "rows"
    row_kind: Literal["degree", "node", "edge", "graph_only", "node_or_graph"] = "node"
    output_columns: Tuple[ProcedureOutputColumn, ...] = field(default_factory=tuple)
    call_params: Mapping[str, Any] = field(default_factory=dict)
