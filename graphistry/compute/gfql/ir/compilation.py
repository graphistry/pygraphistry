"""Compilation-state and planning-context dataclasses for compiler IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Literal, Optional, Tuple

from graphistry.compute.gfql.ir.bound_ir import BoundIR, SemanticTable
from graphistry.compute.gfql.ir.logical_plan import LogicalPlan
from graphistry.compute.gfql.ir.query_graph import QueryGraph

if TYPE_CHECKING:
    from graphistry.compute.gfql.physical_planner import PhysicalOperator
else:
    PhysicalOperator = Any

NodeId = int


class QueryLanguage(str, Enum):
    """Frontend dialect identifier."""

    CYPHER = "cypher"
    CHAIN_DSL = "chain_dsl"


@dataclass(frozen=True)
class GraphSchemaCatalog:
    """Schema summary consumed by planning and validation stages."""

    node_columns: FrozenSet[str] = field(default_factory=frozenset)
    edge_columns: FrozenSet[str] = field(default_factory=frozenset)
    node_id_column: Optional[str] = None
    edge_source_column: Optional[str] = None
    edge_destination_column: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


GFQLSchema = GraphSchemaCatalog


@dataclass(frozen=True)
class StatsQuery:
    """Placeholder stats facade for planning hooks."""

    source: str = "uniform"


@dataclass(frozen=True)
class IndexDescriptor:
    """Index descriptor used by planning/index rewrite passes."""

    label: str = ""
    properties: Tuple[str, ...] = ()
    supported_ops: FrozenSet[str] = field(default_factory=frozenset)
    unique: bool = False


@dataclass(frozen=True)
class BackendCapabilities:
    """Backend capability flags used for physical planning."""

    name: str = "default"


@dataclass(frozen=True)
class CompilerConfig:
    """Compiler flag bundle."""

    enable_cbo: bool = False
    max_hop_depth: Optional[int] = None


@dataclass(frozen=True)
class PlanContext:
    """Long-lived, read-only planning context."""

    catalog: GraphSchemaCatalog = field(default_factory=GraphSchemaCatalog)
    stats: StatsQuery = field(default_factory=StatsQuery)
    indexes: List[IndexDescriptor] = field(default_factory=list)
    backend: BackendCapabilities = field(default_factory=BackendCapabilities)
    config: CompilerConfig = field(default_factory=CompilerConfig)


@dataclass(frozen=True)
class CompilerError:
    """Compiler diagnostic payload placeholder."""

    message: str = ""


@dataclass(frozen=True)
class PhysicalPlan:
    """Physical plan wrapper contract for M3 planner routing."""

    route: Literal["same_path", "wavefront", "row_pipeline"] = "row_pipeline"
    operators: Tuple[PhysicalOperator, ...] = field(default_factory=tuple)
    logical_op_ids: Tuple[int, ...] = field(default_factory=tuple)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationState:
    """Mutable per-query accumulator for frontend and planning phases."""

    query_text: str = ""
    ctx: PlanContext = field(default_factory=PlanContext)
    frontend: QueryLanguage = QueryLanguage.CYPHER

    frontend_ast: Optional[Any] = None
    bound_ir: Optional[BoundIR] = None
    semantic_table: Optional[SemanticTable] = None
    logical_plan: Optional[LogicalPlan] = None
    query_graph: Optional[QueryGraph] = None
    physical_plan: Optional[PhysicalPlan] = None

    diagnostics: List[CompilerError] = field(default_factory=list)
    _cardinalities: Dict[NodeId, float] = field(default_factory=dict)
    _provided_orders: Dict[NodeId, List[Any]] = field(default_factory=dict)
    _solved_predicates: Dict[NodeId, List[Any]] = field(default_factory=dict)
