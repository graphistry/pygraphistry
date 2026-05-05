"""Compilation-state and planning-context dataclasses for compiler IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, Iterable, List, Literal, Optional, Tuple

from graphistry.compute.gfql.ir.bound_ir import BoundIR, ScopeFrame, SemanticTable
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

    @classmethod
    def from_schema_parts(
        cls,
        *,
        node_columns: Iterable[str] = (),
        edge_columns: Iterable[str] = (),
        node_id_column: Optional[str] = None,
        edge_source_column: Optional[str] = None,
        edge_destination_column: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "GraphSchemaCatalog":
        """Build a catalog from iterables while normalizing into stable shapes.

        T1 contract: callers can pass any iterable of column names and receive
        a frozen-set-backed catalog. Metadata is copied so caller-side
        mutations after construction do not alias into the catalog.
        """
        return cls(
            node_columns=frozenset(node_columns),
            edge_columns=frozenset(edge_columns),
            node_id_column=node_id_column,
            edge_source_column=edge_source_column,
            edge_destination_column=edge_destination_column,
            metadata=dict(metadata or {}),
        )

    @property
    def node_id(self) -> Optional[str]:
        """Canonical accessor for the node identity column name."""
        return self.node_id_column

    @property
    def edge_source(self) -> Optional[str]:
        """Canonical accessor for the edge source column name."""
        return self.edge_source_column

    @property
    def edge_destination(self) -> Optional[str]:
        """Canonical accessor for the edge destination column name."""
        return self.edge_destination_column

    def has_node_column(self, column: str) -> bool:
        """Return whether a node column exists in the catalog contract."""
        return column in self.node_columns

    def has_edge_column(self, column: str) -> bool:
        """Return whether an edge column exists in the catalog contract."""
        return column in self.edge_columns


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
    # Pass-visible binder scope metadata for optimization safety checks.
    scope_stack: Tuple[ScopeFrame, ...] = field(default_factory=tuple)


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
