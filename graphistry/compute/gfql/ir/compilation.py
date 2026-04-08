"""
Frontend-boundary and planning-context dataclasses for compiler IR.

This module contains engine-agnostic metadata used before and during
compilation. Frontend AST payloads are intentionally opaque to the engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, FrozenSet, Optional


class QueryLanguage(Enum):
    """Frontend dialect identifier for compiler entry points."""

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


# Back-compat alias used by older call sites and docs.
GFQLSchema = GraphSchemaCatalog


@dataclass(frozen=True)
class CompilationState:
    """Frontend-owned inputs carried into compiler stages."""

    frontend: QueryLanguage = QueryLanguage.CYPHER
    frontend_ast: Optional[Any] = None


@dataclass(frozen=True)
class PlanContext:
    """Planner context for semantic/binding stages."""

    catalog: GraphSchemaCatalog = field(default_factory=GraphSchemaCatalog)
