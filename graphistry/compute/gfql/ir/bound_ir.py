"""Bound IR dataclasses produced by frontend binders."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Literal

from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import BoundPredicate, LogicalType


@dataclass(frozen=True)
class BoundVariable:
    """A single bound variable in semantic scope."""

    name: str
    logical_type: LogicalType
    nullable: bool
    null_extended_from: FrozenSet[str]
    entity_kind: Literal["node", "edge", "scalar"]
    scope_id: int = 0


@dataclass(frozen=True)
class SemanticTable:
    """Variable table for a bound query."""

    variables: Dict[str, BoundVariable] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeFrame:
    """Scope boundary metadata (e.g., MATCH/WITH/UNWIND)."""

    visible_vars: FrozenSet[str]
    schema: RowSchema
    origin_clause: str


@dataclass(frozen=True)
class BoundQueryPart:
    """One logical part of a query between scope boundaries."""

    clause: str = ""
    inputs: FrozenSet[str] = field(default_factory=frozenset)
    outputs: FrozenSet[str] = field(default_factory=frozenset)
    predicates: List[BoundPredicate] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoundIR:
    """Frontend-neutral output of semantic binding."""

    query_parts: List[BoundQueryPart] = field(default_factory=list)
    semantic_table: SemanticTable = field(default_factory=SemanticTable)
    scope_stack: List[ScopeFrame] = field(default_factory=list)
    params: Dict[str, Any] = field(default_factory=dict)
