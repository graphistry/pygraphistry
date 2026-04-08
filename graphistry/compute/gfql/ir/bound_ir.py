"""
Semantic IR dataclasses produced by Cypher binding.

This module defines the bound query representation and scope/variable metadata.
It contains data definitions only and no execution logic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal

from graphistry.compute.gfql.ir.types import LogicalType, QueryGraph


@dataclass(frozen=True)
class BoundVariable:
    """A single Cypher variable after semantic analysis.

    Attributes
    ----------
    name:
        The variable name as it appears in the query (e.g. ``"n"``, ``"r"``).
    logical_type:
        The semantic kind — one of ``NodeRef | EdgeRef | ScalarType | PathType``.
    nullable:
        True if the variable can be null (e.g. introduced by OPTIONAL MATCH).
    null_extended_from:
        The set of OPTIONAL MATCH arm identifiers that introduced this
        variable as nullable.
    entity_kind:
        Coarse kind tag — one of ``"node" | "edge" | "scalar" | "path"``.
    """

    name: str
    logical_type: LogicalType
    nullable: bool
    null_extended_from: FrozenSet[str]
    entity_kind: Literal["node", "edge", "scalar", "path"]


@dataclass(frozen=True)
class SemanticTable:
    """Maps variable names to their bound definitions for a single query scope.

    ``variables`` is a plain ``dict`` stored inside a frozen dataclass.
    The dataclass itself is frozen (cannot reassign ``variables``), but the
    dict is mutable — callers should treat it as read-only after construction.
    """

    variables: Dict[str, BoundVariable] = field(default_factory=dict)


@dataclass(frozen=True)
class ScopeFrame:
    """One frame on the scope stack — tracks which variables are visible.

    Attributes
    ----------
    visible_vars:
        Names of variables in scope at this frame.
    scope_kind:
        What kind of clause introduced this frame.
    """

    visible_vars: FrozenSet[str]
    scope_kind: Literal["match", "with", "union", "subquery"]


@dataclass(frozen=True)
class BoundIR:
    """The output of the semantic binder — a fully annotated, bound query.

    Attributes
    ----------
    semantic_table:
        Variable → BoundVariable mapping for the whole query.
    scope_stack:
        Ordered list of scope frames from outermost to innermost.
        Stored as a ``list``; callers should treat it as read-only.
    query_graph:
        The pattern graph extracted from all MATCH clauses.
        This is currently a placeholder container.
    """

    semantic_table: SemanticTable
    scope_stack: List[ScopeFrame]
    query_graph: QueryGraph
