"""
BoundIR — the output of the Cypher semantic binder (M1).

This module defines the core type layer that M1 (Binder extraction) will
produce.  No behaviour lives here — pure frozen dataclass definitions.

Acceptance criteria (issue #1091):
- importable as ``from graphistry.compute.gfql.ir.bound_ir import BoundIR``
- ``mypy --strict graphistry/compute/gfql/ir/bound_ir.py`` passes with zero errors
- no behaviour changes; all existing tests continue to pass
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Literal

from graphistry.compute.gfql.ir.types import CypherAST, LogicalType, QueryGraph


# ---------------------------------------------------------------------------
# BoundVariable
# ---------------------------------------------------------------------------


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
        The set of OPTIONAL MATCH arm identifiers that *introduced* this
        variable as nullable.  Non-empty only for OPTIONAL MATCH sourced
        variables.  A ``frozenset`` (not ``Optional[str]``) to correctly
        handle multi-arm OPTIONAL MATCH (IC1 correctness requirement).
    entity_kind:
        Coarse kind tag — one of ``"node" | "edge" | "scalar" | "path"``.
        Redundant with ``logical_type`` but useful for fast dispatch without
        isinstance checks.
    """

    name: str
    logical_type: LogicalType
    nullable: bool
    null_extended_from: FrozenSet[str]
    entity_kind: Literal["node", "edge", "scalar", "path"]


# ---------------------------------------------------------------------------
# SemanticTable
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SemanticTable:
    """Maps variable names to their bound definitions for a single query scope.

    ``variables`` is a plain ``dict`` stored inside a frozen dataclass.
    The dataclass itself is frozen (cannot reassign ``variables``), but the
    dict is mutable — callers should treat it as read-only after construction.
    """

    variables: Dict[str, BoundVariable] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ScopeFrame
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# BoundIR
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoundIR:
    """The output of the semantic binder — a fully annotated, bound query.

    Attributes
    ----------
    ast:
        The original parse tree, unmodified.
    semantic_table:
        Variable → BoundVariable mapping for the whole query.
    scope_stack:
        Ordered list of scope frames from outermost to innermost.
        A ``list`` per spec (not tuple) — callers treat as read-only.
    query_graph:
        The pattern graph extracted from all MATCH clauses.
        Stub implementation for M0; full content added in M1.
    """

    ast: CypherAST
    semantic_table: SemanticTable
    scope_stack: List[ScopeFrame]
    query_graph: QueryGraph
