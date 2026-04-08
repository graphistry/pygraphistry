"""Logical type definitions for frontend-neutral compiler IR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Literal, Optional, Union


@dataclass(frozen=True)
class BoundPredicate:
    """Frontend-normalized predicate payload."""

    expression: str = ""


@dataclass(frozen=True)
class NodeRef:
    """Semantic type for a node variable."""

    labels: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class EdgeRef:
    """Semantic type for an edge variable."""

    type: Optional[str] = None
    src_label: Optional[str] = None
    dst_label: Optional[str] = None


@dataclass(frozen=True)
class ScalarType:
    """Semantic type for scalar values."""

    kind: str = "unknown"
    nullable: bool = True


@dataclass(frozen=True)
class PathType:
    """Semantic type for a path value."""

    min_hops: int = 1
    max_hops: int = 1


@dataclass(frozen=True)
class ListType:
    """Semantic type for homogeneous lists."""

    element_type: "LogicalType" = field(default_factory=ScalarType)


LogicalType = Union[NodeRef, EdgeRef, ScalarType, PathType, ListType]


@dataclass(frozen=True)
class NodeSpec:
    """PatternGraph node entry."""

    alias: str
    labels: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class RelSpec:
    """PatternGraph relationship entry."""

    alias: Optional[str] = None
    types: FrozenSet[str] = field(default_factory=frozenset)
    src_alias: str = ""
    dst_alias: str = ""
    direction: Literal["forward", "reverse", "undirected"] = "undirected"
    min_hops: int = 1
    max_hops: Optional[int] = 1
    to_fixed_point: bool = False
    properties: list[BoundPredicate] = field(default_factory=list)


@dataclass(frozen=True)
class PatternGraph:
    """Flat pattern representation used by logical planning."""

    nodes: list[NodeSpec] = field(default_factory=list)
    rels: list[RelSpec] = field(default_factory=list)
