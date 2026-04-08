"""QueryGraph dataclasses used by logical planning."""
from __future__ import annotations

from dataclasses import dataclass, field

from graphistry.compute.gfql.ir.types import BoundPredicate, LogicalType


@dataclass(frozen=True)
class OptionalArm:
    """OPTIONAL MATCH arm metadata."""

    arm_id: str
    join_aliases: frozenset[str] = field(default_factory=frozenset)
    nullable_aliases: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ConnectedComponent:
    """Connected pattern component."""

    node_aliases: list[str] = field(default_factory=list)
    edge_aliases: list[str] = field(default_factory=list)
    predicates: list[BoundPredicate] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    hop_order: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class QueryGraph:
    """Join-ordering and optional-arm scaffold."""

    components: list[ConnectedComponent] = field(default_factory=list)
    boundary_aliases: dict[str, LogicalType] = field(default_factory=dict)
    optional_arms: list[OptionalArm] = field(default_factory=list)
