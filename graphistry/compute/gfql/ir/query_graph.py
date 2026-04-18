"""QueryGraph dataclasses and extraction from BoundIR."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, FrozenSet, List, Set

from graphistry.compute.gfql.ir.types import BoundPredicate, LogicalType

if TYPE_CHECKING:
    from graphistry.compute.gfql.ir.bound_ir import BoundIR, BoundQueryPart


@dataclass(frozen=True)
class OptionalArm:
    """OPTIONAL MATCH arm metadata."""

    arm_id: str
    join_aliases: FrozenSet[str] = field(default_factory=frozenset)
    nullable_aliases: FrozenSet[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class ConnectedComponent:
    """Connected pattern component.

    ``predicates``, ``entry_points``, and ``hop_order`` are populated by a
    later physical-planning pass; ``extract_query_graph`` leaves them empty.
    """

    node_aliases: List[str] = field(default_factory=list)
    edge_aliases: List[str] = field(default_factory=list)
    predicates: List[BoundPredicate] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    hop_order: List[int] = field(default_factory=list)


@dataclass(frozen=True)
class QueryGraph:
    """Join-ordering and optional-arm scaffold."""

    components: List[ConnectedComponent] = field(default_factory=list)
    boundary_aliases: Dict[str, LogicalType] = field(default_factory=dict)
    optional_arms: List[OptionalArm] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Union-find helpers (module-level to avoid redefinition inside loops)
# ---------------------------------------------------------------------------

def _uf_find(parent: Dict[str, str], x: str) -> str:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _uf_union(parent: Dict[str, str], a: str, b: str) -> None:
    ra, rb = _uf_find(parent, a), _uf_find(parent, b)
    if ra != rb:
        parent[rb] = ra


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------

_SCOPE_SPLIT_CLAUSES: Set[str] = {"with", "return"}


def extract_query_graph(bound_ir: BoundIR) -> QueryGraph:
    """Build a QueryGraph from a BoundIR by walking query_parts."""
    # --- 1. Split into scope groups; collect boundary aliases from WITH ---
    scope_groups: List[List[BoundQueryPart]] = []
    boundary_aliases: Dict[str, LogicalType] = {}
    current_scope: List[BoundQueryPart] = []

    for part in bound_ir.query_parts:
        if part.clause in _SCOPE_SPLIT_CLAUSES:
            # Only WITH projects aliases into the next scope; RETURN is terminal.
            if part.clause == "with":
                for alias in part.inputs:
                    var = bound_ir.semantic_table.variables.get(alias)
                    if var is not None:
                        boundary_aliases[alias] = var.logical_type
            if current_scope:
                scope_groups.append(current_scope)
            current_scope = []
        else:
            current_scope.append(part)

    if current_scope:
        scope_groups.append(current_scope)

    # --- 2. Connected components via union-find within each scope ---
    components: List[ConnectedComponent] = []

    for scope_parts in scope_groups:
        parent: Dict[str, str] = {}
        empty_part_count = 0

        for part in scope_parts:
            aliases = list(part.outputs)
            if not aliases:
                empty_part_count += 1
                continue
            for alias in aliases:
                if alias not in parent:
                    parent[alias] = alias
            root = aliases[0]
            for alias in aliases[1:]:
                _uf_union(parent, root, alias)

        root_to_aliases: Dict[str, List[str]] = {}
        for alias in parent:
            r = _uf_find(parent, alias)
            root_to_aliases.setdefault(r, []).append(alias)

        for alias_list in root_to_aliases.values():
            node_aliases: List[str] = []
            edge_aliases: List[str] = []
            for alias in sorted(alias_list):
                var = bound_ir.semantic_table.variables.get(alias)
                if var is not None and var.entity_kind == "edge":
                    edge_aliases.append(alias)
                else:
                    node_aliases.append(alias)
            components.append(ConnectedComponent(
                node_aliases=node_aliases,
                edge_aliases=edge_aliases,
            ))

        for _ in range(empty_part_count):
            components.append(ConnectedComponent())

    # --- 3. Optional arms from null_extended_from ---
    arm_to_nullable: Dict[str, Set[str]] = {}
    for alias, var in bound_ir.semantic_table.variables.items():
        for arm_id in var.null_extended_from:
            arm_to_nullable.setdefault(arm_id, set()).add(alias)

    arm_to_join: Dict[str, Set[str]] = {}
    for part in bound_ir.query_parts:
        if part.clause != "optional_match":
            continue
        part_arm_ids: Set[str] = set()
        for out_alias in part.outputs:
            out_var = bound_ir.semantic_table.variables.get(out_alias)
            if out_var is not None:
                part_arm_ids |= out_var.null_extended_from
        # Required inputs shared with an optional arm → join aliases
        for alias in part.inputs:
            var = bound_ir.semantic_table.variables.get(alias)
            if var is not None and not var.nullable:
                for arm_id in part_arm_ids:
                    arm_to_join.setdefault(arm_id, set()).add(alias)

    optional_arms: List[OptionalArm] = [
        OptionalArm(
            arm_id=arm_id,
            join_aliases=frozenset(arm_to_join.get(arm_id, set())),
            nullable_aliases=frozenset(nullable_set),
        )
        for arm_id, nullable_set in sorted(arm_to_nullable.items())
    ]

    return QueryGraph(
        components=components,
        boundary_aliases=boundary_aliases,
        optional_arms=optional_arms,
    )
