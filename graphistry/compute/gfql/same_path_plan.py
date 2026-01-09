"""Planner toggles for same-path WHERE comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set

from graphistry.compute.gfql.same_path_types import WhereComparison


@dataclass
class BitsetPlan:
    aliases: Set[str]
    lane_count: int = 64


@dataclass
class StateTablePlan:
    aliases: Set[str]
    cap: int = 128


@dataclass
class SamePathPlan:
    minmax_aliases: Dict[str, Set[str]] = field(default_factory=dict)
    bitsets: Dict[str, BitsetPlan] = field(default_factory=dict)
    state_tables: Dict[str, StateTablePlan] = field(default_factory=dict)

    def requires_minmax(self, alias: str) -> bool:
        return alias in self.minmax_aliases


def plan_same_path(
    where: Optional[Sequence[WhereComparison]],
    max_bitset_domain: int = 64,
    state_cap: int = 128,
) -> SamePathPlan:
    plan = SamePathPlan()
    if not where:
        return plan

    for clause in where:
        if clause.op in {"<", "<=", ">", ">="}:
            for ref in (clause.left, clause.right):
                plan.minmax_aliases.setdefault(ref.alias, set()).add(ref.column)
        elif clause.op in {"==", "!="}:
            key = _equality_key(clause)
            plan.bitsets.setdefault(key, BitsetPlan(set())).aliases.update(
                {clause.left.alias, clause.right.alias}
            )

    return plan


def _equality_key(clause: WhereComparison) -> str:
    cols = sorted(
        [
            f"{clause.left.alias}.{clause.left.column}",
            f"{clause.right.alias}.{clause.right.column}",
        ]
    )
    return "::".join(cols)
