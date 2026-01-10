"""Chain metadata for efficient step/alias lookups.

Precomputes chain structure once to avoid repeated O(n) scans.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import AliasBinding


@dataclass(frozen=True)
class ChainMeta:
    """Precomputed chain structure for O(1) lookups.

    Attributes:
        node_indices: List of step indices that are node operations
        edge_indices: List of step indices that are edge operations
        step_to_alias: Map from step index to alias name (if any)
        alias_to_step: Map from alias name to step index
    """
    node_indices: List[int]
    edge_indices: List[int]
    step_to_alias: Dict[int, str]
    alias_to_step: Dict[str, int]

    @staticmethod
    def from_chain(
        chain: Sequence[ASTObject],
        alias_bindings: Dict[str, "AliasBinding"]
    ) -> "ChainMeta":
        """Build ChainMeta from a chain and its alias bindings.

        Args:
            chain: Sequence of ASTNode/ASTEdge operations
            alias_bindings: Map from alias names to AliasBinding objects

        Returns:
            ChainMeta with precomputed indices and alias maps
        """
        node_indices: List[int] = []
        edge_indices: List[int] = []

        for i, op in enumerate(chain):
            if isinstance(op, ASTNode):
                node_indices.append(i)
            elif isinstance(op, ASTEdge):
                edge_indices.append(i)

        step_to_alias = {b.step_index: alias for alias, b in alias_bindings.items()}
        alias_to_step = {alias: b.step_index for alias, b in alias_bindings.items()}

        return ChainMeta(
            node_indices=node_indices,
            edge_indices=edge_indices,
            step_to_alias=step_to_alias,
            alias_to_step=alias_to_step,
        )

    def alias_for_step(self, step_index: int) -> Optional[str]:
        """Get alias for a step index, or None if no alias."""
        return self.step_to_alias.get(step_index)

    def are_steps_adjacent_nodes(self, step1: int, step2: int) -> bool:
        """Check if two step indices represent adjacent nodes (one edge apart).

        For nodes in a chain, adjacent means step indices differ by exactly 2
        (node - edge - node pattern).
        """
        return abs(step1 - step2) == 2

    def validate(self) -> None:
        """Validate chain structure for same-path execution.

        Raises:
            ValueError: If chain doesn't have proper node/edge alternation
        """
        if not self.node_indices:
            raise ValueError("Same-path executor requires at least one node step")
        if len(self.node_indices) != len(self.edge_indices) + 1:
            raise ValueError("Chain must alternate node/edge steps for same-path execution")
