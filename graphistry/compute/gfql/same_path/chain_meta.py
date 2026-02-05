from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING
from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject

if TYPE_CHECKING:
    from graphistry.compute.gfql.df_executor import AliasBinding


@dataclass(frozen=True)
class ChainMeta:
    node_indices: List[int]
    edge_indices: List[int]
    step_to_alias: Dict[int, str]

    @staticmethod
    def from_chain(chain: Sequence[ASTObject], alias_bindings: Dict[str, "AliasBinding"]) -> "ChainMeta":
        node_indices = [i for i, op in enumerate(chain) if isinstance(op, ASTNode)]
        edge_indices = [i for i, op in enumerate(chain) if isinstance(op, ASTEdge)]
        step_to_alias = {b.step_index: alias for alias, b in alias_bindings.items()}
        return ChainMeta(node_indices=node_indices, edge_indices=edge_indices, step_to_alias=step_to_alias)

    def alias_for_step(self, step_index: int) -> Optional[str]:
        return self.step_to_alias.get(step_index)

    def are_steps_adjacent_nodes(self, step1: int, step2: int) -> bool:
        return abs(step1 - step2) == 2

    def validate(self) -> None:
        if not self.node_indices:
            raise ValueError("Same-path executor requires at least one node step")
        if len(self.node_indices) != len(self.edge_indices) + 1:
            raise ValueError("Chain must alternate node/edge steps for same-path execution")
