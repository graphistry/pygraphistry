from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class GraphFixture:
    nodes: Sequence[Dict[str, Any]]
    edges: Sequence[Dict[str, Any]]
    node_id: str = "id"
    src: str = "src"
    dst: str = "dst"
    edge_id: str = "edge_id"
    node_columns: Tuple[str, ...] = ("id", "labels")
    edge_columns: Tuple[str, ...] = ("src", "dst", "edge_id", "type")


@dataclass(frozen=True)
class Expected:
    node_ids: Optional[Sequence[Any]] = None
    edge_ids: Optional[Sequence[Any]] = None
    rows: Optional[List[Dict[str, Any]]] = None


@dataclass(frozen=True)
class Scenario:
    key: str
    feature_path: str
    scenario: str
    cypher: str
    graph: GraphFixture
    expected: Expected
    gfql: Optional[Sequence[Any]]
    status: str = "supported"
    reason: Optional[str] = None
    tags: Tuple[str, ...] = ()
    return_alias: Optional[str] = None
