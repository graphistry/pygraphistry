from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graphistry.compute import n


@dataclass(frozen=True)
class GraphFixture:
    nodes: Sequence[Dict[str, Any]]
    edges: Sequence[Dict[str, Any]]
    node_id: str = "id"
    src: str = "src"
    dst: str = "dst"
    edge_id: str = "edge_id"
    node_columns: Tuple[str, ...] = ("id",)
    edge_columns: Tuple[str, ...] = ("src", "dst", "edge_id")


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


SCENARIOS = [
    Scenario(
        key="match1-1",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[1] Match non-existent nodes returns empty",
        cypher="MATCH (n)\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(node_ids=[]),
        gfql=[n()],
        tags=("match", "return", "empty-graph"),
    ),
    Scenario(
        key="match1-2",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[2] Matching all nodes",
        cypher="MATCH (n)\nRETURN n",
        graph=GraphFixture(nodes=[], edges=[]),
        expected=Expected(
            rows=[
                {"n": "(:A)"},
                {"n": "(:B {name: 'b'})"},
                {"n": "({name: 'c'})"},
            ]
        ),
        gfql=None,
        status="xfail",
        reason="Requires CREATE setup parsing for TCK graph initialization",
        tags=("match", "return", "create", "tck-setup"),
    ),
]
