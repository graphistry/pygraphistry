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
    node_columns: Tuple[str, ...] = ("id", "labels")
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
        graph=GraphFixture(
            nodes=[
                {"id": "a", "labels": ["A"]},
                {"id": "b", "labels": ["B"], "name": "b"},
                {"id": "c", "name": "c"},
            ],
            edges=[],
        ),
        expected=Expected(
            node_ids=["a", "b", "c"],
            rows=[
                {"n": "(:A)"},
                {"n": "(:B {name: 'b'})"},
                {"n": "({name: 'c'})"},
            ],
        ),
        gfql=[n()],
        tags=("match", "return", "manual-graph"),
    ),
    Scenario(
        key="match1-3",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[3] Matching nodes using multiple labels",
        cypher="MATCH (a:A:B)\nRETURN a",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "labels": ["A", "B", "C"]},
                {"id": "n2", "labels": ["A", "B"]},
                {"id": "n3", "labels": ["A", "C"]},
                {"id": "n4", "labels": ["B", "C"]},
                {"id": "n5", "labels": ["A"]},
                {"id": "n6", "labels": ["B"]},
                {"id": "n7", "labels": ["C"]},
                {"id": "n8", "name": ":A:B:C"},
                {"id": "n9", "abc": "abc"},
                {"id": "n10"},
            ],
            edges=[],
        ),
        expected=Expected(node_ids=["n1", "n2"]),
        gfql=[n({"label__A": True, "label__B": True})],
        tags=("match", "labels", "manual-graph"),
    ),
    Scenario(
        key="match1-4",
        feature_path="tck/features/clauses/match/Match1.feature",
        scenario="[4] Simple node inline property predicate",
        cypher="MATCH (n {name: 'bar'})\nRETURN n",
        graph=GraphFixture(
            nodes=[
                {"id": "n1", "name": "bar"},
                {"id": "n2", "name": "monkey"},
                {"id": "n3", "firstname": "bar"},
            ],
            edges=[],
        ),
        expected=Expected(
            node_ids=["n1"],
            rows=[{"n": "({name: 'bar'})"}],
        ),
        gfql=[n({"name": "bar"})],
        tags=("match", "property", "inline-predicate"),
    ),
]
