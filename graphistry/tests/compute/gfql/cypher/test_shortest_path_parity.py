from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, cast

import pandas as pd
import pytest

from graphistry.compute.exceptions import GFQLValidationError
from graphistry.compute.gfql.cypher import compile_cypher, parse_cypher
from graphistry.tests.test_compute import CGFull


class _CypherTestGraph(CGFull):
    _dgl_graph = None

    def search_graph(
        self,
        query: str,
        scale: float = 0.5,
        top_n: int = 100,
        thresh: float = 5000,
        broader: bool = False,
        inplace: bool = False,
    ):
        raise NotImplementedError

    def search(self, query: str, cols=None, thresh: float = 5000, fuzzy: bool = True, top_n: int = 10):
        raise NotImplementedError

    def embed(self, relation: str, *args, **kwargs):
        raise NotImplementedError


def _mk_graph(nodes_df: pd.DataFrame, edges_df: pd.DataFrame) -> _CypherTestGraph:
    return cast(_CypherTestGraph, _CypherTestGraph().nodes(nodes_df, "id").edges(edges_df, "s", "d"))


def _mk_empty_graph() -> _CypherTestGraph:
    return _mk_graph(pd.DataFrame({"id": []}), pd.DataFrame({"s": [], "d": []}))

@pytest.mark.parametrize(
    "query",
    [
        "MATCH path = allShortestPaths((a)-[:KNOWS*]-(b)) RETURN length(path)",
    ],
)
def test_string_cypher_failfast_rejects_shortest_path(query: str) -> None:
    """allShortestPaths remains out of scope for the local compiler."""
    graph = _mk_empty_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)
    assert "allshortestpaths" in exc_info.value.message.lower()


@pytest.mark.parametrize(
    "query",
    [
        "MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN path",
        "MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN relationships(path) AS rels",
        "MATCH path = shortestPath((a)-[:KNOWS*]-(b)) RETURN length(path) AS n ORDER BY path",
    ],
)
def test_string_cypher_failfast_rejects_shortest_path_carrier_forms(query: str) -> None:
    graph = _mk_empty_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)
    assert "shortestpath" in exc_info.value.message.lower()
    assert "length(path)" in exc_info.value.message.lower() or "path is null" in exc_info.value.message.lower()


def test_string_cypher_executes_shortest_path_length_case_official_shape() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: $person1Id}),
            (person2:Person {id: $person2Id}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength
        """,
        params={"person1Id": "p1", "person2Id": "p3"},
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 2}]


def test_string_cypher_executes_shortest_path_length_case_disconnected_returns_minus_one() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: $person1Id}),
            (person2:Person {id: $person2Id}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength
        """,
        params={"person1Id": "p1", "person2Id": "p3"},
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": -1}]


def test_string_cypher_executes_bounded_shortest_path_length_prefers_shorter_alternative() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "p4"],
                "d": ["p2", "p4", "p3"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 2}]


def test_string_cypher_executes_shortest_path_is_null_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN path IS NULL AS noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"noPath": True}]


def test_string_cypher_executes_shortest_path_is_null_projection_when_reachable() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN path IS NULL AS noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"noPath": False}]


def test_string_cypher_executes_shortest_path_length_projection_when_disconnected() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": None}]


def test_string_cypher_executes_shortest_path_reverse_direction_length_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p3'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 2}]


def test_string_cypher_executes_shortest_path_endpoint_projection_with_length() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2}
    ]


def test_string_cypher_executes_shortest_path_disconnected_endpoint_projection_with_length() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": None, "noPath": True}
    ]


def test_string_cypher_executes_shortest_path_with_length_stage_and_order_by() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2}
    ]


def test_string_cypher_executes_shortest_path_disconnected_with_is_null_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH path IS NULL AS noPath
        RETURN noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"noPath": True}]


def test_string_cypher_executes_shortest_path_disconnected_with_length_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH length(path) AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": None}]


def test_string_cypher_executes_shortest_path_disconnected_with_endpoint_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        RETURN person1Id, person2Id, shortestPathLength, noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": None, "noPath": True}
    ]


def test_string_cypher_executes_bounded_shortest_path_disconnected_with_length_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        WITH length(path) AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": None}]


def test_string_cypher_executes_shortest_path_disconnected_with_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": -1}]


def test_string_cypher_executes_reverse_shortest_path_disconnected_with_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p2"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p3'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WITH CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": -1}]


def test_string_cypher_executes_reverse_shortest_path_with_endpoint_and_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p3'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WITH person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person2Id, shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person2Id": "p1", "shortestPathLength": 2}
    ]


def test_string_cypher_executes_shortest_path_zero_hop_length() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 0}]


def test_string_cypher_executes_shortest_path_zero_hop_case_and_is_null() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength,
            path IS NULL AS noPath
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"shortestPathLength": 0, "noPath": False}
    ]


def test_string_cypher_executes_shortest_path_zero_hop_with_length_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH length(path) AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 0}]


def test_string_cypher_executes_reverse_shortest_path_zero_hop_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3"],
                "label__Person": [True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2"],
                "d": ["p2", "p3"],
                "type": ["KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WITH CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 0}]


def test_string_cypher_executes_shortest_path_on_cyclic_graph_with_multiple_shortest_routes() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "p2", "p3"],
                "d": ["p2", "p3", "p4", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p4'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 2}]


def test_string_cypher_executes_shortest_path_stage_on_cyclic_graph_with_multiple_shortest_routes() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "p2", "p3"],
                "d": ["p2", "p3", "p4", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p4'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p4", "shortestPathLength": 2}
    ]


def test_string_cypher_executes_bounded_shortest_path_on_cyclic_graph_with_multiple_shortest_routes() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p1", "p2", "p3"],
                "d": ["p2", "p3", "p4", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p4'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 2}]


def test_string_cypher_executes_bounded_shortest_path_on_self_loop_without_duplicate_rows() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1"],
                "label__Person": [True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1"],
                "d": ["p1"],
                "type": ["KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        """,
    )

    assert result._nodes.to_dict(orient="records") == [{"shortestPathLength": 1}]


def test_string_cypher_executes_multirow_shortest_path_pairs_without_zero_hop_collapse() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p4']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2},
        {"person1Id": "p1", "person2Id": "p4", "shortestPathLength": 3},
        {"person1Id": "p2", "person2Id": "p3", "shortestPathLength": 1},
        {"person1Id": "p2", "person2Id": "p4", "shortestPathLength": 2},
    ]


def test_string_cypher_executes_multirow_bounded_shortest_path_pairs_without_endpoint_hop_collapse() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4"],
                "label__Person": [True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*1..2]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p4']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2},
        {"person1Id": "p1", "person2Id": "p4", "shortestPathLength": -1},
        {"person1Id": "p2", "person2Id": "p3", "shortestPathLength": 1},
        {"person1Id": "p2", "person2Id": "p4", "shortestPathLength": 2},
    ]


def test_string_cypher_executes_multirow_disconnected_shortest_path_endpoint_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p5']
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
        {"person1Id": "p2", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
    ]


def test_string_cypher_executes_multirow_disconnected_shortest_path_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p5']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p5", "shortestPathLength": -1},
        {"person1Id": "p2", "person2Id": "p5", "shortestPathLength": -1},
    ]


def test_string_cypher_executes_multirow_bounded_disconnected_shortest_path_endpoint_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*1..2]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p5']
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
        {"person1Id": "p2", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
    ]


def test_string_cypher_executes_multirow_bounded_disconnected_shortest_path_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*1..2]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p5']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p5", "shortestPathLength": -1},
        {"person1Id": "p2", "person2Id": "p5", "shortestPathLength": -1},
    ]


def test_string_cypher_executes_multirow_disconnected_reverse_shortest_path_endpoint_projection() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p5'] AND person2.id IN ['p1', 'p2']
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p5", "person2Id": "p1", "shortestPathLength": None, "noPath": True},
        {"person1Id": "p5", "person2Id": "p2", "shortestPathLength": None, "noPath": True},
    ]


def test_string_cypher_executes_multirow_disconnected_reverse_shortest_path_case_stage() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p5'] AND person2.id IN ['p1', 'p2']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p5", "person2Id": "p1", "shortestPathLength": -1},
        {"person1Id": "p5", "person2Id": "p2", "shortestPathLength": -1},
    ]


def test_string_cypher_executes_multirow_shortest_path_is_null_ordering() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p5']
        RETURN person1.id AS person1Id, person2.id AS person2Id, path IS NULL AS noPath
        ORDER BY noPath, person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "noPath": False},
        {"person1Id": "p2", "person2Id": "p3", "noPath": False},
        {"person1Id": "p1", "person2Id": "p5", "noPath": True},
        {"person1Id": "p2", "person2Id": "p5", "noPath": True},
    ]


def test_string_cypher_executes_multirow_bounded_shortest_path_is_null_stage_ordering() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*1..2]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p4', 'p5']
        WITH person1.id AS person1Id, person2.id AS person2Id, path IS NULL AS noPath
        RETURN person1Id, person2Id, noPath
        ORDER BY noPath, person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p1", "person2Id": "p3", "noPath": False},
        {"person1Id": "p2", "person2Id": "p3", "noPath": False},
        {"person1Id": "p2", "person2Id": "p4", "noPath": False},
        {"person1Id": "p1", "person2Id": "p4", "noPath": True},
        {"person1Id": "p1", "person2Id": "p5", "noPath": True},
        {"person1Id": "p2", "person2Id": "p5", "noPath": True},
    ]


def test_string_cypher_executes_multirow_reverse_shortest_path_is_null_stage_ordering() -> None:
    graph = _mk_graph(
        pd.DataFrame(
            {
                "id": ["p1", "p2", "p3", "p4", "p5"],
                "label__Person": [True, True, True, True, True],
            }
        ),
        pd.DataFrame(
            {
                "s": ["p1", "p2", "p3"],
                "d": ["p2", "p3", "p4"],
                "type": ["KNOWS", "KNOWS", "KNOWS"],
            }
        ),
    )

    result = graph.gfql(
        """
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p3', 'p5'] AND person2.id IN ['p1', 'p2']
        WITH person1.id AS person1Id, person2.id AS person2Id, path IS NULL AS noPath
        RETURN person1Id, person2Id, noPath
        ORDER BY noPath, person1Id, person2Id
        """,
    )

    assert result._nodes.to_dict(orient="records") == [
        {"person1Id": "p3", "person2Id": "p1", "noPath": False},
        {"person1Id": "p3", "person2Id": "p2", "noPath": False},
        {"person1Id": "p5", "person2Id": "p1", "noPath": True},
        {"person1Id": "p5", "person2Id": "p2", "noPath": True},
    ]



@pytest.mark.parametrize(
    "query,error_substring",
    [
        (
            "MATCH path = shortestPath((a)-[:KNOWS]-(b)) RETURN length(path) AS n",
            "variable-length relationship pattern",
        ),
        (
            "MATCH path = shortestPath((a)-[:KNOWS*]-()-[:KNOWS*]-(b)) RETURN length(path) AS n",
            "single-relationship path patterns",
        ),
    ],
)
def test_string_cypher_failfast_rejects_shortest_path_invalid_specs(query: str, error_substring: str) -> None:
    graph = _mk_empty_graph()
    with pytest.raises(GFQLValidationError) as exc_info:
        graph.gfql(query)
    assert error_substring in exc_info.value.message.lower()


def test_string_cypher_failfast_rejects_shortest_path_inside_optional_match() -> None:
    with pytest.raises(GFQLValidationError) as exc_info:
        compile_cypher(
            """
            MATCH (person1:Person {id: 'p1'})
            OPTIONAL MATCH
                (person2:Person {id: 'p2'}),
                path = shortestPath((person1)-[:KNOWS*]-(person2))
            RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
            """
        )
    msg = exc_info.value.message.lower()
    assert "optional match" in msg


@dataclass(frozen=True)
class _ExecParityCase:
    name: str
    graph_factory: Callable[[], _CypherTestGraph]
    query: str
    expected_rows: list[dict[str, object]]
    params: Optional[Mapping[str, object]] = None


def _mk_person_chain_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3"], "label__Person": [True, True, True]}),
        pd.DataFrame({"s": ["p1", "p2"], "d": ["p2", "p3"], "type": ["KNOWS", "KNOWS"]}),
    )


def _mk_person_disconnected_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3"], "label__Person": [True, True, True]}),
        pd.DataFrame({"s": ["p1"], "d": ["p2"], "type": ["KNOWS"]}),
    )


def _mk_person_shorter_alternative_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3", "p4"], "label__Person": [True, True, True, True]}),
        pd.DataFrame({"s": ["p1", "p1", "p4"], "d": ["p2", "p4", "p3"], "type": ["KNOWS", "KNOWS", "KNOWS"]}),
    )


def _mk_person_multirow_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3", "p4"], "label__Person": [True, True, True, True]}),
        pd.DataFrame({"s": ["p1", "p2", "p3"], "d": ["p2", "p3", "p4"], "type": ["KNOWS", "KNOWS", "KNOWS"]}),
    )


def _mk_person_multirow_with_isolate_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3", "p4", "p5"], "label__Person": [True, True, True, True, True]}),
        pd.DataFrame({"s": ["p1", "p2", "p3"], "d": ["p2", "p3", "p4"], "type": ["KNOWS", "KNOWS", "KNOWS"]}),
    )


def _mk_person_cycle_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1", "p2", "p3", "p4"], "label__Person": [True, True, True, True]}),
        pd.DataFrame({"s": ["p1", "p1", "p2", "p3"], "d": ["p2", "p3", "p4", "p4"], "type": ["KNOWS"] * 4}),
    )


def _mk_person_self_loop_graph() -> _CypherTestGraph:
    return _mk_graph(
        pd.DataFrame({"id": ["p1"], "label__Person": [True]}),
        pd.DataFrame({"s": ["p1"], "d": ["p1"], "type": ["KNOWS"]}),
    )


_PARITY_EXEC_CASES: tuple[_ExecParityCase, ...] = (
    _ExecParityCase(
        name="official-shape-case-length",
        graph_factory=_mk_person_chain_graph,
        query="""
        MATCH
            (person1:Person {id: $person1Id}),
            (person2:Person {id: $person2Id}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength
        """,
        params={"person1Id": "p1", "person2Id": "p3"},
        expected_rows=[{"shortestPathLength": 2}],
    ),
    _ExecParityCase(
        name="disconnected-case-minus-one",
        graph_factory=_mk_person_disconnected_graph,
        query="""
        MATCH
            (person1:Person {id: $person1Id}),
            (person2:Person {id: $person2Id}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength
        """,
        params={"person1Id": "p1", "person2Id": "p3"},
        expected_rows=[{"shortestPathLength": -1}],
    ),
    _ExecParityCase(
        name="bounded-prefers-shorter-path",
        graph_factory=_mk_person_shorter_alternative_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
        expected_rows=[{"shortestPathLength": 2}],
    ),
    _ExecParityCase(
        name="reverse-direction-length",
        graph_factory=_mk_person_chain_graph,
        query="""
        MATCH
            (person1:Person {id: 'p3'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
        expected_rows=[{"shortestPathLength": 2}],
    ),
    _ExecParityCase(
        name="multirow-bounded-pairs",
        graph_factory=_mk_person_multirow_graph,
        query="""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*1..2]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p4']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
        expected_rows=[
            {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2},
            {"person1Id": "p1", "person2Id": "p4", "shortestPathLength": -1},
            {"person1Id": "p2", "person2Id": "p3", "shortestPathLength": 1},
            {"person1Id": "p2", "person2Id": "p4", "shortestPathLength": 2},
        ],
    ),
    _ExecParityCase(
        name="is-null-disconnected",
        graph_factory=_mk_person_disconnected_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN path IS NULL AS noPath
        """,
        expected_rows=[{"noPath": True}],
    ),
    _ExecParityCase(
        name="is-null-reachable",
        graph_factory=_mk_person_chain_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN path IS NULL AS noPath
        """,
        expected_rows=[{"noPath": False}],
    ),
    _ExecParityCase(
        name="length-projection-disconnected-none",
        graph_factory=_mk_person_disconnected_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p3'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
        expected_rows=[{"shortestPathLength": None}],
    ),
    _ExecParityCase(
        name="zero-hop-case-and-is-null",
        graph_factory=_mk_person_chain_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN
            CASE path IS NULL
                WHEN true THEN -1
                ELSE length(path)
            END AS shortestPathLength,
            path IS NULL AS noPath
        """,
        expected_rows=[{"shortestPathLength": 0, "noPath": False}],
    ),
    _ExecParityCase(
        name="cyclic-multi-route-shortest-length",
        graph_factory=_mk_person_cycle_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p4'}),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        RETURN length(path) AS shortestPathLength
        """,
        expected_rows=[{"shortestPathLength": 2}],
    ),
    _ExecParityCase(
        name="bounded-self-loop-no-duplicate-rows",
        graph_factory=_mk_person_self_loop_graph,
        query="""
        MATCH
            (person1:Person {id: 'p1'}),
            (person2:Person {id: 'p1'}),
            path = shortestPath((person1)-[:KNOWS*1..3]-(person2))
        RETURN CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        """,
        expected_rows=[{"shortestPathLength": 1}],
    ),
    _ExecParityCase(
        name="multirow-unbounded-pairs",
        graph_factory=_mk_person_multirow_graph,
        query="""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p4']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
        expected_rows=[
            {"person1Id": "p1", "person2Id": "p3", "shortestPathLength": 2},
            {"person1Id": "p1", "person2Id": "p4", "shortestPathLength": 3},
            {"person1Id": "p2", "person2Id": "p3", "shortestPathLength": 1},
            {"person1Id": "p2", "person2Id": "p4", "shortestPathLength": 2},
        ],
    ),
    _ExecParityCase(
        name="multirow-disconnected-endpoint-projection",
        graph_factory=_mk_person_multirow_with_isolate_graph,
        query="""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p5']
        RETURN person1.id AS person1Id, person2.id AS person2Id, length(path) AS shortestPathLength, path IS NULL AS noPath
        ORDER BY person1Id, person2Id
        """,
        expected_rows=[
            {"person1Id": "p1", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
            {"person1Id": "p2", "person2Id": "p5", "shortestPathLength": None, "noPath": True},
        ],
    ),
    _ExecParityCase(
        name="multirow-reverse-disconnected-case-stage",
        graph_factory=_mk_person_multirow_with_isolate_graph,
        query="""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)<-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p5'] AND person2.id IN ['p1', 'p2']
        WITH person1.id AS person1Id, person2.id AS person2Id, CASE path IS NULL WHEN true THEN -1 ELSE length(path) END AS shortestPathLength
        RETURN person1Id, person2Id, shortestPathLength
        ORDER BY person1Id, person2Id
        """,
        expected_rows=[
            {"person1Id": "p5", "person2Id": "p1", "shortestPathLength": -1},
            {"person1Id": "p5", "person2Id": "p2", "shortestPathLength": -1},
        ],
    ),
    _ExecParityCase(
        name="multirow-is-null-ordering",
        graph_factory=_mk_person_multirow_with_isolate_graph,
        query="""
        MATCH
            (person1:Person),
            (person2:Person),
            path = shortestPath((person1)-[:KNOWS*]-(person2))
        WHERE person1.id IN ['p1', 'p2'] AND person2.id IN ['p3', 'p5']
        RETURN person1.id AS person1Id, person2.id AS person2Id, path IS NULL AS noPath
        ORDER BY noPath, person1Id, person2Id
        """,
        expected_rows=[
            {"person1Id": "p1", "person2Id": "p3", "noPath": False},
            {"person1Id": "p2", "person2Id": "p3", "noPath": False},
            {"person1Id": "p1", "person2Id": "p5", "noPath": True},
            {"person1Id": "p2", "person2Id": "p5", "noPath": True},
        ],
    ),
)


def _assert_case_via_lowering(case: _ExecParityCase) -> None:
    graph = case.graph_factory()
    result = graph.gfql(case.query, params=case.params)
    assert result._nodes.to_dict(orient="records") == case.expected_rows


def _assert_case_via_ast_normalizer(case: _ExecParityCase) -> None:
    try:
        from graphistry.compute.gfql.cypher.ast_normalizer import ASTNormalizer  # type: ignore[attr-defined]
    except Exception:
        pytest.xfail("ASTNormalizer path is not available pre-M1")

    parsed = parse_cypher(case.query)
    _ = ASTNormalizer().normalize(parsed)
    pytest.xfail("ASTNormalizer -> Binder pass 2 -> lowering execution path is pending M1")


@pytest.mark.parametrize("case", _PARITY_EXEC_CASES, ids=[c.name for c in _PARITY_EXEC_CASES])
def test_shortest_path_parity_via_lowering(case: _ExecParityCase) -> None:
    _assert_case_via_lowering(case)


@pytest.mark.parametrize("case", _PARITY_EXEC_CASES, ids=[c.name for c in _PARITY_EXEC_CASES])
@pytest.mark.xfail(reason="ASTNormalizer parity path is pending M1", strict=False)
def test_shortest_path_parity_via_ast_normalizer(case: _ExecParityCase) -> None:
    _assert_case_via_ast_normalizer(case)
