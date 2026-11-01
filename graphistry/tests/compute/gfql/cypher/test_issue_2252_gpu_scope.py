from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest

import graphistry


ENGINES = [
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu"
    )),
]


def _records(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas().to_dict(orient="records")
    return frame.to_dict(orient="records")


def _assert_served(engine: str, frame: Any) -> None:
    assert type(frame).__module__.startswith(
        "cudf" if engine == "cudf" else "polars"
    )


def _graph(engine: str, nodes: pd.DataFrame, edges: pd.DataFrame) -> Any:
    if engine == "cudf":
        import cudf
        nodes = cudf.from_pandas(nodes)
        edges = cudf.from_pandas(edges)
    else:
        import polars
        nodes = polars.from_pandas(nodes)
        edges = polars.from_pandas(edges)
    return graphistry.bind(source="s", destination="d", node="id", edge="type").nodes(nodes).edges(edges)


def _linear_graph(engine: str) -> Any:
    nodes = pd.DataFrame({
        "id": ["a1", "a2", "b1", "b2"],
        "label__A": [True, True, False, False],
        "label__B": [False, False, True, True],
        "num": [1, 2, 1, 3],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a2"], "d": ["b1", "b2"], "type": ["R", "R"]
    })
    return _graph(engine, nodes, edges)


@pytest.mark.parametrize("engine", ENGINES)
def test_whole_plus_scalar_prefix_survives_match(engine: str) -> None:
    result = _linear_graph(engine).gfql(
        "MATCH (a:A) WITH a, a.num AS property MATCH (a)-->(b:B) "
        "RETURN property, count(DISTINCT b.id) AS messageCount ORDER BY property",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [
        {"property": 1.0, "messageCount": 1},
        {"property": 2.0, "messageCount": 1},
    ]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_only_prefix_survives_match(engine: str) -> None:
    result = _linear_graph(engine).gfql(
        "MATCH (a:A) WITH a.num AS property MATCH (a)-->(b:B) "
        "RETURN property, count(DISTINCT b.id) AS messageCount ORDER BY property",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [
        {"property": 1.0, "messageCount": 2},
        {"property": 2.0, "messageCount": 2},
    ]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_connected_prefix_and_filtered_suffix_stay_path_scoped(engine: str) -> None:
    nodes = pd.DataFrame({
        "id": ["a1", "a2", "b1", "b2", "c1", "c2"],
        "label__A": [True, True, False, False, False, False],
        "label__B": [False, False, True, True, False, False],
        "label__C": [False, False, False, False, True, True],
        "score": [None, None, 10, 20, None, None],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a2", "b1", "b2"],
        "d": ["b1", "b2", "c1", "c1"],
        "type": ["R", "R", "S", "S"],
        "weight": [None, None, 10, 20],
    })
    graph = _graph(engine, nodes, edges)
    positive = graph.gfql(
        "MATCH (a:A)-[:R]->(b:B) WITH b, b.score AS property "
        "MATCH (b)-[:S {weight: 10}]->(c:C) "
        "RETURN property, count(DISTINCT c.id) AS messageCount ORDER BY property",
        engine=engine,
        language="cypher",
    )
    empty = graph.gfql(
        "MATCH (a:A)-[:R]->(b:B) WITH b, b.score AS property "
        "MATCH (b)-[:S {weight: 99}]->(c:C) "
        "RETURN property, count(DISTINCT c.id) AS messageCount",
        engine=engine,
        language="cypher",
    )
    assert _records(positive._nodes) == [{"property": 10.0, "messageCount": 1}]
    _assert_served(engine, positive._nodes)
    assert _records(empty._nodes) == []
    _assert_served(engine, empty._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_edge_prefix_keeps_path_scalar(engine: str) -> None:
    nodes = pd.DataFrame({
        "id": ["a1", "a2", "b1", "b2", "c1", "c2"],
        "label__A": [True, True, False, False, False, False],
        "label__B": [False, False, True, True, False, False],
        "label__C": [False, False, False, False, True, True],
        "num": [1, 2, None, None, None, None],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a2", "b1", "b2"],
        "d": ["b1", "b2", "c1", "c2"],
        "type": ["R", "R", "S", "S"],
    })
    result = _graph(engine, nodes, edges).gfql(
        "MATCH (a:A) WITH a, a.num AS p "
        "MATCH (a)-[:R]->(b:B)-[:S]->(c:C) "
        "RETURN p, c.id AS cid ORDER BY p",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [
        {"p": 1.0, "cid": "c1"},
        {"p": 2.0, "cid": "c2"},
    ]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_multi_edge_prefix_negative_is_empty(engine: str) -> None:
    nodes = pd.DataFrame({
        "id": ["a1", "a2", "b1", "b2", "c1", "c2"],
        "label__A": [True, True, False, False, False, False],
        "label__B": [False, False, True, True, False, False],
        "label__C": [False, False, False, False, True, True],
        "num": [1, 2, None, None, None, None],
    })
    edges = pd.DataFrame({
        "s": ["a1", "a2", "b1", "b2"],
        "d": ["b1", "b2", "c1", "c2"],
        "type": ["R", "R", "S", "S"],
    })
    result = _graph(engine, nodes, edges).gfql(
        "MATCH (a:A {num: 99}) WITH a, a.num AS p "
        "MATCH (a)-[:R]->(b:B)-[:S]->(c:C) RETURN p, c.id AS cid",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == []
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_scalar_only_prefix_negative_is_empty(engine: str) -> None:
    result = _linear_graph(engine).gfql(
        "MATCH (a:A {num: 99}) WITH a.num AS property MATCH (a)-->(b:B) "
        "RETURN property, count(DISTINCT b.id) AS messageCount",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == []
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_no_reentry_match_retains_multiplicity(engine: str) -> None:
    result = _linear_graph(engine).gfql(
        "MATCH (a:A)-[:R]->(b:B) RETURN count(*) AS pathCount",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [{"pathCount": 2}]
    _assert_served(engine, result._nodes)


@pytest.mark.parametrize("engine", ENGINES)
def test_typed_carried_scalar_survives_match(engine: str) -> None:
    nodes = pd.DataFrame({
        "id": ["a1", "a2", "b1", "b2"],
        "label__A": [True, True, False, False],
        "label__B": [False, False, True, True],
        "num": [1, 2, 1, 3],
        "ts": pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"]),
    })
    edges = pd.DataFrame({
        "s": ["a1", "a2"], "d": ["b1", "b2"], "type": ["R", "R"]
    })
    result = _graph(engine, nodes, edges).gfql(
        "MATCH (a:A) WITH a.ts AS event MATCH (a)-->(b:B) "
        "RETURN event, count(DISTINCT b.id) AS messageCount",
        engine=engine,
        language="cypher",
    )
    assert _records(result._nodes) == [
        {"event": pd.Timestamp("2020-01-01"), "messageCount": 2},
        {"event": pd.Timestamp("2020-01-02"), "messageCount": 2},
    ]
    _assert_served(engine, result._nodes)


@pytest.mark.skipif(
    os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
)
def test_whole_relationship_return_keeps_property_columns_on_cudf() -> None:
    nodes = pd.DataFrame({"id": ["a", "b"]})
    edges = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["X"], "r": [None]})
    graph = _graph("cudf", nodes, edges)
    result = graph.gfql("MATCH (a)-[r:X]->(b) RETURN r AS edge", engine="cudf")
    assert _records(result._nodes) == [{"edge.type": "X"}]
    _assert_served("cudf", result._nodes)
