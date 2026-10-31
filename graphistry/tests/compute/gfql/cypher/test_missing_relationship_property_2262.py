from __future__ import annotations

import pandas as pd
import pytest
from typing import Any

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLValidationError


def _records(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    return frame.to_dict(orient="records")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_missing_relationship_property_does_not_return_alias_marker(engine: str) -> None:
    pl = pytest.importorskip("polars") if engine == "polars" else None
    nodes: Any = pd.DataFrame({"id": ["a", "b"]})
    missing_edges: Any = pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["X"]})
    present_edges: Any = pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": ["edge-value"]
    })
    boolean_edges: Any = pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": [False]
    })
    nullable_edges: Any = pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": [None]
    })
    if pl is not None:
        nodes = pl.from_pandas(nodes)
        missing_edges = pl.from_pandas(missing_edges)
        present_edges = pl.from_pandas(present_edges)
        boolean_edges = pl.from_pandas(boolean_edges)
        nullable_edges = pl.from_pandas(nullable_edges)

    def graph(edges: Any) -> Any:
        return (
            graphistry.bind(source="s", destination="d", node="id", edge="type")
            .nodes(nodes)
            .edges(edges)
        )

    missing_graph = graph(missing_edges)
    present_graph = graph(present_edges)
    boolean_graph = graph(boolean_edges)
    nullable_graph = graph(nullable_edges)

    missing_return = missing_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r.r AS x", engine=engine)
    assert _records(missing_return._nodes) == [{"x": None}]
    assert _records(
        missing_graph.gfql(
            "MATCH (a)-[r:X]->(b) WHERE r.r IS NULL RETURN b.id AS id", engine=engine
        )._nodes
    ) == [{"id": "b"}]
    assert _records(
        missing_graph.gfql(
            "MATCH (a)-[r:X]->(b) WHERE r.r = 'edge-value' RETURN b.id AS id", engine=engine
        )._nodes
    ) == []

    assert _records(
        present_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r.r AS x", engine=engine)._nodes
    ) == [{"x": "edge-value"}]
    assert _records(
        boolean_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r.r AS x", engine=engine)._nodes
    ) == [{"x": False}]
    assert _records(
        missing_graph.gfql("MATCH (a)-[r:X]->(b) WITH r RETURN r.r AS x", engine=engine)._nodes
    ) == [{"x": None}]
    assert _records(
        boolean_graph.gfql("MATCH (a)-[r:X]->(b) WITH r RETURN r.r AS x", engine=engine)._nodes
    ) == [{"x": False}]
    assert _records(
        present_graph.gfql("MATCH (a)-[r:X]->(b) WITH r RETURN r.r AS x", engine=engine)._nodes
    ) == [{"x": "edge-value"}]

    with pytest.raises(GFQLSchemaError):
        missing_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r.r AS x", engine=engine, strict=True)
    assert _records(
        nullable_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r.r AS x", engine=engine, strict=True)._nodes
    ) == [{"x": None}]
    assert _records(
        present_graph.gfql(
            "MATCH (a)-[r:X]->(b) WHERE r.r IS NULL RETURN b.id AS id", engine=engine
        )._nodes
    ) == []
    assert _records(
        present_graph.gfql(
            "MATCH (a)-[r:X]->(b) WHERE r.r = 'edge-value' RETURN b.id AS id", engine=engine
        )._nodes
    ) == [{"id": "b"}]

    noncolliding_missing = "MATCH (a)-[rel:X]->(b) RETURN rel.r AS x"
    if engine == "polars":
        with pytest.raises(NotImplementedError, match="row op 'select'"):
            missing_graph.gfql(noncolliding_missing, engine=engine)
    else:
        assert _records(missing_graph.gfql(noncolliding_missing, engine=engine)._nodes) == [
            {"x": None}
        ]
    assert _records(
        missing_graph.gfql(
            "MATCH (a)-[rel:X]->(b) WHERE rel.r IS NULL RETURN b.id AS id", engine=engine
        )._nodes
    ) == [{"id": "b"}]
    assert _records(
        present_graph.gfql(
            "MATCH (a)-[rel:X]->(b) WHERE rel.r = 'edge-value' RETURN b.id AS id", engine=engine
        )._nodes
    ) == [{"id": "b"}]

    if engine == "pandas":
        relationship = missing_graph.gfql("MATCH (a)-[r:X]->(b) RETURN r AS edge", engine=engine)
        assert _records(relationship._nodes) == [{"edge.type": "X"}]

    with pytest.raises(GFQLValidationError) as exc_info:
        missing_graph.gfql(
            "MATCH (a)-[r:X]->(b) WITH a, r "
            "MATCH (a)-[r2:X]->(b2) RETURN r2.r AS x",
            engine=engine,
        )
    assert exc_info.value.code == ErrorCode.E108
