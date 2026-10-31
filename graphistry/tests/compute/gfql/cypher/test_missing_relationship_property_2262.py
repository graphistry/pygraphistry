from __future__ import annotations

import pandas as pd
import pytest
from typing import Any

import graphistry


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
    if pl is not None:
        nodes = pl.from_pandas(nodes)
        missing_edges = pl.from_pandas(missing_edges)
        present_edges = pl.from_pandas(present_edges)
        boolean_edges = pl.from_pandas(boolean_edges)

    def graph(edges: Any) -> Any:
        return (
            graphistry.bind(source="s", destination="d", node="id", edge="type")
            .nodes(nodes)
            .edges(edges)
        )

    missing_graph = graph(missing_edges)
    present_graph = graph(present_edges)
    boolean_graph = graph(boolean_edges)

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
