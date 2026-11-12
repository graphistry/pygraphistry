from __future__ import annotations

import os
import pandas as pd
import pytest
from typing import Any

import graphistry
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError, GFQLValidationError


def _records(frame: Any) -> list[dict[str, Any]]:
    if hasattr(frame, "to_dicts"):
        return frame.to_dicts()
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas().to_dict(orient="records")
    return frame.to_dict(orient="records")


ENGINES = [
    "pandas",
    pytest.param("cudf", marks=pytest.mark.skipif(
        os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1"
    )),
    "polars",
    pytest.param("polars-gpu", marks=pytest.mark.skipif(
        os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu"
    )),
]


def _frames_for_engine(engine: str, *frames: pd.DataFrame) -> tuple[Any, ...]:
    if engine == "cudf":
        import cudf
        return tuple(cudf.from_pandas(frame) for frame in frames)
    if engine in {"polars", "polars-gpu"}:
        pl = pytest.importorskip("polars")
        return tuple(pl.from_pandas(frame) for frame in frames)
    return frames


@pytest.mark.parametrize("engine", ENGINES)
def test_missing_relationship_property_does_not_return_alias_marker(engine: str) -> None:
    nodes, missing_edges, present_edges = _frames_for_engine(
        engine,
        pd.DataFrame({"id": ["a", "b"]}),
        pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["X"]}),
        pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": ["edge-value"]
        }),
    )
    boolean_edges, nullable_edges = _frames_for_engine(engine, pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": [False]
    }), pd.DataFrame({
        "s": ["a"], "d": ["b"], "type": ["X"], "r": [None]
    }))

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
    if engine in {"polars", "polars-gpu"}:
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


@pytest.mark.parametrize("engine", ENGINES)
def test_boolean_node_property_shadowing_alias_is_restored(engine: str) -> None:
    nodes, edges = _frames_for_engine(engine, pd.DataFrame({
        "id": ["a", "b"], "label__P": [True, True], "kind": [False, True]
    }), pd.DataFrame({"s": ["a"], "d": ["b"], "type": ["X"]}))
    g = graphistry.bind(source="s", destination="d", node="id", edge="type").nodes(nodes).edges(edges)

    for query in (
        "MATCH (kind:P) RETURN kind.kind AS x",
        "MATCH (kind:P) WITH kind RETURN kind.kind AS x",
    ):
        assert _records(g.gfql(query, engine=engine)._nodes) == [
            {"x": False}, {"x": True}
        ]

    # A traversed Boolean property remains a real property after a WITH re-entry too.
    assert _records(g.gfql(
        "MATCH (kind:P) WITH kind MATCH (kind)-[r:X]->(b) RETURN kind.kind AS x",
        engine=engine,
    )._nodes) == [{"x": False}]

    # Negative control: without a real same-named property, the generated Boolean
    # alias marker stays internal and resolves as a missing property.
    (marker_only_nodes,) = _frames_for_engine(
        engine, pd.DataFrame({"id": ["a", "b"], "label__P": [True, True]})
    )
    marker_only = graphistry.bind(
        source="s", destination="d", node="id", edge="type"
    ).nodes(marker_only_nodes).edges(edges)
    assert _records(
        marker_only.gfql("MATCH (a:P) RETURN a.a AS x", engine=engine)._nodes
    ) == [{"x": None}, {"x": None}]
