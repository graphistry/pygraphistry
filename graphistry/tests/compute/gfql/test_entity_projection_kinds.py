"""Explicit whole-entity provenance preserves the row contract across engines."""
import os
import pandas as pd
import pytest

import graphistry
from graphistry.Engine import Engine, df_to_engine

ENGINES = [
    "pandas", "polars",
    pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="requires TEST_CUDF=1")),
    pytest.param("polars-gpu", marks=pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires TEST_POLARS_GPU=1")),
]


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("labelled", [False, True])
@pytest.mark.parametrize("projection", ["x", "x AS renamed", "DISTINCT x", "x.name", "count(DISTINCT x) AS c"])
def test_entity_projection_kind_and_identity_are_independent(engine, labelled, projection):
    if engine.startswith("polars"):
        pytest.importorskip("polars")
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "name": ["A", "B", "same", "same"]})
    if labelled:
        nodes = nodes.assign(label__Person=True)
    edges = pd.DataFrame({"s": [0, 0, 1, 1], "d": [2, 3, 2, 3]})
    g = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(df_to_engine(edges, Engine(engine)), "s", "d")
    query = "MATCH (a {name: 'A'}), (b {name: 'B'}) MATCH (a)-->(x)<-->(b) RETURN " + projection
    out = g.gfql(query, engine=engine)
    frame = out._nodes.to_pandas() if hasattr(out._nodes, "to_pandas") else out._nodes
    kinds = getattr(out, "_cypher_entity_projection_kinds", {})
    if projection == "count(DISTINCT x) AS c":
        assert frame.to_dict("records") == [{"c": 2}]
        assert kinds == {}
    elif projection == "x.name":
        assert frame.to_dict("records") == [{"x.name": "same"}, {"x.name": "same"}]
        assert kinds == {}
    else:
        alias = "renamed" if projection == "x AS renamed" else "x"
        assert kinds == {alias: "nodes"}
        assert len(frame) == 2
        assert frame[f"{alias}.name"].tolist() == ["same", "same"]
        if projection == "DISTINCT x":
            assert sorted(frame[f"{alias}.id"].tolist()) == [2, 3]
    assert not hasattr(g, "_cypher_entity_projection_kinds")
