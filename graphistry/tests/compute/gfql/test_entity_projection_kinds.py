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


@pytest.fixture(params=ENGINES)
def engine(request):
    name = request.param
    if name.startswith("polars"):
        pytest.importorskip("polars")
    return name


@pytest.mark.parametrize("labelled", [False, True])
@pytest.mark.parametrize("projection", ["x", "x AS renamed", "DISTINCT x", "x.name", "count(DISTINCT x) AS c"])
def test_entity_projection_kind_and_identity_are_independent(engine, labelled, projection):
    nodes = pd.DataFrame({"id": [0, 1, 2, 3], "name": ["A", "B", "same", "same"]})
    if labelled:
        nodes = nodes.assign(label__Person=True)
    edges = pd.DataFrame({"s": [0, 0, 1, 1], "d": [2, 3, 2, 3]})
    g = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id").edges(df_to_engine(edges, Engine(engine)), "s", "d")
    query = "MATCH (a {name: 'A'}), (b {name: 'B'}) MATCH (a)-->(x)<-->(b) RETURN " + projection
    out = g.gfql(query, engine=engine)
    frame = df_to_engine(out._nodes, Engine.PANDAS)
    kinds = out._cypher_entity_projection_kinds
    if projection == "count(DISTINCT x) AS c":
        assert frame.to_dict("records") == [{"c": 2}]
        assert kinds is None
    elif projection == "x.name":
        assert frame.to_dict("records") == [{"x.name": "same"}, {"x.name": "same"}]
        assert kinds is None
    else:
        alias = "renamed" if projection == "x AS renamed" else "x"
        assert kinds == {alias: "nodes"}
        assert len(frame) == 2
        assert frame[f"{alias}.name"].tolist() == ["same", "same"]
        if projection == "DISTINCT x":
            assert sorted(frame[f"{alias}.id"].tolist()) == [2, 3]
    assert g._cypher_entity_projection_kinds is None


def test_null_entity_presence_survives_missing_identity_and_renaming(engine):
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionColumn, ResultProjectionPlan
    from graphistry.compute.gfql.cypher.result_postprocess import apply_result_projection, render_entity_text

    nodes = pd.DataFrame({"x": pd.Series([True, None], dtype="boolean"), "x.name": pd.Series([None, None], dtype="string")})
    g = graphistry.nodes(df_to_engine(nodes, Engine(engine)), "id")
    plan = ResultProjectionPlan(
        alias="x", table="nodes",
        columns=(ResultProjectionColumn("renamed", "whole_row"),),
    )
    out = apply_result_projection(g, plan)
    assert out._cypher_entity_projection_meta == {}
    assert g._cypher_entity_projection_presence == {}
    rendered_graph = out.bind()
    if engine.startswith("polars"):
        rendered_graph._nodes = out._nodes.to_pandas()
    text = render_entity_text(rendered_graph, "renamed")
    if engine == "cudf":
        text = text.to_pandas()
    assert len(text) == 2
    assert text.iloc[0] == "()"
    assert pd.isna(text.iloc[1])
    assert out._cypher_entity_projection_kinds == {"renamed": "nodes"}


@pytest.mark.parametrize("empty", [False, True])
def test_presence_alignment_preserves_reordered_entities_and_inserted_nulls(engine, empty):
    from graphistry.compute.gfql.cypher.result_postprocess import entity_projection_presence_for_rows

    marker = df_to_engine(pd.DataFrame({"x": pd.Series([True, None, True], dtype="boolean")}), Engine(engine))
    if empty:
        marker = marker.head(0)
    g = graphistry.nodes(marker, "id")
    g._cypher_entity_projection_presence = {"renamed": marker}
    indices = [None, None] if empty else [2, None, None, 0, 1]
    aligned = entity_projection_presence_for_rows(g, indices)["renamed"]
    aligned = df_to_engine(aligned, Engine.PANDAS)
    assert aligned["x"].notna().tolist() == ([False, False] if empty else [True, False, False, True, False])
    assert len(g._cypher_entity_projection_presence["renamed"]) == (0 if empty else 3)


def test_nullable_same_named_property_does_not_override_entity_marker(engine):
    from graphistry.compute.gfql.cypher.lowering import ResultProjectionColumn, ResultProjectionPlan
    from graphistry.compute.gfql.cypher.result_postprocess import apply_result_projection, render_entity_text

    frame = pd.DataFrame({"x": [True], "x.x": [None], "x.id": [1], "x.name": ["A"]})
    g = graphistry.nodes(df_to_engine(frame, Engine(engine)), "id")
    plan = ResultProjectionPlan(
        alias="x", table="nodes",
        columns=(ResultProjectionColumn("renamed", "whole_row"),),
    )
    out = apply_result_projection(g, plan)
    assert list(out._cypher_entity_projection_presence["renamed"].columns) == ["x"]
    rendered_graph = out.bind()
    if engine.startswith("polars"):
        rendered_graph._nodes = out._nodes.to_pandas()
    text = render_entity_text(rendered_graph, "renamed")
    if engine == "cudf":
        text = text.to_pandas()
    assert text.notna().tolist() == [True]
    ids = df_to_engine(out._nodes, Engine.PANDAS)
    assert ids["renamed.id"].tolist() == [1]
