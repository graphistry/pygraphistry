from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

import graphistry
from graphistry.compute.gfql.cypher.lowering import ResultProjectionColumn, ResultProjectionPlan
from graphistry.compute.gfql.cypher.result_postprocess import apply_result_projection

from graphistry.tests.compute.gfql.cypher._whole_entity_compat import entity_text_records


def _project_alice() -> Any:
    rows = pd.DataFrame(
        {
            "id": ["a"],
            "n": ["a"],
            "n.id": ["a"],
            "n.name": ["Alice"],
            "n.label__Person": [True],
        }
    )
    result = graphistry.bind(node="id").nodes(rows)
    return apply_result_projection(
        result,
        ResultProjectionPlan(
            alias="n",
            table="nodes",
            columns=(
                ResultProjectionColumn("node", "whole_row"),
                ResultProjectionColumn("name", "property", "name"),
            ),
        ),
    )


def test_apply_result_projection_emits_flat_whole_row_columns() -> None:
    # #1650: whole-entity projection is structured (one column per field), not a
    # Cypher display string. The scalar `name` projection passes through alongside.
    out = _project_alice()
    assert out._nodes.to_dict(orient="records") == [
        {"node.id": "a", "node.name": "Alice", "node.label__Person": True, "name": "Alice"}
    ]


def test_apply_result_projection_renders_whole_row_text_via_helper() -> None:
    # The structured columns losslessly reconstruct the pre-#1650 Cypher text form.
    out = _project_alice()
    assert entity_text_records(out, {"node": "nodes"}) == [
        {"node": "(:Person {name: 'Alice'})", "name": "Alice"}
    ]


def test_apply_result_projection_preserves_prefixed_whole_row_metadata() -> None:
    out = _project_alice()
    assert out._cypher_entity_projection_meta["node"]["table"] == "nodes"
    assert out._cypher_entity_projection_meta["node"]["alias"] == "n"
    assert out._cypher_entity_projection_meta["node"]["id_column"] == "id"
    assert out._cypher_entity_projection_meta["node"]["ids"].tolist() == ["a"]


def test_project_property_column_numeric_passes_through_unchanged() -> None:
    # Numeric/bool columns must pass through with dtype + values intact — the
    # temporal-constructor scan can never match them, so the gate skips the
    # spurious astype(str) and returns the column as-is (byte-identical).
    from graphistry.compute.gfql.cypher.result_postprocess import _project_property_column

    df = pd.DataFrame(
        {
            "x": pd.Series([3, 1, 2], dtype="int64"),
            "f": pd.Series([2.0, 1.5, 3.0], dtype="float64"),
            "b": pd.Series([True, False, True]),
        }
    )
    for col, dtype in [("x", "int64"), ("f", "float64"), ("b", "bool")]:
        out = _project_property_column(df, column=ResultProjectionColumn(col, "property", col))
        assert str(out.dtype) == dtype, f"{col}: dtype changed to {out.dtype}"
        assert out.tolist() == df[col].tolist()


def test_numeric_return_order_by_is_numeric_not_lexical() -> None:
    # Behavioral guard: a numeric ORDER BY must sort numerically. If the projected
    # column were stringified, a lexical sort of [2,10,1] would give [1,10,2].
    nd = pd.DataFrame({"id": [0, 1, 2, 3], "val": [2, 10, 1, 30]})
    ed = pd.DataFrame({"s": [0, 1, 2], "d": [1, 2, 3]})
    g = graphistry.nodes(nd, "id").edges(ed, "s", "d")
    out = g.gfql("MATCH (a) RETURN a.val ORDER BY a.val", engine="pandas")._nodes
    vals = out[out.columns[0]].tolist()
    assert vals == [1, 2, 10, 30]


def test_whole_entity_projection_records_kind_without_identity_column() -> None:
    pl = pytest.importorskip("polars")

    frame = pl.DataFrame({"x": [True, True], "x.name": ["same", None]})
    g = graphistry.nodes(frame, "id")
    plan = ResultProjectionPlan(
        alias="x", table="nodes",
        columns=(ResultProjectionColumn("renamed", "whole_row"),),
    )
    out = apply_result_projection(g, plan)
    assert out._cypher_entity_projection_kinds == {"renamed": "nodes"}
    assert out._nodes.to_dicts() == [{"renamed.name": "same"}, {"renamed.name": None}]
    assert g._cypher_entity_projection_kinds is None
    assert out._cypher_entity_projection_meta == {}


def test_property_projection_clears_whole_entity_provenance() -> None:
    g = graphistry.nodes(pd.DataFrame({"x": [True], "name": ["Alice"]}), "id")
    g._cypher_entity_projection_kinds = {"x": "nodes"}
    g._cypher_entity_projection_presence = {"x": pd.DataFrame({"x": [True]})}
    plan = ResultProjectionPlan(
        alias="x", table="nodes",
        columns=(ResultProjectionColumn("x.name", "property", "name"),),
    )
    out = apply_result_projection(g, plan)
    assert out._cypher_entity_projection_kinds == {}
    assert out._cypher_entity_projection_presence == {}
    assert g._cypher_entity_projection_presence["x"].to_dict("records") == [{"x": True}]
    assert g._cypher_entity_projection_kinds == {"x": "nodes"}
    assert out._nodes.to_dict("records") == [{"x.name": "Alice"}]


def test_render_entity_text_polars_matches_pandas_issue_2314() -> None:
    # #2314: the presentation helper called pandas-only rename(columns=...) on polars results.
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text

    expected = render_entity_text(
        graphistry.nodes(pd.DataFrame({"id": ["old"], "type": ["Old"]}), "id").gfql(
            "MATCH (n) RETURN n", engine="pandas"
        ),
        "n",
        table="nodes",
    ).tolist()
    assert expected == ["(:Old)"]

    g = graphistry.nodes(pl.DataFrame({"id": ["old"], "type": ["Old"]}), "id")
    r = g.gfql("MATCH (n) RETURN n", engine="polars")
    nodes_before = r._nodes.clone()
    presence_before = r._cypher_entity_projection_presence["n"].clone()
    out = render_entity_text(r, "n", table="nodes")
    assert isinstance(out, pl.Series)
    assert out.to_list() == expected
    assert r._nodes.equals(nodes_before)
    assert r._cypher_entity_projection_presence["n"].equals(presence_before)


_ENTITY_TEXT_NODE_DATA = {
    "n.id": ["a", "b", "c", None],
    "n.label__A": [True, False, True, None],
    "n.name": ["it's", None, "back\\slash", None],
    "n.age": [1, None, 3, None],
    "n.flag": [True, False, None, None],
}
_ENTITY_TEXT_NODE_EXPECTED = [
    "(:A {name: 'it\\'s', age: 1, flag: true})",
    "({flag: false})",
    "(:A {name: 'back\\\\slash', age: 3})",
    None,
]


def _entity_text_frame(engine: str, data: dict) -> Any:
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return pl.DataFrame(data)
    frame = pd.DataFrame(data)
    for col, values in data.items():
        if any(isinstance(v, int) and not isinstance(v, bool) for v in values):
            frame[col] = pd.array(values, dtype="Int64")
    return frame


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("with_presence", [True, False])
def test_render_entity_text_nodes_engine_parity(engine: str, with_presence: bool) -> None:
    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text

    rows = _entity_text_frame(engine, _ENTITY_TEXT_NODE_DATA)
    g = graphistry.nodes(rows)
    if with_presence:
        g._cypher_entity_projection_presence = {"n": _entity_text_frame(engine, {"n": ["a", "b", "c", None]})}
    out = render_entity_text(g, "n", table="nodes")
    assert list(out.to_list() if engine == "polars" else out.where(out.notna(), None).tolist()) == (
        _ENTITY_TEXT_NODE_EXPECTED
    )


@pytest.mark.parametrize("engine", ["pandas", "polars"])
def test_render_entity_text_edges_engine_parity(engine: str) -> None:
    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text

    rows = _entity_text_frame(engine, {"r.type": ["KNOWS", " ", None], "r.since": [2020, None, None]})
    g = graphistry.nodes(rows)
    g._cypher_entity_projection_presence = {"r": _entity_text_frame(engine, {"r": [1, 2, None]})}
    out = render_entity_text(g, "r", table="edges")
    assert list(out.to_list() if engine == "polars" else out.where(out.notna(), None).tolist()) == [
        "[:KNOWS {since: 2020}]",
        "[]",
        None,
    ]


def test_render_entity_text_polars_unsupported_dtype_declines_explicitly() -> None:
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text

    g = graphistry.nodes(pl.DataFrame({"n.id": ["a"], "n.score": [1.5]}))
    with pytest.raises(NotImplementedError, match="render_entity_text"):
        render_entity_text(g, "n", table="nodes")


def test_render_entity_text_polars_misaligned_presence_raises() -> None:
    pl = pytest.importorskip("polars")
    from graphistry.compute.gfql.cypher.result_postprocess import render_entity_text

    g = graphistry.nodes(pl.DataFrame({"n.id": ["a"], "n.name": ["x"]}))
    g._cypher_entity_projection_presence = {"n": pl.DataFrame({"n": ["a", "b"]})}
    with pytest.raises(ValueError, match="not aligned"):
        render_entity_text(g, "n", table="nodes")


def test_empty_projection_keeps_whole_entity_provenance() -> None:
    g = graphistry.nodes(pd.DataFrame({"x": pd.Series([], dtype="bool"), "name": pd.Series([], dtype="str")}), "id")
    plan = ResultProjectionPlan(
        alias="x", table="nodes",
        columns=(ResultProjectionColumn("x", "whole_row"),),
    )
    out = apply_result_projection(g, plan)
    assert out._cypher_entity_projection_kinds == {"x": "nodes"}
    assert len(out._nodes) == 0
