from __future__ import annotations

from typing import Any

import pandas as pd

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
