from __future__ import annotations

import pandas as pd

import graphistry
from graphistry.compute.gfql.cypher.lowering import ResultProjectionColumn, ResultProjectionPlan
from graphistry.compute.gfql.cypher.result_postprocess import apply_result_projection


def test_apply_result_projection_preserves_prefixed_whole_row_metadata() -> None:
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

    out = apply_result_projection(
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

    assert out._nodes.to_dict(orient="records") == [{"node": "(:Person {name: 'Alice'})", "name": "Alice"}]
    assert out._cypher_entity_projection_meta["node"]["table"] == "nodes"
    assert out._cypher_entity_projection_meta["node"]["alias"] == "n"
    assert out._cypher_entity_projection_meta["node"]["id_column"] == "id"
    assert out._cypher_entity_projection_meta["node"]["ids"].tolist() == ["a"]
