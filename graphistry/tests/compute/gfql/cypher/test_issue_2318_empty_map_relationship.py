from typing import Any, Dict, List

import pandas as pd
import pytest

import graphistry


def _engine_data(engine: str) -> tuple[Any, Any, Any]:
    nodes = pd.DataFrame({"id": [0, 1]})
    edges = pd.DataFrame(
        {
            "s": [0, 1],
            "d": [1, 0],
            "type": ["R", "S"],
            "name": ["left", "right"],
        }
    )
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(nodes), cudf.from_pandas(edges), cudf
    if engine == "polars":
        polars = pytest.importorskip("polars")
        return polars.from_pandas(nodes), polars.from_pandas(edges), polars
    return nodes, edges, pd


def _records(frame: Any, engine: str) -> List[Dict[str, Any]]:
    if engine == "cudf":
        return frame.to_arrow().to_pylist()
    if engine == "polars":
        return frame.to_dicts()
    return frame.to_dict("records")


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize(
    ("pattern", "expected"),
    [
        ("r:R", ["left"]),
        ("r:R {}", ["left"]),
        ("r {}", ["left", "right"]),
    ],
)
def test_empty_relationship_property_map_preserves_types(
    engine: str, pattern: str, expected: List[str]
) -> None:
    nodes, edges, _ = _engine_data(engine)
    graph = graphistry.nodes(nodes, "id").edges(edges, "s", "d")
    result = graph.gfql(
        f"MATCH ()-[{pattern}]->() RETURN r.name AS name", engine=engine
    )

    assert sorted(row["name"] for row in _records(result._nodes, engine)) == expected
    assert result._nodes.__class__.__module__.split(".")[0] == engine
    assert graph._edges is edges
