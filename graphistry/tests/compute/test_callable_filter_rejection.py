"""Python callables in matcher dicts are rejected with a structured error (#2169, #967).

The quick reference documents declarative predicates such as ``gt(30)``; restoring local
callables is tracked separately in #967. Until then a callable must fail validation with
``E201`` and a suggestion, on every engine, before any traversal runs.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, n
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

NODES = pd.DataFrame({"id": ["a", "b", "c"], "age": [30, 25, 40]})
EDGES = pd.DataFrame({"s": ["a", "b"], "d": ["b", "c"]})


def _graph(engine: str):
    if engine == "polars":
        pl = pytest.importorskip("polars")
        return graphistry.nodes(pl.from_pandas(NODES), "id").edges(pl.from_pandas(EDGES), "s", "d")
    return graphistry.nodes(NODES, "id").edges(EDGES, "s", "d")


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize(
    "build, field",
    [
        (lambda: n({"age": lambda x: x > 30}), "filter_dict.age"),
        (lambda: e_forward(edge_match={"w": lambda x: x > 1}), "edge_match.w"),
        (lambda: e_forward(destination_node_match={"age": lambda x: x < 30}), "destination_node_match.age"),
    ],
    ids=["node_filter_dict", "edge_match", "destination_node_match"],
)
def test_callable_filter_values_are_rejected(engine: str, build, field: str) -> None:
    g = _graph(engine)
    with pytest.raises(GFQLTypeError) as exc_info:
        g.gfql([build()])
    err = exc_info.value
    assert err.code == ErrorCode.E201
    assert field in str(err)
    assert "predicates" in str(err)
