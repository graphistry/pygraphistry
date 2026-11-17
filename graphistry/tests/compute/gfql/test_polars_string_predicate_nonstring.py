"""Regression: a string predicate on a non-String column preserves Cypher unknown.

Categorical and numeric columns must not be stringified or raise a type error; every row
is excluded from WHERE because its predicate result is null.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLSchemaError

try:
    import polars  # noqa: F401
    _HAS_POLARS = True
except Exception:  # pragma: no cover
    _HAS_POLARS = False

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover
    _HAS_CUDF = False

_ENGINES = ["pandas"] + (["polars"] if _HAS_POLARS else []) + (
    ["cudf"] if _HAS_CUDF and __import__("os").environ.get("TEST_CUDF") == "1" else []
)

_N = pd.DataFrame({
    "id": [0, 1, 2, 3, 4],
    "cat": pd.Categorical(["Alice", "Bob", "Amy", None, "Al"]),
    "age": [30, 25, 40, 22, None],
})
_E = pd.DataFrame({"s": [0, 1, 2, 3], "d": [1, 2, 3, 4], "eid": [0, 1, 2, 3]})


def _g():
    return graphistry.nodes(_N, "id").edges(_E, "s", "d").bind(edge="eid")


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query", [
    "MATCH (m) WHERE m.age CONTAINS '2' RETURN m.id AS id",          # numeric
])
def test_string_predicate_on_nonstring_column_returns_unknown_all_engines(engine, query):
    nodes = _g().gfql(query, engine=engine)._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    assert nodes.empty


@pytest.mark.parametrize("engine", _ENGINES)
@pytest.mark.parametrize("query,expected", [
    ("MATCH (m) WHERE m.cat STARTS WITH 'A' RETURN m.id AS id", [0, 2, 4]),
    ("MATCH (m) WHERE m.cat CONTAINS 'm' RETURN m.id AS id", [2]),
])
def test_string_valued_categorical_predicates_still_match(engine, query, expected):
    nodes = _g().gfql(query, engine=engine)._nodes
    if hasattr(nodes, "to_pandas"):
        nodes = nodes.to_pandas()
    assert nodes["id"].tolist() == expected


@pytest.mark.parametrize("engine", _ENGINES)
def test_equality_on_categorical_still_works(engine):
    # `=` on a categorical is fine on all engines (not a `.str` op).
    r = _g().gfql("MATCH (m) WHERE m.cat = 'Bob' RETURN m.id AS id", engine=engine)._nodes
    if hasattr(r, "to_pandas"):
        r = r.to_pandas()
    assert r.to_dict("records") == [{"id": 1}]
