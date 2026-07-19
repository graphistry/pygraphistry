"""Regression: a string predicate on a non-String polars column must raise the SAME clean
GFQLSchemaError that pandas/cuDF raise — not an opaque polars `InvalidOperationError`
("expected String type, got: cat"). Categorical/Enum are treated as non-string here, exactly
as `filter_by_dict` does on pandas/cuDF, so the three engines stay consistent.
"""
import pandas as pd
import pytest

import graphistry
from graphistry.compute.exceptions import GFQLSchemaError

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except Exception:  # pragma: no cover
    _HAS_CUDF = False

_ENGINES = ["pandas", "polars"] + (["cudf"] if _HAS_CUDF else [])

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
    "MATCH (m) WHERE m.cat STARTS WITH 'A' RETURN m.id AS id",       # categorical
    "MATCH (m) WHERE m.cat CONTAINS 'm' RETURN m.id AS id",          # categorical
    "MATCH (m) WHERE m.age CONTAINS '2' RETURN m.id AS id",          # numeric
])
def test_string_predicate_on_nonstring_column_raises_schema_error_all_engines(engine, query):
    with pytest.raises(GFQLSchemaError):
        _g().gfql(query, engine=engine)._nodes


@pytest.mark.parametrize("engine", _ENGINES)
def test_equality_on_categorical_still_works(engine):
    # `=` on a categorical is fine on all engines (not a `.str` op).
    r = _g().gfql("MATCH (m) WHERE m.cat = 'Bob' RETURN m.id AS id", engine=engine)._nodes
    if hasattr(r, "to_pandas"):
        r = r.to_pandas()
    assert r.to_dict("records") == [{"id": 1}]
