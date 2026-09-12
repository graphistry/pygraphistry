"""group_by aggregation sources that name an existing column validate as that column.

Both engines read an aggregation source that is an existing column as the column itself: pandas
checks the column before parsing an expression, and polars accepts only columns. The validator
parsed every source as an expression, splitting ``analysis.cohorts.Activist`` at the dot and
``x.Pro-Vax`` at the minus, so uploaded column names the executors would serve were rejected as
missing columns ``analysis`` and ``Vax``.
"""
from __future__ import annotations

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import group_by, rows
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError

try:
    import polars  # noqa: F401
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

ENGINES = [
    "pandas",
    pytest.param("polars", marks=pytest.mark.skipif(not HAS_POLARS, reason="polars not installed")),
]

COLUMNS = ["plain", "x.y.Dotted", "x.y.Pro-Vax", "x.user-y.Left Wing"]


def _graph():
    values = [0, 1, 2, 0, 3, 0]
    nodes = pd.DataFrame({
        "id": list(range(6)),
        "kind": ["a", "a", "b", "b", "a", "b"],
        **{c: values for c in COLUMNS},
    })
    return graphistry.nodes(nodes, "id")


def _max_by_kind(source):
    return [rows(), group_by(keys=["kind"], aggregations=[["m", "max", source]])]


def _validates_clean(g, query) -> bool:
    try:
        out = g.gfql_validate(query)
    except GFQLValidationError:
        return False
    return out["ok"] is True and out["diagnostics"] == []


def _max_per_kind(g, query, engine):
    df = g.gfql(query, engine=engine)._nodes
    pdf = df.to_pandas() if hasattr(df, "to_pandas") else df
    return sorted((r["kind"], r["m"]) for r in pdf.to_dict("records"))


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("column", COLUMNS)
def test_existing_column_validates_and_executes_as_that_column(column, engine):
    g = _graph()
    query = _max_by_kind(column)

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, engine) == [("a", 3), ("b", 2)]


def test_missing_column_is_still_diagnosed():
    g = _graph()
    query = _max_by_kind("x.y.Missing")

    assert not _validates_clean(g, query)
    with pytest.raises(GFQLValidationError) as exc:
        g.gfql(query, engine="pandas")
    assert exc.value.code == ErrorCode.E301


def test_expression_over_existing_columns_still_validates():
    g = _graph()
    query = _max_by_kind("plain * 2")

    assert _validates_clean(g, query)
    assert _max_per_kind(g, query, "pandas") == [("a", 6), ("b", 4)]
