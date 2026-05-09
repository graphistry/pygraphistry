import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.dataframe.join import (
    binding_join_columns,
    connected_inner_join_rows,
    joined_alias_columns,
    joined_hidden_scalar_columns,
)


def _records(df):
    if hasattr(df, "to_pandas"):
        return df.to_pandas().to_dict(orient="records")
    return df.to_dict(orient="records")


def test_binding_join_columns_selects_dotted_columns_only() -> None:
    frame = pd.DataFrame(
        {
            "a.id": ["n1"],
            "a.label": ["A"],
            "plain": [1],
            5: [9],
        }
    )
    assert binding_join_columns(frame) == ["a.id", "a.label"]


def test_joined_hidden_scalar_columns_coalesces_suffix_variants() -> None:
    frame = pd.DataFrame(
        {
            "a.__cypher_reentry_property__": [None, 2],
            "b.__cypher_reentry_property__": [1, None],
            "a.id": ["a1", "a2"],
        }
    )
    out = joined_hidden_scalar_columns(frame)
    assert _records(out[["__cypher_reentry_property__"]]) == [
        {"__cypher_reentry_property__": 1.0},
        {"__cypher_reentry_property__": 2.0},
    ]


def test_joined_alias_columns_recovers_alias_identity_columns() -> None:
    frame = pd.DataFrame(
        {
            "a.id": ["a1"],
            "b.b": ["b1"],
            "b.id": ["b1-id-fallback"],
        }
    )
    out = joined_alias_columns(frame)
    assert _records(out[["a", "b"]]) == [{"a": "a1", "b": "b1"}]


def test_connected_inner_join_rows_pandas_path() -> None:
    joined_rows = pd.DataFrame(
        {
            "a.id": ["a1", "a2"],
            "a.num": [1, 2],
        }
    )
    pattern_rows = pd.DataFrame(
        {
            "a.id": ["a1", "a1", "a3"],
            "b.id": ["b1", "b2", "b3"],
        }
    )
    out = connected_inner_join_rows(
        joined_rows,
        pattern_rows,
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.PANDAS,
    )
    assert _records(out[["a.id", "b.id"]]) == [
        {"a.id": "a1", "b.id": "b1"},
        {"a.id": "a1", "b.id": "b2"},
    ]


def test_connected_inner_join_rows_cudf_path() -> None:
    cudf = pytest.importorskip("cudf")
    joined_rows = cudf.DataFrame(
        {
            "a.id": ["a1", "a2"],
            "a.num": [1, 2],
        }
    )
    pattern_rows = cudf.DataFrame(
        {
            "a.id": ["a1", "a1", "a3"],
            "b.id": ["b1", "b2", "b3"],
        }
    )
    out = connected_inner_join_rows(
        joined_rows,
        pattern_rows,
        join_cols=["a.id"],
        keep_cols=["a.id", "b.id"],
        engine=Engine.CUDF,
    )
    assert type(out).__module__.startswith("cudf")
    assert _records(out[["a.id", "b.id"]]) == [
        {"a.id": "a1", "b.id": "b1"},
        {"a.id": "a1", "b.id": "b2"},
    ]
