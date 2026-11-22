from __future__ import annotations

import os
from typing import Any

import pandas as pd
import pytest

from graphistry.compute.gfql.cypher.reentry.execution import _empty_frame


def _frame(engine: str) -> Any:
    if engine == "pandas":
        return pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"), "flag": [True, False]})
    if engine in {"polars", "polars-gpu"}:
        pl = pytest.importorskip("polars")
        return pl.DataFrame({"id": [1, 2], "flag": [True, False]}, schema={"id": pl.Int64, "flag": pl.Boolean})
    if engine == "polars-lazy":
        pl = pytest.importorskip("polars")
        return pl.DataFrame({"id": [1, 2], "flag": [True, False]}, schema={"id": pl.Int64, "flag": pl.Boolean}).lazy()
    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        return cudf.from_pandas(pd.DataFrame({"id": pd.Series([1, 2], dtype="int64"), "flag": [True, False]}))
    raise AssertionError(engine)


def _dtypes(frame: Any) -> list[str]:
    if hasattr(frame, "collect_schema"):
        return [str(dtype) for dtype in frame.collect_schema().dtypes()]
    if hasattr(frame, "schema"):
        return [str(dtype) for dtype in frame.schema.dtypes()]
    return [str(dtype) for dtype in frame.dtypes]


@pytest.mark.parametrize(
    "engine",
    [
        "pandas",
        "polars",
        "polars-lazy",
        pytest.param("polars-gpu", marks=pytest.mark.skipif(os.environ.get("TEST_POLARS_GPU") != "1", reason="requires actual polars-gpu")),
        pytest.param("cudf", marks=pytest.mark.skipif(os.environ.get("TEST_CUDF") != "1", reason="cudf tests need TEST_CUDF=1")),
    ],
)
def test_reentry_empty_frame_is_engine_native(engine: str) -> None:
    frame = _frame(engine)
    empty = _empty_frame(frame)

    assert type(empty) is type(frame)
    if hasattr(empty, "collect"):
        assert empty.limit(1).collect().is_empty()
    else:
        assert len(empty) == 0
    assert list(empty.columns) == ["id", "flag"]
    assert _dtypes(empty) == _dtypes(frame)
