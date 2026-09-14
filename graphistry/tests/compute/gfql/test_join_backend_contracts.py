"""Join helper contracts at backend, multiplicity, and empty-input boundaries."""
import os

import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.dataframe.join import semijoin_by_column


@pytest.mark.parametrize("backend", ["pandas", "polars", "polars-lazy", "cudf"])
@pytest.mark.parametrize("key_values", [[], [2], [2, 2], [2, 3]])
def test_semijoin_keeps_left_duplicates_without_multiplying_by_right(backend, key_values):
    left = pd.DataFrame({"id": [2, 1, 2, 3], "payload": [20, 10, 21, 30]})
    right = pd.DataFrame({"key": pd.Series(key_values, dtype="int64")})
    if backend == "cudf":
        if not os.environ.get("TEST_CUDF"):
            pytest.skip("TEST_CUDF is required for GPU tests")
        cudf = pytest.importorskip("cudf")
        frame, keys = cudf.from_pandas(left), cudf.from_pandas(right)
        engine = Engine.CUDF
    elif backend.startswith("polars"):
        pl = pytest.importorskip("polars")
        frame, keys = pl.from_pandas(left), pl.from_pandas(right)
        if backend == "polars-lazy":
            frame, keys = frame.lazy(), keys.lazy()
        engine = Engine.POLARS
    else:
        frame, keys, engine = left.copy(), right.copy(), Engine.PANDAS
    result = semijoin_by_column(frame, keys, left_on="id", right_on="key", engine=engine)
    if backend == "polars-lazy":
        assert isinstance(result, pl.LazyFrame)
        result = result.collect()
    if backend.startswith("polars"):
        records = result.to_dicts()
    else:
        records = (result.to_pandas() if backend == "cudf" else result).to_dict("records")
    assert sorted(records, key=lambda row: row["payload"]) == sorted(
        left[left.id.isin(key_values)].to_dict("records"), key=lambda row: row["payload"]
    )
    assert list(result.columns) == ["id", "payload"]


@pytest.mark.parametrize("lazy_left", [False, True])
def test_polars_semijoin_rejects_mixed_frame_flavors(lazy_left):
    pl = pytest.importorskip("polars")
    frame, keys = pl.DataFrame({"id": [1]}), pl.DataFrame({"key": [1]})
    if lazy_left:
        frame = frame.lazy()
    else:
        keys = keys.lazy()
    with pytest.raises(AssertionError):
        semijoin_by_column(frame, keys, left_on="id", right_on="key", engine=Engine.POLARS)
