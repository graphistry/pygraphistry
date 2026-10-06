"""CPU index gathers preserve native values, index labels and source isolation."""
import numpy as np
import pandas as pd
import pytest

from graphistry.Engine import Engine
from graphistry.compute.gfql.index.engine_arrays import take_rows


@pytest.mark.parametrize("positions", [
    np.array([0], dtype=np.uint32), np.array([1]), np.array([2], dtype=np.int8),
    np.array([-1]), np.array([3]), np.array([2**64 - 1], dtype=np.uint64),
    np.array([2, 0, 2]), np.array([], dtype=np.int64), np.array([1.0]),
    np.array([True]), np.array([[0]]), np.array(1),
])
def test_native_pandas_gather_values_errors_and_cell_isolation(positions):
    frame = pd.DataFrame({
        "id": [0, 1, 2], "nullable": pd.array([1, None, -(2**63)], dtype="Int64"),
        "category": pd.Categorical(["red", None, "blue"]),
        "timestamp": pd.to_datetime([1, None, 10**18 + 123], unit="ns"),
    }, index=pd.Index([7, 7, 2], name="row_key"))
    original = frame.copy(deep=True)
    try:
        expected = frame.iloc[positions]
    except (IndexError, ValueError, TypeError) as error:
        with pytest.raises(type(error)) as actual:
            take_rows(frame, positions, Engine.PANDAS)
        assert str(actual.value) == str(error)
    else:
        result = take_rows(frame, positions, Engine.PANDAS)
        if isinstance(expected, pd.Series):
            pd.testing.assert_series_equal(result, expected)
        else:
            pd.testing.assert_frame_equal(result, expected)
            if len(result):
                result.iloc[0, 0] = 999
                pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("role,kind", [("nodes", "node_prop"), ("edges", "edge_prop")])
@pytest.mark.parametrize("pattern", ["all", "partial", "nulls", "none", "error"])
def test_public_indexed_filter_reuses_only_exact_matches_and_isolates_source(role, kind, pattern):
    import graphistry
    from graphistry.compute.exceptions import GFQLSchemaError
    from graphistry.compute.gfql.index import with_index_policy

    frame = pd.DataFrame({
        "id": [0, 1, 2, 3], "s": [0, 1, 2, 3], "d": [1, 2, 3, 4],
        "v": [1, 1, 2, 3], "keep": pd.array([1, 2 if pattern == "partial" else None, 2, 3], dtype="Int64"),
        "text": ["a", "b", "c", "d"],
    }, index=pd.Index([7, 7, 2, 1], name="row_key"))
    original = frame.copy(deep=True)
    base = graphistry.nodes(frame, "id").edges(frame, "s", "d")
    indexed = with_index_policy(base.create_index(kind, column="v", engine="pandas"), "force")
    filters = {"v": 1}
    if pattern in ("partial", "nulls", "none"):
        filters["keep"] = 1 if pattern != "none" else 999
    elif pattern == "error":
        filters["text"] = 123
    method = f"filter_{role}_by_dict"
    if pattern == "error":
        with pytest.raises(GFQLSchemaError) as expected:
            getattr(base, method)(filters, engine="pandas")
        with pytest.raises(GFQLSchemaError) as actual:
            getattr(indexed, method)(filters, engine="pandas")
        assert actual.value.code == expected.value.code
        assert actual.value.context == expected.value.context
    else:
        expected = getattr(getattr(base, method)(filters, engine="pandas"), f"_{role}")
        result = getattr(getattr(indexed, method)(filters, engine="pandas"), f"_{role}")
        pd.testing.assert_frame_equal(result, expected)
        if len(result):
            result.iloc[0, 0] = 999
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("storage", ["int64", "uint64", "float64", "category", "datetime64[us]"])
@pytest.mark.parametrize("value", [0, 1, 0.1, True, "1", 2**65])
def test_owned_scalar_candidate_residual_keeps_native_values_errors_and_isolation(storage, value):
    from graphistry.compute.filter_by_dict import _filter_property_candidates, filter_by_dict
    from graphistry.compute.exceptions import GFQLSchemaError

    values = (["0", "1", None] if storage == "category" else
              pd.to_datetime([0, 1, None], unit="s") if storage.startswith("datetime") else [0, 1, 2])
    frame = pd.DataFrame({"v": pd.Series(values, dtype=storage), "id": [0, 1, 2]})
    frame.index = pd.Index([7, 7, 2], name="row_key")
    original = frame.copy(deep=True)
    candidate = frame.iloc[[0, 1, 2]].copy(deep=True)
    predicate = {"v": value}
    try:
        expected = filter_by_dict(candidate, predicate, "pandas")
    except (GFQLSchemaError, ValueError, TypeError, OverflowError) as error:
        with pytest.raises(type(error)) as actual:
            _filter_property_candidates(frame, candidate, predicate, Engine.PANDAS)
        assert getattr(actual.value, "code", None) == getattr(error, "code", None)
        assert getattr(actual.value, "context", None) == getattr(error, "context", None)
    else:
        result = _filter_property_candidates(frame, candidate, predicate, Engine.PANDAS)
        pd.testing.assert_frame_equal(result, expected)
        if len(result):
            result.iloc[0, 1] = 999
    pd.testing.assert_frame_equal(frame, original)
