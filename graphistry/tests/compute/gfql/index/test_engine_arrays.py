"""Native index gathers preserve Polars values, schemas, and indexing errors."""
from decimal import Decimal
import numpy as np
import pytest

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal
from graphistry.Engine import Engine
from graphistry.compute.gfql.index.engine_arrays import take_rows


def frame():
    return pl.DataFrame([
        pl.Series("integer", [1, None, -(2**63)], dtype=pl.Int64),
        pl.Series("unsigned", [2**63 + 1, None, 2**64 - 1], dtype=pl.UInt64),
        pl.Series("float", [float("nan"), float("inf"), -0.0]),
        pl.Series("bool", [True, None, False], dtype=pl.Boolean),
        pl.Series("string", ["雪", None, ""], dtype=pl.String),
        pl.Series("binary", [b"x", None, b""], dtype=pl.Binary),
        pl.Series("datetime", [1, None, 10**18 + 123], dtype=pl.Int64).cast(pl.Datetime("ns")),
        pl.Series("duration", [1, None, 123], dtype=pl.Int64).cast(pl.Duration("ns")),
        pl.Series("date", [0, None, 20000], dtype=pl.Int64).cast(pl.Date),
        pl.Series("time", [1, None, 123], dtype=pl.Int64).cast(pl.Time),
        pl.Series("decimal", [Decimal("1.23"), None, Decimal("-4.56")], dtype=pl.Decimal(10, 2)),
        pl.Series("list", [[1, None], None, []], dtype=pl.List(pl.Int64)),
        pl.Series("struct", [{"a": 1}, None, {"a": None}], dtype=pl.Struct({"a": pl.Int64})),
        pl.Series("category", ["red", None, "blue"], dtype=pl.Categorical),
        pl.Series("enum", ["red", None, "blue"], dtype=pl.Enum(["red", "blue"])),
    ])


POSITIONS = [
    np.array([0], dtype=np.uint32), np.array([1], dtype=np.uint64), np.array([2], dtype=np.int8),
    np.array([1], dtype=np.int16), np.array([1], dtype=np.int32), np.array([1], dtype=np.int64),
    np.array([-1]), np.array([-3]), np.array([-4]), np.array([3]),
    np.array([2, 0, 2]), np.array([], dtype=np.int64), np.array([], dtype=float),
    np.array([1.0]), np.array([True]), np.array([[0]]), np.array(1),
    np.array([2**64 - 1], dtype=np.uint64),
]


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.POLARS_GPU])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("positions", POSITIONS)
def test_take_rows_matches_native_array_indexing(engine, chunked, positions):
    data = frame()
    if chunked:
        data = pl.concat([data.head(1), data.tail(2)], rechunk=False)
    original = data.clone()
    try:
        expected = data[positions]
    except Exception as error:
        with pytest.raises(type(error)) as actual:
            take_rows(data, positions, engine)
        assert str(actual.value) == str(error)
    else:
        actual = take_rows(data, positions, engine)
        assert_frame_equal(actual, expected, check_exact=True)
    assert_frame_equal(data, original, check_exact=True)


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.POLARS_GPU])
def test_take_rows_empty_frame_keeps_bounds_error(engine):
    data = frame().clear()
    positions = np.array([0])
    with pytest.raises(Exception) as expected:
        data[positions]
    with pytest.raises(type(expected.value)) as actual:
        take_rows(data, positions, engine)
    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.POLARS_GPU])
def test_singleton_result_can_replace_column_without_mutating_source(engine):
    data = frame()
    original = data.clone()
    result = take_rows(data, np.array([0]), engine)
    result.replace_column(0, pl.Series("integer", [999], dtype=pl.Int64))
    assert result["integer"].item() == 999
    assert_frame_equal(data, original, check_exact=True)


@pytest.mark.parametrize("engine", [Engine.POLARS, Engine.POLARS_GPU])
@pytest.mark.parametrize("width", [15, 32, 33, 64])
@pytest.mark.parametrize("chunked", [False, True])
@pytest.mark.parametrize("positions", [
    np.array([], dtype=np.int64), np.array([0]), np.array([0, 1]), np.array([1, 2]), np.array([2, 0]), np.array([1, 1]),
    np.array([2, 0, 1, 2, 0, 1, 2, 0]), np.array([0] * 9),
    np.array([0, 3]), np.array([0, -1]), np.array([0, -4]),
    np.array([0, 2**64 - 1], dtype=np.uint64),
    np.array([0, 1], dtype=np.uint8), np.array([0, 1], dtype=np.uint64),
    np.array([0.0, 1.0]), np.array([True, False]), np.array([[0, 1]]),
])
def test_small_gather_matches_native_values_errors_and_isolation(engine, width, chunked, positions):
    data = frame()
    for column in range(data.width, width):
        data.insert_column(data.width, data["integer"].alias(f"extra_{column}"))
    if chunked:
        data = pl.concat([data.head(1), data.tail(2)], rechunk=False)
    original = data.clone()
    try:
        expected = data[positions]
    except Exception as error:
        with pytest.raises(type(error)) as actual_error:
            take_rows(data, positions, engine)
        assert str(actual_error.value) == str(error)
    else:
        result = take_rows(data, positions, engine)
        assert_frame_equal(result, expected, check_exact=True)
        result.replace_column(0, pl.Series("integer", [999] * result.height, dtype=pl.Int64))
        assert_frame_equal(data, original, check_exact=True)
    assert_frame_equal(data, original, check_exact=True)
