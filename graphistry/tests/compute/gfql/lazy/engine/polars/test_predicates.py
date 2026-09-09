"""Singleton filters preserve expression filtering, including errors and fallback."""
import pytest

pl = pytest.importorskip("polars")
from polars.testing import assert_frame_equal
from graphistry.Engine import Engine
from graphistry.compute.chain_fast_paths import _verify_scalar_filters_on_hit
from graphistry.compute.gfql.lazy import ExecutionTarget, target_mode
from graphistry.compute.gfql.lazy.engine.polars.predicates import (
    _filter_singleton_equalities, filter_by_dict_polars, filter_expr_by_dict_polars,
)


def oracle(frame, filters):
    expr = filter_expr_by_dict_polars(frame, filters)
    return frame if expr is None else frame.filter(expr)


@pytest.mark.parametrize("dtype,bits,signed", [
    (pl.Int8, 8, True), (pl.Int16, 16, True), (pl.Int32, 32, True), (pl.Int64, 64, True),
    (pl.UInt8, 8, False), (pl.UInt16, 16, False), (pl.UInt32, 32, False), (pl.UInt64, 64, False),
])
def test_integer_boundaries(dtype, bits, signed):
    low, high = (-(2 ** (bits - 1)), 2 ** (bits - 1) - 1) if signed else (0, 2**bits - 1)
    for actual in (None, low, high, 0, 1):
        frame = pl.DataFrame({"x": pl.Series([actual], dtype=dtype), "keep": ["payload"]})
        original = frame.clone()
        for expected in (low - 1, low, high, high + 1, -1, 0, 1, 2**53 + 1, 2**63 - 1):
            filters = {"x": expected}
            assert_frame_equal(filter_by_dict_polars(frame, filters), oracle(frame, filters))
            verified = _verify_scalar_filters_on_hit(frame, filters, Engine.POLARS)
            if verified is not None:
                assert_frame_equal(verified, oracle(frame, filters))
        assert_frame_equal(frame, original)


@pytest.mark.parametrize("dtype,value,expected", [
    (pl.Boolean, True, True), (pl.Boolean, False, True), (pl.Boolean, None, False),
    (pl.String, "", ""), (pl.String, "雪", "雪"), (pl.String, None, "x"),
    (pl.String, "date('2020-01-01')", "date('2020-01-01')"),
])
def test_supported_scalar_parity(dtype, value, expected):
    frame = pl.DataFrame({"x": pl.Series([value], dtype=dtype)})
    result = _filter_singleton_equalities(frame, {"x": expected})
    assert result is not None
    assert_frame_equal(result, oracle(frame, {"x": expected}))


@pytest.mark.parametrize("value", [1.0, float("nan"), None, [1], True, "1"])
def test_unsupported_values_and_errors(value):
    frame = pl.DataFrame({"first": [False], "x": [1]})
    filters = {"first": True, "x": value}
    assert _filter_singleton_equalities(frame, filters) is None
    try:
        expected = oracle(frame, filters)
    except Exception as error:
        with pytest.raises(type(error)) as caught:
            filter_by_dict_polars(frame, filters)
        assert str(caught.value) == str(error)
    else:
        assert_frame_equal(filter_by_dict_polars(frame, filters), expected)


@pytest.mark.parametrize("frame,filters", [
    (pl.DataFrame({"x": []}, schema={"x": pl.Int64}), {"x": 1}),
    (pl.DataFrame({"x": [1, 1]}), {"x": 1}),
    (pl.DataFrame({"x": [1]}).lazy(), {"x": 1}),
    (pl.DataFrame({"x": [1]}), {}),
    (pl.DataFrame({"x": [None]}), {"x": 1}),
    (pl.DataFrame({"type": ["Person"]}), {"label__Person": True}),
    (pl.DataFrame({"labels": [["Person"]]}), {"labels": "Person"}),
    (pl.DataFrame({"label__Person": [True]}), {"type": "Person"}),
    (pl.DataFrame({"x": [1.0]}), {"x": 1}),
])
def test_fallback_parity(frame, filters):
    assert _filter_singleton_equalities(frame, filters) is None
    actual, expected = filter_by_dict_polars(frame, filters), oracle(frame, filters)
    if isinstance(actual, pl.LazyFrame):
        actual, expected = actual.collect(), expected.collect()
    assert_frame_equal(actual, expected)


def test_missing_column_error_after_mismatch():
    frame = pl.DataFrame({"x": [1]})
    filters = {"x": 2, "missing": 1}
    assert _filter_singleton_equalities(frame, filters) is None
    with pytest.raises(Exception) as expected:
        oracle(frame, filters)
    with pytest.raises(type(expected.value)) as actual:
        filter_by_dict_polars(frame, filters)
    assert str(actual.value) == str(expected.value)


def test_gpu_target_declines_scalar_read(monkeypatch):
    frame = pl.DataFrame({"x": [1]})
    def forbidden(*args, **kwargs):
        raise AssertionError("GPU target must not extract a Python scalar")
    monkeypatch.setattr(pl.Series, "item", forbidden)
    with target_mode(ExecutionTarget.GPU):
        assert _filter_singleton_equalities(frame, {"x": 1}) is None
        assert _verify_scalar_filters_on_hit(frame, {"x": 1}, Engine.POLARS) is None
        assert_frame_equal(filter_by_dict_polars(frame, {"x": 1}), frame)
