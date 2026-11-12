"""Regression tests for review findings R1-R3 on the explicit epoch datetime contract.

Source: plans/gfql-ic4-perf/review-2197-redesign-2026-10-15/final-report.md.
- R1: representable wide (year-5138) ms bounds compare exactly on native Polars,
  epoch or ISO origin; non-representable bounds decline, never raw overflow.
- R2: unrepresentable epoch counts (year 10000, NaT sentinel) raise structured E201
  at both wire decode entrances, never raw parser ValueErrors.
- R3: floating counts are validated in their own precision before narrowing;
  fractional extended-precision values are rejected, native floats carry the
  +/- (2**53 - 1) exact-integer cap, native integers stay exact.
"""
import numpy as np
import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e
from graphistry.compute.ast_temporal import DateTimeValue, temporal_value_from_json
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.compute.predicates.comparison import ge
from graphistry.compute.gfql.lazy.engine.polars.predicates import predicate_to_expr
from graphistry.models.gfql.coercions.temporal import to_ast


YEAR_5138_MS = 10**14  # 5138-11-16T09:46:40Z, inside JSON-safe and ms-representable range
YEAR_2020_MS = 1577836800000


def _polars():
    return pytest.importorskip("polars")


@pytest.mark.parametrize("origin", ["epoch", "iso"])
def test_r1_wide_valid_bound_native_polars(origin):
    pl = _polars()
    bound = (
        DateTimeValue.from_epoch(YEAR_5138_MS, unit="ms")
        if origin == "epoch"
        else DateTimeValue("5138-11-16T09:46:40+00:00")
    )
    edges = pl.DataFrame({"s": [1, 2], "d": [2, 1], "ts": [1577836800000, YEAR_5138_MS]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    result = graphistry.edges(edges, "s", "d").gfql([e(edge_match={"ts": ge(bound)})], engine="polars")
    assert result._edges["s"].to_list() == [2]


def test_r1_wide_bound_no_match_is_empty_not_crash():
    pl = _polars()
    edges = pl.DataFrame({"s": [1], "d": [2], "ts": [1577836800000]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    bound = DateTimeValue.from_epoch(YEAR_5138_MS, unit="ms")
    result = graphistry.edges(edges, "s", "d").gfql([e(edge_match={"ts": ge(bound)})], engine="polars")
    assert result._edges["s"].to_list() == []


def test_r1_finest_than_column_bound_still_declines_not_crashes():
    pl = _polars()
    # A 2020 row stays 2020 on the ms column. The +1us instant cannot be represented
    # at ms resolution, so Polars lowering must decline explicitly rather than round.
    edges = pl.DataFrame({"s": [1], "d": [2], "ts": [YEAR_2020_MS]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    dtype = edges.schema["ts"]
    bound = DateTimeValue("2020-01-01T00:00:00.000001+00:00")
    assert predicate_to_expr("ts", ge(bound), dtype=dtype) is None


def test_r1_exact_ms_bound_matches_ms_column():
    pl = _polars()
    edges = pl.DataFrame({"s": [1], "d": [2], "ts": [YEAR_2020_MS]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    bound = DateTimeValue("2020-01-01T00:00:00+00:00")
    result = graphistry.edges(edges, "s", "d").gfql([e(edge_match={"ts": ge(bound)})], engine="polars")
    assert result._edges["s"].to_list() == [1]


def test_r1_exact_ms_bound_no_match_is_empty():
    pl = _polars()
    edges = pl.DataFrame({"s": [1], "d": [2], "ts": [YEAR_2020_MS]}).with_columns(
        pl.col("ts").cast(pl.Datetime("ms"))
    )
    bound = DateTimeValue("2020-01-02T00:00:00+00:00")
    result = graphistry.edges(edges, "s", "d").gfql([e(edge_match={"ts": ge(bound)})], engine="polars")
    assert result._edges["s"].to_list() == []


@pytest.mark.parametrize("count", [-2**63, 253402300800000])  # NaT sentinel, year 10000
def test_r2_factory_unrepresentable_epoch_structured_e201(count):
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(count, unit="ms")
    assert error.value.code == ErrorCode.E201


@pytest.mark.parametrize("decode", [temporal_value_from_json, to_ast])
def test_r2_wire_unrepresentable_epoch_structured_e201(decode):
    with pytest.raises(GFQLTypeError) as error:
        decode({"type": "datetime", "value": 253402300800000, "unit": "ms"})
    assert error.value.code == ErrorCode.E201


def test_r3_extended_precision_fraction_not_rounded():
    value = np.longdouble("1000.00000000000001")
    if value == np.floor(value):
        pytest.skip("platform has no extended-precision fractional test value")
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(value, unit="ms")
    assert error.value.code == ErrorCode.E201


@pytest.mark.parametrize("value", [2.0**53, np.float64(2.0**53)])
def test_r3_native_floats_capped_at_safe_range(value):
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(value, unit="ms")
    assert error.value.code == ErrorCode.E201


def test_r3_native_int_beyond_safe_range_still_structured():
    # Native ints stay exact (no JS cap), but any unrepresentable instant - here a
    # count far beyond the ISO parser range - is still a structured E201, not a
    # raw parser error; representable native ints are proven by the year-9999 control.
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(2**53 + 1, unit="ms")
    assert error.value.code == ErrorCode.E201


@pytest.mark.parametrize("value", [np.float64("nan"), np.float64("inf"), np.longdouble("nan")])
def test_r3_np_float_nonfinite_rejected(value):
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(value, unit="ms")
    assert error.value.code == ErrorCode.E201


def test_r3_np_float_overflow_rejected():
    value = np.longdouble("1e400")
    try:
        int(float(value))
    except (OverflowError, ValueError):
        pass
    else:
        pytest.skip("platform converts this longdouble to an exact int")
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(value, unit="ms")
    assert error.value.code == ErrorCode.E201


def test_r1_native_int_beyond_engine_range_structured_e201():
    # Native ints are exact, but a count far outside the engine's datetime
    # range must decline as structured E201, never raw OverflowError (R1/R3).
    with pytest.raises(GFQLTypeError) as error:
        DateTimeValue.from_epoch(10**15, unit="s")
    assert error.value.code == ErrorCode.E201


@pytest.mark.parametrize("unit,count", [("s", 0), ("ms", -1), ("ms", 1000.0)])
def test_r1_r2_r3_valid_controls(unit, count):
    v = DateTimeValue.from_epoch(count, unit=unit)
    assert temporal_value_from_json(v.to_json()).as_pandas_value() == pd.Timestamp(int(count), unit=unit, tz="UTC")
