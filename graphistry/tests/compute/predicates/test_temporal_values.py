import pytest
import pandas as pd
from datetime import datetime, date, time

import graphistry.compute.gfql.temporal.constructors as temporal_constructors
from graphistry.compute.ast_temporal import (
    DateTimeValue, DateValue, TimeValue, temporal_value_from_json
)
from graphistry.compute.predicates.comparison import gt, lt, le, ge, eq, ne
from graphistry.embed_utils import check_cudf


has_cudf, _ = check_cudf()

# Skip tests that require cuDF when it's not available
requires_cudf = pytest.mark.skipif(
    not has_cudf,
    reason="cudf not installed"
)


class TestDateTimeValue:
    def test_parse_iso8601_with_timezone(self):
        dt = DateTimeValue("2024-01-01T12:00:00+00:00", "UTC")
        assert dt.value == "2024-01-01T12:00:00+00:00"
        assert dt.timezone == "UTC"
        assert isinstance(dt.as_pandas_value(), pd.Timestamp)
        assert dt.as_pandas_value().hour == 12
    
    def test_parse_iso8601_naive(self):
        dt = DateTimeValue("2024-01-01T12:00:00", "UTC")
        assert dt.timezone == "UTC"
        assert dt.as_pandas_value().hour == 12
        assert str(dt.as_pandas_value().tz) == "UTC"
    
    def test_timezone_conversion(self):
        # Create datetime in UTC
        dt_utc = DateTimeValue("2024-01-01T12:00:00+00:00", "UTC")
        # Create same instant in EST (UTC-5)
        dt_est = DateTimeValue("2024-01-01T12:00:00+00:00", "US/Eastern")
        
        # Should be same instant but displayed in EST
        assert dt_est.as_pandas_value().hour == 7  # 12 UTC = 7 EST
        assert dt_utc.as_pandas_value().timestamp() == dt_est.as_pandas_value().timestamp()

    def test_timezone_conversion_falls_back_without_zoneinfo(self, monkeypatch):
        monkeypatch.setattr(temporal_constructors, "ZoneInfo", None)

        dt_est = DateTimeValue("2024-01-01T12:00:00+00:00", "US/Eastern")

        assert dt_est.as_pandas_value().hour == 7

    def test_from_pandas_timestamp_naive_utc(self):
        ts = pd.Timestamp("2024-01-01 12:00:00")
        dt = DateTimeValue.from_pandas_timestamp(ts)
        assert dt.timezone == "UTC"
        assert str(dt.as_pandas_value().tz) == "UTC"
    
    def test_to_json(self):
        dt = DateTimeValue("2024-01-01T12:00:00Z", "UTC")
        json_data = dt.to_json()
        assert json_data == {
            "type": "datetime",
            "value": "2024-01-01T12:00:00Z",
            "timezone": "UTC"
        }


class TestDateValue:
    def test_parse_date(self):
        d = DateValue("2024-01-01")
        assert d.value == "2024-01-01"
        assert d._parsed == date(2024, 1, 1)
        assert isinstance(d.as_pandas_value(), pd.Timestamp)
        assert d.as_pandas_value().date() == date(2024, 1, 1)
    
    def test_to_json(self):
        d = DateValue("2024-01-01")
        json_data = d.to_json()
        assert json_data == {
            "type": "date",
            "value": "2024-01-01"
        }


class TestTimeValue:
    def test_parse_time(self):
        t = TimeValue("14:30:00")
        assert t.value == "14:30:00"
        assert t._parsed == time(14, 30, 0)
        assert isinstance(t.as_pandas_value(), time)
        assert t.as_pandas_value().hour == 14
        assert t.as_pandas_value().minute == 30
    
    def test_to_json(self):
        t = TimeValue("14:30:00")
        json_data = t.to_json()
        assert json_data == {
            "type": "time",
            "value": "14:30:00"
        }


class TestTemporalValueFromJson:
    def test_datetime_from_json(self):
        json_data = {"type": "datetime", "value": "2024-01-01T12:00:00Z", "timezone": "UTC"}
        dt = temporal_value_from_json(json_data)
        assert isinstance(dt, DateTimeValue)
        assert dt.value == "2024-01-01T12:00:00Z"
        assert dt.timezone == "UTC"
    
    def test_date_from_json(self):
        json_data = {"type": "date", "value": "2024-01-01"}
        d = temporal_value_from_json(json_data)
        assert isinstance(d, DateValue)
        assert d.value == "2024-01-01"
    
    def test_time_from_json(self):
        json_data = {"type": "time", "value": "14:30:00"}
        t = temporal_value_from_json(json_data)
        assert isinstance(t, TimeValue)
        assert t.value == "14:30:00"
    
    def test_invalid_type(self):
        json_data = {"type": "invalid", "value": "something"}
        with pytest.raises(ValueError, match="Unknown temporal value type"):
            temporal_value_from_json(json_data)


class TestTemporalComparisons:
    def test_gt_localizes_naive_series(self):
        s = pd.Series(pd.to_datetime(["2024-01-01 05:00:00", "2024-01-01 08:00:00"]))
        predicate = gt(DateTimeValue("2024-01-01T06:00:00", "UTC"))
        result = predicate(s)
        expected = pd.Series([False, True])
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize("unit", ["us", "ms"])
    def test_gt_naive_series_unit_variants(self, unit):
        s = pd.Series(pd.to_datetime(["2024-01-01 05:00:00", "2024-01-01 08:00:00"]))
        s = s.astype(f"datetime64[{unit}]")
        predicate = gt(DateTimeValue("2024-01-01T06:00:00", "UTC"))
        result = predicate(s)
        expected = pd.Series([False, True])
        pd.testing.assert_series_equal(result, expected)

    @pytest.mark.parametrize("unit", ["us", "ms"])
    @pytest.mark.parametrize(
        "predicate_factory, expected",
        [
            (lt, [True, False]),
            (le, [True, False]),
            (ge, [False, True]),
            (eq, [False, False]),
            (ne, [True, True]),
        ],
    )
    def test_comparison_unit_variants(self, unit, predicate_factory, expected):
        s = pd.Series(pd.to_datetime(["2024-01-01 05:00:00", "2024-01-01 08:00:00"]))
        s = s.astype(f"datetime64[{unit}]")
        predicate = predicate_factory(DateTimeValue("2024-01-01T06:00:00", "UTC"))
        result = predicate(s)
        expected_series = pd.Series(expected)
        pd.testing.assert_series_equal(result, expected_series)

    def test_gt_converts_timezone_aware_series(self):
        s = pd.Series(pd.to_datetime(["2024-01-01 12:00:00", "2024-01-01 14:00:00"], utc=True))
        predicate = gt(DateTimeValue("2024-01-01T08:00:00", "US/Eastern"))
        result = predicate(s)
        expected = pd.Series([False, True])
        pd.testing.assert_series_equal(result, expected)

    @requires_cudf
    def test_gt_cudf_parity(self):
        import cudf
        s_pandas = pd.Series(pd.to_datetime(["2024-01-01 05:00:00", "2024-01-01 08:00:00"]))
        s_cudf = cudf.Series(s_pandas)
        if not hasattr(s_cudf, "dt") or not hasattr(s_cudf.dt, "tz_localize"):
            pytest.skip("cudf timezone localization not supported")
        try:
            _ = s_cudf.dt.tz_localize("UTC")
        except Exception:
            pytest.skip("cudf timezone localization not supported")
        predicate = gt(DateTimeValue("2024-01-01T06:00:00", "UTC"))
        result_pandas = predicate(s_pandas)
        result_cudf = predicate(s_cudf).to_pandas()
        pd.testing.assert_series_equal(result_pandas, result_cudf)

@pytest.mark.parametrize("engine", ["pandas", "cudf"])
def test_date_time_comparison_is_engine_identical(engine: str) -> None:
    """Date/time comparisons must agree across engines and cudf versions.

    ``Series.dt.date`` / ``.dt.time`` do not exist on cudf 26.02, so the predicate
    day-truncates instead -- one formulation available on every engine and every
    version, which is why there is no capability branch and no minimum-cudf
    assumption to encode in the types. This pins that the substitute is an
    EQUIVALENCE, not merely something that does not raise: the same comparisons,
    including the boundary equalities where date-vs-datetime64 semantics could
    diverge, must give identical answers on each engine.
    """
    from graphistry.compute.predicates.comparison import EQ, GT
    from graphistry.compute.ast_temporal import DateValue, TimeValue

    if engine == "cudf":
        cudf = pytest.importorskip("cudf")
        make = cudf.Series
    else:
        make = pd.Series

    stamps = ["2026-01-01 03:00:00", "2026-01-05 09:30:00", "2026-01-10 21:45:00"]
    s = make(pd.to_datetime(stamps))

    def as_list(mask: object) -> list:
        return [bool(x) for x in (mask.to_pandas() if hasattr(mask, "to_pandas") else mask)]

    assert as_list(GT(DateValue(value="2026-01-03"))(s)) == [False, True, True]
    assert as_list(GT(TimeValue(value="08:00:00"))(s)) == [False, True, True]
    # boundary equality: where .dt.date (object) and floor('D') (datetime64) could differ
    assert as_list(EQ(DateValue(value="2026-01-05"))(s)) == [False, True, False]
    assert as_list(EQ(TimeValue(value="09:30:00"))(s)) == [False, True, False]
