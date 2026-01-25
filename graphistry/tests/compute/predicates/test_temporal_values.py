import pytest
import pandas as pd
from datetime import datetime, date, time

from graphistry.compute.ast_temporal import (
    DateTimeValue, DateValue, TimeValue, temporal_value_from_json
)
from graphistry.compute.predicates.comparison import gt
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
