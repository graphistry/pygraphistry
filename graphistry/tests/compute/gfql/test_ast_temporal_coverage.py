from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, timezone, tzinfo

import pandas as pd
import pytest

import graphistry.compute.ast_temporal as ast_temporal
from graphistry.compute.ast_temporal import DateTimeValue, DateValue, TimeValue, temporal_value_from_json


class _FixedTz(tzinfo):
    def utcoffset(self, dt: datetime | None) -> timedelta | None:
        return timezone.utc.utcoffset(dt)

    def dst(self, dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, dt: datetime | None) -> str:
        return "FixedUTC"


def _raise_zoneinfo(_name: str) -> tzinfo:
    raise ValueError("missing timezone")


def test_resolve_timezone_prefers_utc_and_falls_back_to_dateutil(monkeypatch: pytest.MonkeyPatch) -> None:
    assert ast_temporal._resolve_timezone("UTC") is timezone.utc

    fallback = _FixedTz()
    monkeypatch.setattr(ast_temporal, "ZoneInfo", _raise_zoneinfo)
    monkeypatch.setattr(ast_temporal, "_dateutil_gettz", lambda _name: fallback)

    assert ast_temporal._resolve_timezone("Example/Zone") is fallback


def test_resolve_timezone_returns_none_without_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ast_temporal, "ZoneInfo", _raise_zoneinfo)
    monkeypatch.setattr(ast_temporal, "_dateutil_gettz", None)

    assert ast_temporal._resolve_timezone("Example/Zone") is None


def test_datetime_value_round_trips_and_converts_timezone() -> None:
    value = DateTimeValue("2024-01-01T12:00:00+00:00", "US/Eastern")

    assert value.to_json() == {
        "type": "datetime",
        "value": "2024-01-01T12:00:00+00:00",
        "timezone": "US/Eastern",
    }
    assert value.as_pandas_value().hour == 7
    assert value.as_pandas_value().timestamp() == pd.Timestamp("2024-01-01T12:00:00Z").timestamp()


def test_datetime_value_native_factories_localize_naive_inputs() -> None:
    from_datetime = DateTimeValue.from_datetime(datetime(2024, 1, 2, 3, 4, 5))
    assert from_datetime.to_json() == {
        "type": "datetime",
        "value": "2024-01-02T03:04:05",
        "timezone": "UTC",
    }
    assert str(from_datetime.as_pandas_value().tz) == "UTC"

    from_timestamp = DateTimeValue.from_pandas_timestamp(pd.Timestamp("2024-01-02T03:04:05"))
    assert from_timestamp.timezone == "UTC"
    assert str(from_timestamp.as_pandas_value().tz) == "UTC"


def test_date_and_time_values_round_trip_native_and_datetime_text() -> None:
    date_value = DateValue.from_date(date(2024, 2, 3))
    assert date_value.to_json() == {"type": "date", "value": "2024-02-03"}
    assert date_value.as_pandas_value() == pd.Timestamp("2024-02-03")

    time_value = TimeValue.from_time(time(14, 30, 5))
    assert time_value.to_json() == {"type": "time", "value": "14:30:05"}
    assert time_value.as_pandas_value() == time(14, 30, 5)

    from_datetime_text = TimeValue("2024-02-03T06:07:08")
    assert from_datetime_text.as_pandas_value() == time(6, 7, 8)


@pytest.mark.parametrize(
    ("payload", "expected_type"),
    [
        ({"type": "datetime", "value": "2024-01-02T03:04:05"}, DateTimeValue),
        ({"type": "date", "value": "2024-01-02"}, DateValue),
        ({"type": "time", "value": "03:04:05"}, TimeValue),
    ],
)
def test_temporal_value_from_json_dispatches_supported_types(
    payload: dict[str, str],
    expected_type: type[DateTimeValue] | type[DateValue] | type[TimeValue],
) -> None:
    value = temporal_value_from_json(payload)

    assert isinstance(value, expected_type)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: temporal_value_from_json({"type": "duration", "value": "P1D"}),
        lambda: DateTimeValue("not-a-datetime"),
        lambda: DateValue("not-a-date"),
        lambda: TimeValue("not-a-time"),
    ],
)
def test_temporal_values_reject_unknown_types_and_invalid_literals(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError):
        factory()
