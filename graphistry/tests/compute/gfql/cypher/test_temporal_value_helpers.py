from __future__ import annotations

from datetime import date as py_date

from graphistry.compute.gfql.temporal.values import _TemporalValue, _comparable_datetime


def test_comparable_datetime_handles_trimmed_fraction_without_fromisoformat_dependency() -> None:
    value = _TemporalValue(
        kind="localtime",
        date_value=None,
        hour=15,
        minute=32,
        second=38,
        nanosecond=947_410_000,
        tz_suffix=None,
    )

    result = _comparable_datetime(
        value,
        include_date=False,
        keep_timezone=False,
        anchor_date=py_date(1970, 1, 1),
    )

    assert result.isoformat() == "1970-01-01T15:32:38.947410"
