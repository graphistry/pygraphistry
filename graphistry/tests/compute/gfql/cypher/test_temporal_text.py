from __future__ import annotations

from datetime import date as py_date

import graphistry.compute.gfql.temporal_text as temporal_text


def test_normalize_temporal_constructor_text_supports_named_zones_without_zoneinfo(monkeypatch) -> None:
    monkeypatch.setattr(temporal_text, "ZoneInfo", None)

    result = temporal_text.normalize_temporal_constructor_text(
        "datetime('2015-07-21T21:40:32.142[Europe/London]')"
    )

    assert result == "2015-07-21T21:40:32.142+01:00[Europe/London]"


def test_comparable_datetime_handles_trimmed_fraction_without_fromisoformat_dependency() -> None:
    value = temporal_text._TemporalValue(
        kind="localtime",
        date_value=None,
        hour=15,
        minute=32,
        second=38,
        nanosecond=947_410_000,
        tz_suffix=None,
    )

    result = temporal_text._comparable_datetime(
        value,
        include_date=False,
        keep_timezone=False,
        anchor_date=py_date(1970, 1, 1),
    )

    assert result.isoformat() == "1970-01-01T15:32:38.947410"
