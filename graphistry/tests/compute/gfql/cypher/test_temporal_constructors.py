from __future__ import annotations

import pytest

from graphistry.compute.gfql.temporal import constructors


def test_normalize_temporal_constructor_text_supports_named_zones_without_zoneinfo(monkeypatch) -> None:
    monkeypatch.setattr(constructors, "ZoneInfo", None)

    result = constructors.normalize_temporal_constructor_text(
        "datetime('2015-07-21T21:40:32.142[Europe/London]')"
    )

    assert result == "2015-07-21T21:40:32.142+01:00[Europe/London]"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("date({date: '1984-10-11', quarter: 4})", "1984-10-11"),
        ("date({year: 1984, ordinalDay: 42})", "1984-02-11"),
        ("date('+10000-01-01')", "10000-01-01"),
        ("date('198410')", "1984-10-01"),
        ("date('1984W103')", "1984-03-07"),
        ("date('1984042')", "1984-02-11"),
        ("date('1984')", "1984-01-01"),
        ("localtime({time: '12:00:00.123456789'})", "12:00:00.123456789"),
        ("time({time: '12:00:00+01:00', timezone: '+00:00'})", "11:00:00Z"),
        ("time('12:00')", "12:00Z"),
        ("localdatetime('1984-10-11T12')", "1984-10-11T12:00"),
        ("datetime({datetime: '1984-10-11T12:00:00+01:00', timezone: '+00:00'})", "1984-10-11T11:00:00Z"),
        ("datetime({datetime: '1984-10-11T12:00:00+01:00', day: 12, timezone: '+00:00'})", "1984-10-12T11:00:00Z"),
        ("duration('P1Y2M3W4DT5H6M7.8S')", "P1Y2M25DT5H6M7.8S"),
        ("duration({})", "PT0S"),
    ],
)
def test_normalize_temporal_constructor_text_covers_refactored_branches(expr: str, expected: str) -> None:
    assert constructors.normalize_temporal_constructor_text(expr) == expected


@pytest.mark.parametrize(
    "expr",
    [
        "date({week: 1})",
        "date({ordinalDay: nope})",
        "date('+10000-13-01')",
        "date({bad})",
        "time('not')",
        "datetime('not')",
        "duration('')",
        "duration('P')",
        "duration('P1H')",
        "duration('PT1D')",
        "duration(1)",
        "nontemporal('1984-10-11')",
    ],
)
def test_normalize_temporal_constructor_text_rejects_invalid_shapes(expr: str) -> None:
    assert constructors.normalize_temporal_constructor_text(expr) is None
