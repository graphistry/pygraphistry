from __future__ import annotations

from graphistry.compute.gfql.temporal import constructors


def test_normalize_temporal_constructor_text_supports_named_zones_without_zoneinfo(monkeypatch) -> None:
    monkeypatch.setattr(constructors, "ZoneInfo", None)

    result = constructors.normalize_temporal_constructor_text(
        "datetime('2015-07-21T21:40:32.142[Europe/London]')"
    )

    assert result == "2015-07-21T21:40:32.142+01:00[Europe/London]"
