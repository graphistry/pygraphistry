from __future__ import annotations

from datetime import date as py_date
from datetime import timedelta
import re
from typing import Optional, cast

from graphistry.compute.gfql.temporal import constructors as _tt
from graphistry.compute.gfql.expr_parser import ExprNode, Literal, MapLiteral
from graphistry.compute.gfql.temporal.rendering import _render_temporal_arg
from graphistry.compute.gfql.temporal.values import (
    _TemporalValue,
    _format_localdatetime_parts,
    _format_localtime_parts,
    _parse_temporal_value,
    _truncate_year,
)

_DATE_TRUNCATION_UNITS = frozenset({"millennium", "century", "decade", "year", "weekYear", "quarter", "month", "week", "day"})

def _truncate_date_value(source_date: py_date, unit: str, overrides: dict[str, str]) -> Optional[py_date]:
    if unit == "millennium":
        base = py_date(_truncate_year(source_date.year, 1000), 1, 1)
    elif unit == "century":
        base = py_date(_truncate_year(source_date.year, 100), 1, 1)
    elif unit == "decade":
        base = py_date(_truncate_year(source_date.year, 10), 1, 1)
    elif unit == "year":
        base = py_date(source_date.year, 1, 1)
    elif unit == "quarter":
        base = py_date(source_date.year, ((source_date.month - 1) // 3) * 3 + 1, 1)
    elif unit == "month":
        base = py_date(source_date.year, source_date.month, 1)
    elif unit == "week":
        base = source_date - timedelta(days=source_date.isoweekday() - 1)
        if "dayOfWeek" in overrides:
            return base + timedelta(days=int(overrides["dayOfWeek"]) - 1)
        if "day" in overrides:
            return base + timedelta(days=int(overrides["day"]) - 1)
    elif unit == "weekYear":
        base = py_date.fromisocalendar(source_date.isocalendar()[0], 1, 1)
        if "day" in overrides:
            return py_date(base.year, 1, 1) + timedelta(days=int(overrides["day"]) - 1)
        if "dayOfWeek" in overrides:
            return base + timedelta(days=int(overrides["dayOfWeek"]) - 1)
    elif unit == "day":
        base = source_date
    else:
        return None
    fields = {"date": _tt._format_date(base.year, base.month, base.day)}
    fields.update(overrides)
    return _tt._date_from_fields(fields)


def _truncate_time_parts(
    source_value: _TemporalValue,
    unit: str,
    overrides: dict[str, str],
) -> tuple[int, int, int, int]:
    if unit in _DATE_TRUNCATION_UNITS:
        hour = 0
        minute = 0
        second = 0
        nanosecond = 0
    else:
        hour = source_value.hour
        minute = source_value.minute
        second = source_value.second
        nanosecond = source_value.nanosecond

    if unit == "day":
        hour = 0
        minute = 0
        second = 0
        nanosecond = 0
    elif unit == "hour":
        minute = 0
        second = 0
        nanosecond = 0
    elif unit == "minute":
        second = 0
        nanosecond = 0
    elif unit == "second":
        nanosecond = 0
    elif unit == "millisecond":
        nanosecond = (nanosecond // 1_000_000) * 1_000_000
    elif unit == "microsecond":
        nanosecond = (nanosecond // 1_000) * 1_000

    if "hour" in overrides:
        hour = int(overrides["hour"])
    if "minute" in overrides:
        minute = int(overrides["minute"])
    if "second" in overrides:
        second = int(overrides["second"])
    if any(key in overrides for key in ("nanosecond", "microsecond", "millisecond")):
        truncated_millisecond = nanosecond // 1_000_000
        truncated_microsecond = (nanosecond // 1_000) % 1_000
        truncated_nanosecond = nanosecond % 1_000
        millisecond = _tt._parse_int(
            overrides.get("millisecond", overrides.get("milliseconds")),
            truncated_millisecond,
        )
        microsecond = _tt._parse_int(
            overrides.get("microsecond", overrides.get("microseconds")),
            truncated_microsecond,
        )
        sub_nanosecond = _tt._parse_int(
            overrides.get("nanosecond", overrides.get("nanoseconds")),
            truncated_nanosecond,
        )
        if millisecond is None or microsecond is None or sub_nanosecond is None:
            return hour, minute, second, nanosecond
        nanosecond = (millisecond * 1_000_000) + (microsecond * 1_000) + sub_nanosecond

    return hour, minute, second, nanosecond


def _zone_compatible_local_datetime_text(date_value: py_date, local_time_text: str) -> str:
    local_dt_text = f"{_tt._format_date(date_value.year, date_value.month, date_value.day)}T{local_time_text}"
    match = re.fullmatch(r"(?P<prefix>.+?\.\d{6})\d+(?P<suffix>)", local_dt_text)
    if match is not None:
        return match.group("prefix")
    return local_dt_text


def _target_timezone_suffix(
    target_kind: str,
    source_value: _TemporalValue,
    overrides: dict[str, str],
    *,
    target_date: Optional[py_date],
    local_time_text: str,
) -> Optional[str]:
    if target_kind not in {"time", "datetime"}:
        return None
    timezone_text = overrides.get("timezone")
    if timezone_text is None:
        return source_value.tz_suffix or "Z"
    zone_name = _tt._parse_quoted(timezone_text)
    if zone_name is None and not re.fullmatch(r"Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?", timezone_text):
        zone_name = timezone_text
    if zone_name is None:
        return _tt._normalize_offset_text(timezone_text)
    zone_base_date = target_date or source_value.date_value or py_date(1970, 1, 1)
    suffix = _tt._zone_suffix(
        zone_name,
        _zone_compatible_local_datetime_text(zone_base_date, local_time_text),
    )
    return suffix


def _fold_temporal_truncate_call(
    fn_name: str,
    args: tuple[ExprNode, ...],
) -> Optional[Literal]:
    if len(args) != 3:
        return None
    unit_node, value_node, override_node = args
    if not isinstance(unit_node, Literal) or not isinstance(unit_node.value, str):
        return None
    if not isinstance(value_node, Literal) or not isinstance(value_node.value, str):
        return None
    if not isinstance(override_node, MapLiteral):
        return None

    overrides: dict[str, str] = {}
    for key, value in override_node.items:
        rendered = _render_temporal_arg(value)
        if rendered is None:
            return None
        parsed = _tt._parse_quoted(rendered)
        overrides[key] = rendered if parsed is None else parsed

    source_value = _parse_temporal_value(value_node.value)
    if source_value is None:
        return None

    unit = unit_node.value
    target_kind = fn_name.split(".", 1)[0]
    target_date: Optional[py_date] = source_value.date_value
    if target_kind == "date":
        if target_date is None:
            target_date = py_date(1970, 1, 1)
        target_date = _truncate_date_value(target_date, unit, overrides)
        if target_date is None:
            return None
    elif target_kind in {"datetime", "localdatetime"} and unit in _DATE_TRUNCATION_UNITS:
        if target_date is None:
            target_date = py_date(1970, 1, 1)
        target_date = _truncate_date_value(target_date, unit, overrides)
        if target_date is None:
            return None

    hour, minute, second, nanosecond = _truncate_time_parts(source_value, unit, overrides)

    if target_kind == "date":
        assert target_date is not None
        return Literal(_tt._format_date(target_date.year, target_date.month, target_date.day))

    local_time_text = _format_localtime_parts(hour, minute, second, nanosecond)
    if target_kind == "localtime":
        return Literal(local_time_text)

    tz_suffix = _target_timezone_suffix(
        target_kind,
        source_value,
        overrides,
        target_date=target_date,
        local_time_text=local_time_text,
    )

    if target_kind == "time":
        return Literal(local_time_text + cast(str, tz_suffix))

    assert target_date is not None
    local_dt_text = _format_localdatetime_parts(target_date, hour, minute, second, nanosecond)
    if target_kind == "localdatetime":
        return Literal(local_dt_text)
    return Literal(local_dt_text + cast(str, tz_suffix))
