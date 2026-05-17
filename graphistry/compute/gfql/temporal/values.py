from __future__ import annotations

from dataclasses import dataclass
from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import timedelta
from datetime import timezone as py_timezone
import re
from typing import Optional, cast

from graphistry.compute.gfql import temporal_text as _tt

@dataclass(frozen=True)
class _TemporalValue:
    kind: str
    date_value: Optional[py_date]
    hour: int = 0
    minute: int = 0
    second: int = 0
    nanosecond: int = 0
    tz_suffix: Optional[str] = None


@dataclass(frozen=True)
class _WideTemporalValue:
    kind: str
    year: int
    month: int
    day: int
    hour: int = 0
    minute: int = 0
    second: int = 0
    nanosecond: int = 0


_ZONE_SUFFIX_RE = re.compile(r"^(?P<core>.+?)(?:\[(?P<zone>[^\]]+)\])?$")


def _format_localtime_parts(hour: int, minute: int, second: int, nanosecond: int) -> str:
    out = f"{hour:02d}:{minute:02d}"
    if second != 0 or nanosecond != 0:
        out += f":{second:02d}{_tt._normalize_fraction(nanosecond)}"
    return out


def _format_localdatetime_parts(
    value_date: py_date,
    hour: int,
    minute: int,
    second: int,
    nanosecond: int,
) -> str:
    return f"{_tt._format_date(value_date.year, value_date.month, value_date.day)}T{_format_localtime_parts(hour, minute, second, nanosecond)}"


def _split_zone_name(text: str) -> tuple[str, Optional[str]]:
    match = _ZONE_SUFFIX_RE.fullmatch(text)
    if match is None:
        return text, None
    return match.group("core"), match.group("zone")


def _parse_temporal_value(text: str) -> Optional[_TemporalValue]:
    stripped = text.strip()
    if stripped.startswith(("P", "-P")):
        return None
    if "T" in stripped:
        date_text, time_text = stripped.split("T", 1)
        value_date = _tt._base_date_from_text(date_text)
        if value_date is None:
            return None
        time_core, zone_name = _split_zone_name(time_text)
        parts = _tt._base_time_parts_from_text(time_core)
        if parts is None:
            return None
        tz = cast(Optional[str], parts.get("tz"))
        tz_suffix = tz if zone_name is None else ("" if tz is None else tz) + f"[{zone_name}]"
        return _TemporalValue(
            kind="datetime" if tz_suffix is not None else "localdatetime",
            date_value=value_date,
            hour=cast(int, parts["hour"]),
            minute=cast(int, parts["minute"]),
            second=cast(int, parts["second"]),
            nanosecond=cast(int, parts["nanosecond"]),
            tz_suffix=tz_suffix,
        )
    if ":" in stripped:
        time_core, zone_name = _split_zone_name(stripped)
        parts = _tt._base_time_parts_from_text(time_core)
        if parts is None:
            return None
        tz = cast(Optional[str], parts.get("tz"))
        tz_suffix = tz if zone_name is None else ("" if tz is None else tz) + f"[{zone_name}]"
        return _TemporalValue(
            kind="time" if tz_suffix is not None else "localtime",
            date_value=None,
            hour=cast(int, parts["hour"]),
            minute=cast(int, parts["minute"]),
            second=cast(int, parts["second"]),
            nanosecond=cast(int, parts["nanosecond"]),
            tz_suffix=tz_suffix,
        )
    value_date = _tt._base_date_from_text(stripped)
    if value_date is None:
        return None
    return _TemporalValue(kind="date", date_value=value_date)


def _truncate_year(year: int, span: int) -> int:
    return (year // span) * span


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0) and ((year % 100 != 0) or (year % 400 == 0))


def _days_in_month(year: int, month: int) -> int:
    if month in {1, 3, 5, 7, 8, 10, 12}:
        return 31
    if month in {4, 6, 9, 11}:
        return 30
    if month == 2:
        return 29 if _is_leap_year(year) else 28
    raise ValueError(f"invalid month: {month}")


def _days_from_civil(year: int, month: int, day: int) -> int:
    adjusted_year = year - (1 if month <= 2 else 0)
    era = adjusted_year // 400
    year_of_era = adjusted_year - (era * 400)
    month_prime = month + (-3 if month > 2 else 9)
    day_of_year = ((153 * month_prime) + 2) // 5 + day - 1
    day_of_era = year_of_era * 365 + year_of_era // 4 - year_of_era // 100 + day_of_year
    return era * 146097 + day_of_era - 719468


def _parse_wide_temporal_value(text: str) -> Optional[_WideTemporalValue]:
    stripped = text.strip()
    localdatetime_match = _tt._WIDE_LOCALDATETIME_TEXT_RE.fullmatch(stripped)
    if localdatetime_match is not None:
        year = int(localdatetime_match.group("year"))
        month = int(localdatetime_match.group("month"))
        day = int(localdatetime_match.group("day"))
        hour = int(localdatetime_match.group("hour"))
        minute = int(localdatetime_match.group("minute"))
        second = int(localdatetime_match.group("second") or "0")
        if not 1 <= month <= 12 or not 1 <= day <= _days_in_month(year, month):
            return None
        if hour > 23 or minute > 59 or second > 59:
            return None
        return _WideTemporalValue(
            kind="localdatetime",
            year=year,
            month=month,
            day=day,
            hour=hour,
            minute=minute,
            second=second,
            nanosecond=_tt._parse_fraction_to_nanos(localdatetime_match.group("frac")),
        )
    date_match = _tt._WIDE_DATE_TEXT_RE.fullmatch(stripped)
    if date_match is None:
        return None
    year = int(date_match.group("year"))
    month = int(date_match.group("month"))
    day = int(date_match.group("day"))
    if not 1 <= month <= 12 or not 1 <= day <= _days_in_month(year, month):
        return None
    return _WideTemporalValue(kind="date", year=year, month=month, day=day)


def _comparable_datetime(
    value: _TemporalValue,
    *,
    include_date: bool,
    keep_timezone: bool,
    anchor_date: Optional[py_date] = None,
    anchor_tz_suffix: Optional[str] = None,
) -> py_datetime:
    if include_date:
        value_date = value.date_value or anchor_date or py_date(1970, 1, 1)
    else:
        value_date = anchor_date or py_date(1970, 1, 1)
    effective_tz_suffix = value.tz_suffix or anchor_tz_suffix
    naive = py_datetime(
        value_date.year,
        value_date.month,
        value_date.day,
        value.hour,
        value.minute,
        value.second,
        value.nanosecond // 1_000,
    )
    if keep_timezone and effective_tz_suffix is not None:
        offset = effective_tz_suffix.split("[", 1)[0]
        zone_match = re.search(r"\[(?P<zone>[^\]]+)\]$", effective_tz_suffix)
        if zone_match is not None:
            zone_name = zone_match.group("zone")
            historical_override = _tt._neo4j_historical_zone_offset(zone_name, naive)
            if historical_override is not None:
                return naive.replace(tzinfo=py_timezone(historical_override))
            zone = _tt._named_timezone(zone_name)
            if zone is not None:
                return naive.replace(tzinfo=zone)
        if offset == "Z":
            return naive.replace(tzinfo=py_timezone.utc)
        offset_delta = py_timedelta_from_offset(offset)
        if offset_delta is not None:
            return naive.replace(tzinfo=py_timezone(offset_delta))
        return naive
    return naive


def py_timedelta_from_offset(offset: str) -> Optional[timedelta]:
    match = re.fullmatch(r"(?P<sign>[+-])(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2}))?", offset)
    if match is None:
        return None
    sign = 1 if match.group("sign") == "+" else -1
    delta = timedelta(
        hours=int(match.group("hour")),
        minutes=int(match.group("minute")),
        seconds=int(match.group("second") or "0"),
    )
    return delta if sign > 0 else -delta


def _timedelta_total_microseconds(delta: timedelta) -> int:
    return ((delta.days * 24 * 60 * 60) + delta.seconds) * 1_000_000 + delta.microseconds


def _absolute_temporal_delta(start_dt: py_datetime, end_dt: py_datetime) -> timedelta:
    if start_dt.tzinfo is not None and end_dt.tzinfo is not None:
        return end_dt.astimezone(py_timezone.utc) - start_dt.astimezone(py_timezone.utc)
    return end_dt - start_dt
