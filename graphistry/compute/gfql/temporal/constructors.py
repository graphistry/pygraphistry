from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import timedelta
from datetime import timezone as py_timezone
from datetime import tzinfo as py_tzinfo
import re
import sys
from typing import Callable, Optional, cast

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    ZoneInfo = None

try:
    from dateutil.tz import gettz as _dateutil_gettz  # type: ignore[import-untyped]
except Exception:
    _dateutil_gettz = None

from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal,
    MapLiteral,
    QuantifierExpr,
    SliceExpr,
    SubscriptExpr,
    UnaryOp,
    Wildcard,
)


_TEMPORAL_FUNC_RE = re.compile(
    r"^(?P<fn>date|localtime|time|localdatetime|datetime|duration)\((?P<arg>.*)\)$"
)
TEMPORAL_CALL_EXPR_RE = re.compile(
    r"(?:localdatetime|localtime|datetime|time|date|duration)\((?:\{[^()]*\}|'[^']*')\)"
)
CURRENT_TEMPORAL_CALL_EXPR_RE = re.compile(r"\b(?P<fn>localdatetime|localtime|datetime|time|date)\(\)")
_SIMPLE_QUOTED_RE = re.compile(r"^'([^']*)'$")

# Compatibility regexes consumed by row ordering helpers for constructor-text
# detection and vectorized sort-key extraction on stored node/edge properties.
DATE_CALL_TEXT_RE = re.compile(
    r"^date\(\{\s*year:\s*(?P<year>-?\d+)\s*,\s*month:\s*(?P<month>\d+)\s*,\s*day:\s*(?P<day>\d+)\s*\}\)$"
)
LOCALTIME_CALL_TEXT_RE = re.compile(
    r"^localtime\(\{\s*hour:\s*(?P<hour>\d+)\s*,\s*minute:\s*(?P<minute>\d+)"
    r"(?:\s*,\s*second:\s*(?P<second>\d+))?"
    r"(?:\s*,\s*nanosecond:\s*(?P<nano>\d+))?"
    r"\s*\}\)$"
)
TIME_CALL_TEXT_RE = re.compile(
    r"^time\(\{\s*hour:\s*(?P<hour>\d+)\s*,\s*minute:\s*(?P<minute>\d+)"
    r"(?:\s*,\s*second:\s*(?P<second>\d+))?"
    r"(?:\s*,\s*nanosecond:\s*(?P<nano>\d+))?"
    r"\s*,\s*timezone:\s*'(?P<tz>[^']+)'\s*\}\)$"
)
LOCALDATETIME_CALL_TEXT_RE = re.compile(
    r"^localdatetime\(\{\s*year:\s*(?P<year>-?\d+)\s*,\s*month:\s*(?P<month>\d+)\s*,\s*day:\s*(?P<day>\d+)"
    r"\s*,\s*hour:\s*(?P<hour>\d+)\s*,\s*minute:\s*(?P<minute>\d+)"
    r"(?:\s*,\s*second:\s*(?P<second>\d+))?"
    r"(?:\s*,\s*nanosecond:\s*(?P<nano>\d+))?"
    r"\s*\}\)$"
)
DATETIME_CALL_TEXT_RE = re.compile(
    r"^datetime\(\{\s*year:\s*(?P<year>-?\d+)\s*,\s*month:\s*(?P<month>\d+)\s*,\s*day:\s*(?P<day>\d+)"
    r"\s*,\s*hour:\s*(?P<hour>\d+)\s*,\s*minute:\s*(?P<minute>\d+)"
    r"(?:\s*,\s*second:\s*(?P<second>\d+))?"
    r"(?:\s*,\s*nanosecond:\s*(?P<nano>\d+))?"
    r"\s*,\s*timezone:\s*'(?P<tz>[^']+)'\s*\}\)$"
)

_DATE_YMD_RE = re.compile(r"^(?P<year>-?\d{1,9})-(?P<month>\d{2})-(?P<day>\d{2})$")
_DATE_COMPACT_YMD_RE = re.compile(r"^(?P<year>-?\d{4})(?P<month>\d{2})(?P<day>\d{2})$")
_DATE_YM_RE = re.compile(r"^(?P<year>-?\d{1,9})-(?P<month>\d{2})$")
_DATE_COMPACT_YM_RE = re.compile(r"^(?P<year>-?\d{4})(?P<month>\d{2})$")
_DATE_WEEK_RE = re.compile(r"^(?P<year>-?\d{4})-W(?P<week>\d{2})(?:-(?P<dow>\d))?$")
_DATE_COMPACT_WEEK_RE = re.compile(r"^(?P<year>-?\d{4})W(?P<week>\d{2})(?P<dow>\d)?$")
_DATE_ORDINAL_RE = re.compile(r"^(?P<year>-?\d{4})-(?P<ordinal>\d{3})$")
_DATE_COMPACT_ORDINAL_RE = re.compile(r"^(?P<year>-?\d{4})(?P<ordinal>\d{3})$")
_DATE_YEAR_RE = re.compile(r"^(?P<year>-?\d{1,9})$")

_LOCALTIME_RE = re.compile(
    r"^(?P<hour>\d{2})(?::?(?P<minute>\d{2}))?(?::?(?P<second>\d{2}))?(?:\.(?P<frac>\d+))?$"
)
_TIME_RE = re.compile(
    r"^(?P<local>\d{2}(?::?\d{2})?(?::?\d{2})?(?:\.\d+)?)"
    r"(?P<tz>Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?)$"
)
_DATETIME_TZ_RE = re.compile(r"^(?P<core>.+?)(?P<tz>Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?)$")
_NORMALIZED_TIME_PART_RE = re.compile(
    r"^(?P<local>\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?)(?P<tz>Z|[+-]\d{2}:\d{2}(?::\d{2})?)?(?:\[[^\]]+\])?$"
)
_WIDE_DATE_TEXT_RE = re.compile(r"^(?P<year>[+-]?\d{5,})-(?P<month>\d{2})-(?P<day>\d{2})$")
_WIDE_LOCALDATETIME_TEXT_RE = re.compile(
    r"^(?P<year>[+-]?\d{5,})-(?P<month>\d{2})-(?P<day>\d{2})"
    r"T(?P<hour>\d{2}):(?P<minute>\d{2})"
    r"(?::(?P<second>\d{2})(?:\.(?P<frac>\d+))?)?$"
)


def _split_map_items(text: str) -> list[str]:
    inner = text.strip()
    if inner.startswith("{") and inner.endswith("}"):
        inner = inner[1:-1].strip()
    if not inner:
        return []
    items: list[str] = []
    buf: list[str] = []
    in_single = False
    for ch in inner:
        if ch == "'":
            in_single = not in_single
            buf.append(ch)
            continue
        if ch == "," and not in_single:
            item = "".join(buf).strip()
            if item:
                items.append(item)
            buf = []
            continue
        buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        items.append(tail)
    return items


def _parse_map_fields(text: str) -> Optional[dict[str, str]]:
    out: dict[str, str] = {}
    for item in _split_map_items(text):
        if ":" not in item:
            return None
        key, value = item.split(":", 1)
        out[key.strip()] = value.strip()
    return out


def _parse_quoted(value: str) -> Optional[str]:
    match = _SIMPLE_QUOTED_RE.fullmatch(value.strip())
    return match.group(1) if match is not None else None


def _first_present_field_text(fields: dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        raw = fields.get(key)
        if raw is None:
            continue
        return _parse_quoted(raw) or raw
    return None


def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _nanos_from_fields(fields: dict[str, str]) -> int:
    nanos = 0
    if "milliseconds" in fields:
        nanos += int(fields["milliseconds"]) * 1_000_000
    if "millisecond" in fields:
        nanos += int(fields["millisecond"]) * 1_000_000
    if "microseconds" in fields:
        nanos += int(fields["microseconds"]) * 1_000
    if "microsecond" in fields:
        nanos += int(fields["microsecond"]) * 1_000
    if "nanoseconds" in fields:
        nanos += int(fields["nanoseconds"])
    if "nanosecond" in fields:
        nanos += int(fields["nanosecond"])
    return nanos


def _normalize_fraction(nanos: int) -> str:
    if nanos == 0:
        return ""
    frac = str(nanos).zfill(9).rstrip("0")
    return f".{frac}" if frac else ""


def _parse_fraction_to_nanos(frac: Optional[str]) -> int:
    if frac is None or frac == "":
        return 0
    return int(frac.ljust(9, "0")[:9])


def _normalize_offset_text(tz_text: str) -> str:
    if tz_text == "Z":
        return "Z"
    match = re.fullmatch(r"([+-])(\d{2})(?::?(\d{2}))?(?::?(\d{2}))?", tz_text)
    if match is None:
        return tz_text
    sign, hour, minute, second = match.groups()
    minute = minute or "00"
    second = second or "00"
    if hour == "00" and minute == "00" and second == "00":
        return "Z"
    if second == "00":
        return f"{sign}{hour}:{minute}"
    return f"{sign}{hour}:{minute}:{second}"


def _split_tz_suffix_parts(tz_suffix: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    if tz_suffix is None:
        return None, None
    core, zone_name = _split_zone_name(tz_suffix)
    return (core or None), zone_name


def _named_timezone(zone_name: str) -> Optional[py_tzinfo]:
    if ZoneInfo is not None:
        try:
            return cast(py_tzinfo, ZoneInfo(zone_name))
        except Exception:
            pass
    if _dateutil_gettz is not None:
        tzinfo = cast(Callable[[str], Optional[py_tzinfo]], _dateutil_gettz)(zone_name)
        if tzinfo is not None:
            return tzinfo
    return None


def _neo4j_historical_zone_offset(zone_name: str, local_dt: py_datetime) -> Optional[timedelta]:
    """Return known Neo4j/TCK historical offset overrides for named zones.

    Python zone databases and Neo4j/JDK temporal semantics diverge for some
    pre-standard-time historical instants. Keep this mapping narrow and
    explicit so direct-Cypher canonicalization remains deterministic.
    """
    if zone_name == "Europe/Stockholm" and local_dt < py_datetime(1879, 1, 1):
        return timedelta(minutes=53, seconds=28)
    return None


def _format_offset(delta: timedelta) -> str:
    total_seconds = int(delta.total_seconds())
    sign = "+" if total_seconds >= 0 else "-"
    total_seconds = abs(total_seconds)
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if seconds == 0:
        return f"{sign}{hours:02d}:{minutes:02d}"
    return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"


def _format_date(year: int, month: int, day: int) -> str:
    if year >= 0:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return f"-{abs(year):04d}-{month:02d}-{day:02d}"


def _format_arbitrary_date(year: int, month: int, day: int) -> str:
    if -9999 <= year <= 9999:
        return _format_date(year, month, day)
    prefix = "-" if year < 0 else ""
    return f"{prefix}{abs(year)}-{month:02d}-{day:02d}"


def _extract_date_text(text: str) -> Optional[str]:
    candidate = text.strip()
    if "T" in candidate:
        candidate = candidate.split("T", 1)[0]
    return _normalize_date_string(candidate)


def _base_date_from_text(text: Optional[str]) -> Optional[py_date]:
    if text is None:
        return None
    normalized = _extract_date_text(text)
    if normalized is None:
        return None
    match = _DATE_YMD_RE.fullmatch(normalized)
    if match is None:
        return None
    try:
        return py_date(
            int(match.group("year")),
            int(match.group("month")),
            int(match.group("day")),
        )
    except ValueError:
        return None


def _day_of_quarter(value: py_date) -> int:
    start = py_date(value.year, ((value.month - 1) // 3) * 3 + 1, 1)
    return (value - start).days + 1


def _base_time_parts_from_text(text: Optional[str]) -> Optional[dict[str, object]]:
    if text is None:
        return None
    candidate = text.strip()
    if "T" in candidate:
        candidate = candidate.split("T", 1)[1]
    match = _NORMALIZED_TIME_PART_RE.fullmatch(candidate)
    if match is None:
        normalized_local = _normalize_localtime_string(candidate)
        if normalized_local is None:
            normalized_time = _normalize_time_string(candidate)
            if normalized_time is None:
                return None
            candidate = normalized_time
        else:
            candidate = normalized_local
        match = _NORMALIZED_TIME_PART_RE.fullmatch(candidate)
        if match is None:
            return None
    local = match.group("local")
    tz = match.group("tz")
    local_match = re.fullmatch(r"(?P<hour>\d{2}):(?P<minute>\d{2})(?::(?P<second>\d{2})(?:\.(?P<frac>\d+))?)?", local)
    if local_match is None:
        return None
    return {
        "hour": int(local_match.group("hour")),
        "minute": int(local_match.group("minute")),
        "second": int(local_match.group("second") or "0"),
        "nanosecond": _parse_fraction_to_nanos(local_match.group("frac")),
        "tz": tz,
        "has_second": local_match.group("second") is not None,
        "has_fraction": local_match.group("frac") is not None,
    }


def _base_time_int(base: Optional[dict[str, object]], key: str, default: int) -> int:
    if base is None:
        return default
    value = base.get(key)
    return value if isinstance(value, int) else default


def _date_from_fields(
    fields: dict[str, str],
    *,
    base_date: Optional[py_date] = None,
) -> Optional[py_date]:
    base = base_date or _base_date_from_text(_first_present_field_text(fields, "date", "datetime", "localdatetime"))
    if "month" in fields or "day" in fields:
        year = _parse_int(fields.get("year"), base.year if base is not None else None)
        if year is None:
            return None
        month_default = base.month if base is not None else 1
        day_default = base.day if base is not None else 1
        month = _parse_int(fields.get("month"), month_default)
        day = _parse_int(fields.get("day"), day_default)
        if month is None or day is None:
            return None
        return py_date(year, month, day)
    if "week" in fields:
        iso_year_default = base.isocalendar()[0] if base is not None else None
        year = _parse_int(fields.get("year"), iso_year_default)
        if year is None:
            return None
        week = _parse_int(fields.get("week"))
        day_of_week = _parse_int(fields.get("dayOfWeek"), base.isoweekday() if base is not None else 1)
        if week is None or day_of_week is None:
            return None
        return py_date.fromisocalendar(year, week, day_of_week)
    if "ordinalDay" in fields:
        year = _parse_int(fields.get("year"), base.year if base is not None else None)
        if year is None:
            return None
        ordinal = _parse_int(fields.get("ordinalDay"))
        if ordinal is None:
            return None
        return py_date(year, 1, 1) + timedelta(days=ordinal - 1)
    if "quarter" in fields:
        year = _parse_int(fields.get("year"), base.year if base is not None else None)
        if year is None:
            return None
        quarter = _parse_int(fields.get("quarter"))
        day_of_quarter = _parse_int(fields.get("dayOfQuarter"), _day_of_quarter(base) if base is not None else 1)
        if quarter is None or day_of_quarter is None:
            return None
        start = py_date(year, ((quarter - 1) * 3) + 1, 1)
        return start + timedelta(days=day_of_quarter - 1)
    if base is not None:
        year = _parse_int(fields.get("year"), base.year)
        if year is None:
            return None
        month = _parse_int(fields.get("month"), base.month)
        day = _parse_int(fields.get("day"), base.day)
        if month is None or day is None:
            return None
        return py_date(year, month, day)
    year = _parse_int(fields.get("year"))
    if year is None:
        return None
    return py_date(year, 1, 1)


def _normalize_date_map(fields: dict[str, str], *, base_date: Optional[py_date] = None) -> Optional[str]:
    value = _date_from_fields(fields, base_date=base_date)
    if value is None:
        return None
    return _format_date(value.year, value.month, value.day)


def _normalize_date_string(text: str) -> Optional[str]:
    wide_match = _WIDE_DATE_TEXT_RE.fullmatch(text)
    if wide_match is not None:
        year = int(wide_match.group("year"))
        month = int(wide_match.group("month"))
        day = int(wide_match.group("day"))
        if not 1 <= month <= 12 or not 1 <= day <= _days_in_month(year, month):
            return None
        return _format_arbitrary_date(year, month, day)
    for pattern in (
        _DATE_YMD_RE,
        _DATE_COMPACT_YMD_RE,
        _DATE_YM_RE,
        _DATE_COMPACT_YM_RE,
        _DATE_WEEK_RE,
        _DATE_COMPACT_WEEK_RE,
        _DATE_ORDINAL_RE,
        _DATE_COMPACT_ORDINAL_RE,
        _DATE_YEAR_RE,
    ):
        match = pattern.fullmatch(text)
        if match is None:
            continue
        groups = match.groupdict()
        if "month" in groups and "day" in groups:
            return _normalize_date_map(
                {
                    "year": groups["year"],
                    "month": groups["month"],
                    "day": groups["day"],
                }
            )
        if "month" in groups:
            return _normalize_date_map({"year": groups["year"], "month": groups["month"]})
        if "week" in groups:
            fields = {"year": groups["year"], "week": groups["week"]}
            if groups.get("dow") is not None:
                fields["dayOfWeek"] = groups["dow"]
            return _normalize_date_map(fields)
        if "ordinal" in groups:
            return _normalize_date_map({"year": groups["year"], "ordinalDay": groups["ordinal"]})
        return _normalize_date_map({"year": groups["year"]})
    return None


def _normalize_localtime_string(text: str) -> Optional[str]:
    match = _LOCALTIME_RE.fullmatch(text)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or "0")
    second = match.group("second")
    frac = match.group("frac")
    out = f"{hour:02d}:{minute:02d}"
    if second is not None or frac is not None:
        out += f":{int(second or '0'):02d}"
    if frac is not None:
        out += _normalize_fraction(int(frac.ljust(9, "0")[:9]))
    return out


def _normalize_localtime_map(
    fields: dict[str, str],
    *,
    base: Optional[dict[str, object]] = None,
) -> Optional[str]:
    if base is None:
        base = _base_time_parts_from_text(_first_present_field_text(fields, "time", "datetime", "localdatetime"))
    hour = _parse_int(fields.get("hour"), _base_time_int(base, "hour", 0))
    minute = _parse_int(fields.get("minute"), _base_time_int(base, "minute", 0))
    second_default = (
        _base_time_int(base, "second", 0)
        if base is not None and bool(base.get("has_second"))
        else None
    )
    second = _parse_int(fields.get("second"), second_default)
    nanos = _nanos_from_fields(fields) if any(k in fields for k in ("nanosecond", "microsecond", "millisecond")) else (
        _base_time_int(base, "nanosecond", 0)
    )
    if hour is None or minute is None:
        return None
    out = f"{hour:02d}:{minute:02d}"
    if second is not None or nanos:
        out += f":{int(second or 0):02d}{_normalize_fraction(nanos)}"
    return out


def _normalize_time_string(text: str) -> Optional[str]:
    match = _TIME_RE.fullmatch(text)
    if match is None:
        local = _normalize_localtime_string(text)
        return None if local is None else local + "Z"
    local = _normalize_localtime_string(match.group("local"))
    if local is None:
        return None
    return local + _normalize_offset_text(match.group("tz"))


def _source_temporal_value(fields: dict[str, str]) -> Optional[_TemporalValue]:
    source_text = _first_present_field_text(fields, "time", "datetime", "localdatetime")
    if source_text is None:
        return None
    return _parse_temporal_value(source_text)


def _resolve_timezone_target(
    timezone_text: str,
) -> Optional[tuple[object, Optional[str], str]]:
    zone_name = _parse_quoted(timezone_text)
    target_text = zone_name if zone_name is not None else timezone_text
    normalized_offset = _normalize_offset_text(target_text)
    if re.fullmatch(r"Z|[+-]\d{2}:\d{2}(?::\d{2})?", normalized_offset):
        if normalized_offset == "Z":
            return py_timezone.utc, None, "Z"
        offset_delta = py_timedelta_from_offset(normalized_offset)
        if offset_delta is None:
            return None
        return py_timezone(offset_delta), None, normalized_offset
    zone = _named_timezone(target_text)
    if zone is None:
        return None
    return zone, target_text, target_text


def _render_explicit_timezone_suffix(
    timezone_text: str,
    local_datetime_text: str,
    *,
    keep_zone_name: bool,
) -> Optional[str]:
    resolved = _resolve_timezone_target(timezone_text)
    if resolved is None:
        return None
    _, zone_name, fixed_suffix = resolved
    if zone_name is None:
        return fixed_suffix
    suffix = _zone_suffix(zone_name, local_datetime_text)
    if suffix is None:
        return None
    if keep_zone_name:
        return suffix
    return suffix.split("[", 1)[0]


def _preserve_source_timezone_suffix(
    source_value: Optional[_TemporalValue],
    local_datetime_text: str,
    *,
    keep_zone_name: bool,
) -> Optional[str]:
    if source_value is None or source_value.tz_suffix is None:
        return None
    offset, zone_name = _split_tz_suffix_parts(source_value.tz_suffix)
    if zone_name is None:
        return offset or "Z"
    suffix = _zone_suffix(zone_name, local_datetime_text)
    if suffix is None:
        return None
    if keep_zone_name:
        return suffix
    return suffix.split("[", 1)[0]


def _converted_aware_source_base(
    source_value: Optional[_TemporalValue],
    source_base: Optional[dict[str, object]],
    timezone_text: str,
    *,
    target_date: Optional[py_date],
) -> tuple[Optional[dict[str, object]], Optional[py_date]]:
    if source_value is None or source_value.tz_suffix is None:
        return source_base, target_date
    resolved = _resolve_timezone_target(timezone_text)
    if resolved is None:
        return source_base, target_date
    target_tzinfo, _, _ = resolved
    anchor_date = target_date or source_value.date_value or py_date(1970, 1, 1)
    conversion_source = source_value
    if target_date is not None and source_value.date_value is not None:
        conversion_source = _TemporalValue(
            kind=source_value.kind,
            date_value=target_date,
            hour=source_value.hour,
            minute=source_value.minute,
            second=source_value.second,
            nanosecond=source_value.nanosecond,
            tz_suffix=source_value.tz_suffix,
        )
    source_dt = _comparable_datetime(
        conversion_source,
        include_date=True,
        keep_timezone=True,
        anchor_date=anchor_date,
    )
    converted_dt = source_dt.astimezone(cast(py_tzinfo, target_tzinfo))
    converted_base: dict[str, object] = {} if source_base is None else dict(source_base)
    converted_base["hour"] = converted_dt.hour
    converted_base["minute"] = converted_dt.minute
    converted_base["second"] = converted_dt.second
    converted_base["nanosecond"] = source_value.nanosecond
    converted_base["tz"] = _normalize_offset_text(
        "Z" if converted_dt.utcoffset() in {None, timedelta(0)} else _format_offset(cast(timedelta, converted_dt.utcoffset()))
    )
    converted_base["has_second"] = bool(converted_base.get("has_second")) or converted_dt.second != 0
    converted_base["has_fraction"] = bool(converted_base.get("has_fraction")) or source_value.nanosecond != 0
    return converted_base, converted_dt.date()


def _normalize_time_map(fields: dict[str, str]) -> Optional[str]:
    source_value = _source_temporal_value(fields)
    source_base = _base_time_parts_from_text(_first_present_field_text(fields, "time", "datetime", "localdatetime"))
    tz_text = fields.get("timezone")
    effective_date = source_value.date_value if source_value is not None else py_date(1970, 1, 1)
    effective_base = source_base
    if tz_text is not None:
        effective_base, effective_date = _converted_aware_source_base(
            source_value,
            source_base,
            tz_text,
            target_date=effective_date,
        )
    local = _normalize_localtime_map(fields, base=effective_base)
    if local is None:
        return None
    local_datetime_text = _zone_compatible_local_datetime_text(effective_date or py_date(1970, 1, 1), local)
    if tz_text is None:
        preserved_suffix = _preserve_source_timezone_suffix(
            source_value,
            local_datetime_text,
            keep_zone_name=False,
        )
        if preserved_suffix is not None:
            return local + preserved_suffix
        base_tz = None if effective_base is None else cast(Optional[str], effective_base.get("tz"))
        return local + (base_tz or "Z")
    explicit_suffix = _render_explicit_timezone_suffix(
        tz_text,
        local_datetime_text,
        keep_zone_name=False,
    )
    if explicit_suffix is None:
        return None
    return local + explicit_suffix


def _normalize_localdatetime_string(text: str) -> Optional[str]:
    if "[" in text or text.endswith("Z") or re.search(r"[+-]\d{2}(?::?\d{2})?(?::?\d{2})?$", text):
        return None
    if "T" not in text:
        return None
    date_txt, time_txt = text.split("T", 1)
    date_part = _normalize_date_string(date_txt)
    time_part = _normalize_localtime_string(time_txt)
    if date_part is None or time_part is None:
        return None
    return f"{date_part}T{time_part}"


def _normalize_localdatetime_map(fields: dict[str, str]) -> Optional[str]:
    date_part = _normalize_date_map(fields)
    time_part = _normalize_localtime_map(fields)
    if date_part is None or time_part is None:
        return None
    return f"{date_part}T{time_part}"


def _zone_suffix(zone_name: str, local_text: str) -> Optional[str]:
    try:
        local_dt = py_datetime.fromisoformat(local_text)
    except ValueError:
        return None
    historical_override = _neo4j_historical_zone_offset(zone_name, local_dt)
    if historical_override is not None:
        return f"{_format_offset(historical_override)}[{zone_name}]"
    zone = _named_timezone(zone_name)
    if zone is None:
        return None
    delta = local_dt.replace(tzinfo=zone).utcoffset()
    if delta is None:
        return None
    return f"{_format_offset(delta)}[{zone_name}]"


def _normalize_datetime_string(text: str) -> Optional[str]:
    zone: Optional[str] = None
    core = text
    zone_match = re.fullmatch(r"(?P<core>.+)\[(?P<zone>[^\]]+)\]$", core)
    if zone_match is not None:
        core = zone_match.group("core")
        zone = zone_match.group("zone")
    tz: Optional[str] = None
    tz_match = _DATETIME_TZ_RE.fullmatch(core)
    if tz_match is not None:
        core = tz_match.group("core")
        tz = tz_match.group("tz")
    if "T" not in core:
        return None
    date_txt, time_txt = core.split("T", 1)
    date_part = _normalize_date_string(date_txt)
    time_part = _normalize_localtime_string(time_txt)
    if date_part is None or time_part is None:
        return None
    out = f"{date_part}T{time_part}"
    if zone is not None:
        if tz is not None:
            return out + _normalize_offset_text(tz) + f"[{zone}]"
        suffix = _zone_suffix(zone, out)
        return out + suffix if suffix is not None else None
    if tz is not None:
        return out + _normalize_offset_text(tz)
    return out + "Z"


def _normalize_datetime_map(fields: dict[str, str]) -> Optional[str]:
    source_value = _source_temporal_value(fields)
    source_base = _base_time_parts_from_text(_first_present_field_text(fields, "time", "datetime", "localdatetime"))
    tz_text = fields.get("timezone")
    source_date_base = _base_date_from_text(_first_present_field_text(fields, "date", "datetime", "localdatetime"))
    effective_base_date = source_date_base or (source_value.date_value if source_value is not None else None)
    effective_time_base = source_base
    explicit_date_controls = any(
        key in fields
        for key in (
            "date",
            "year",
            "month",
            "day",
            "week",
            "dayOfWeek",
            "ordinalDay",
            "quarter",
            "dayOfQuarter",
        )
    )
    date_part: Optional[str]
    if explicit_date_controls:
        date_part = _normalize_date_map(fields, base_date=effective_base_date)
        if date_part is None:
            return None
        target_date = _base_date_from_text(date_part)
        if tz_text is not None:
            effective_time_base, _ = _converted_aware_source_base(
                source_value,
                source_base,
                tz_text,
                target_date=target_date,
            )
    else:
        if tz_text is not None:
            effective_time_base, converted_base_date = _converted_aware_source_base(
                source_value,
                source_base,
                tz_text,
                target_date=effective_base_date,
            )
            if source_date_base is None and converted_base_date is not None:
                effective_base_date = converted_base_date
        date_part = _normalize_date_map(fields, base_date=effective_base_date)
    time_part = _normalize_localtime_map(fields, base=effective_time_base)
    if date_part is None or time_part is None:
        return None
    out = f"{date_part}T{time_part}"
    if tz_text is None:
        preserved_suffix = _preserve_source_timezone_suffix(
            source_value,
            out,
            keep_zone_name=True,
        )
        if preserved_suffix is not None:
            return out + preserved_suffix
        base_tz = None if effective_time_base is None else cast(Optional[str], effective_time_base.get("tz"))
        return out + (base_tz or "Z")
    explicit_suffix = _render_explicit_timezone_suffix(
        tz_text,
        out,
        keep_zone_name=True,
    )
    if explicit_suffix is None:
        return None
    return out + explicit_suffix


def _normalize_duration_string(text: str) -> Optional[str]:
    stripped = text.strip()
    datetime_match = re.fullmatch(
        r"(?P<sign>-)?P"
        r"(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})"
        r"T(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2})(?:\.(?P<frac>\d+))?",
        stripped,
    )
    if datetime_match is not None:
        sign = "-" if datetime_match.group("sign") else ""
        datetime_fields = {
            "years": sign + datetime_match.group("year"),
            "months": sign + str(int(datetime_match.group("month"))),
            "days": sign + str(int(datetime_match.group("day"))),
            "hours": sign + str(int(datetime_match.group("hour"))),
            "minutes": sign + str(int(datetime_match.group("minute"))),
            "seconds": sign
            + (
                datetime_match.group("second")
                if datetime_match.group("frac") is None
                else f"{int(datetime_match.group('second'))}.{datetime_match.group('frac')}"
            ),
        }
        return _normalize_duration_map(datetime_fields)

    prefix_match = re.fullmatch(r"(?P<sign>-)?P(?P<body>.*)", stripped)
    if prefix_match is None:
        return None
    body = prefix_match.group("body")
    if body == "":
        return None
    sign = "-" if prefix_match.group("sign") else ""
    if "T" in body:
        date_part, time_part = body.split("T", 1)
    else:
        date_part, time_part = body, ""

    def _consume(part: str, allowed_units: set[str]) -> Optional[list[tuple[str, str]]]:
        if part == "":
            return []
        pos = 0
        out: list[tuple[str, str]] = []
        for token_match in _DURATION_TOKEN_RE.finditer(part):
            if token_match.start() != pos:
                return None
            value_txt, unit = token_match.groups()
            if unit not in allowed_units:
                return None
            out.append((value_txt, unit))
            pos = token_match.end()
        if pos != len(part):
            return None
        return out

    date_tokens = _consume(date_part, {"Y", "M", "W", "D"})
    time_tokens = _consume(time_part, {"H", "M", "S"})
    if date_tokens is None or time_tokens is None:
        return None

    component_fields: dict[str, str] = {}
    date_key_map = {"Y": "years", "M": "months", "W": "weeks", "D": "days"}
    time_key_map = {"H": "hours", "M": "minutes", "S": "seconds"}
    for value_txt, unit in date_tokens:
        component_fields[date_key_map[unit]] = sign + value_txt
    for value_txt, unit in time_tokens:
        component_fields[time_key_map[unit]] = sign + value_txt
    return _normalize_duration_map(component_fields)


def _format_signed_day_time_duration(total_nanoseconds: int) -> str:
    if total_nanoseconds == 0:
        return "PT0S"
    sign = -1 if total_nanoseconds < 0 else 1
    remaining = abs(total_nanoseconds)
    days, remaining = divmod(remaining, 24 * 60 * 60 * 1_000_000_000)
    hours, remaining = divmod(remaining, 60 * 60 * 1_000_000_000)
    minutes, remaining = divmod(remaining, 60 * 1_000_000_000)
    seconds, nanoseconds = divmod(remaining, 1_000_000_000)

    def _signed(value: int) -> str:
        return f"{'-' if sign < 0 else ''}{value}"

    parts = ["P"]
    if days:
        parts.append(f"{_signed(days)}D")
    time_parts: list[str] = []
    if hours:
        time_parts.append(f"{_signed(hours)}H")
    if minutes:
        time_parts.append(f"{_signed(minutes)}M")
    if seconds or nanoseconds or (not days and not time_parts):
        if nanoseconds:
            frac = str(nanoseconds).rjust(9, "0").rstrip("0")
            time_parts.append(f"{_signed(seconds)}.{frac}S")
        else:
            time_parts.append(f"{_signed(seconds)}S")
    if time_parts:
        parts.append("T")
        parts.extend(time_parts)
    return "".join(parts)


def _normalize_duration_map(fields: dict[str, str]) -> str:
    # openCypher CIP `Duration`: months, days, seconds+nanoseconds are kept as
    # separate components (months have variable length; days vary under DST).
    # Within a duration, fractional larger units cascade into the next group:
    # fractional months -> days (30.436875 d/mo), fractional days -> seconds.
    # The components are otherwise preserved through toString and equality.
    def _decimal_value(key: str) -> Decimal:
        raw = fields.get(key)
        return Decimal(raw) if raw is not None else Decimal(0)

    months_combined = _decimal_value("years") * 12 + _decimal_value("months")
    months_int = int(months_combined)
    months_frac = months_combined - Decimal(months_int)

    days_combined = (
        _decimal_value("weeks") * 7
        + _decimal_value("days")
        + months_frac * Decimal("30.436875")
    )
    days_int = int(days_combined)
    days_frac = days_combined - Decimal(days_int)

    total_nanoseconds = (
        days_frac * Decimal(24 * 60 * 60 * 1_000_000_000)
        + _decimal_value("hours") * Decimal(60 * 60 * 1_000_000_000)
        + _decimal_value("minutes") * Decimal(60 * 1_000_000_000)
        + _decimal_value("seconds") * Decimal(1_000_000_000)
        + Decimal(_nanos_from_fields(fields))
    )
    nanoseconds_int = int(total_nanoseconds.to_integral_value())

    if months_int == 0 and days_int == 0 and nanoseconds_int == 0:
        return "PT0S"

    if months_int >= 0:
        years_part = months_int // 12
    else:
        years_part = -((-months_int) // 12)
    month_remainder = months_int - years_part * 12

    parts = ["P"]
    if years_part:
        parts.append(f"{years_part}Y")
    if month_remainder:
        parts.append(f"{month_remainder}M")
    if days_int:
        parts.append(f"{days_int}D")

    if nanoseconds_int != 0:
        sign_str = "-" if nanoseconds_int < 0 else ""
        rem = abs(nanoseconds_int)
        hours_part, rem = divmod(rem, 60 * 60 * 1_000_000_000)
        minutes_part, rem = divmod(rem, 60 * 1_000_000_000)
        seconds_part, nanos_part = divmod(rem, 1_000_000_000)
        time_parts: list[str] = []
        if hours_part:
            time_parts.append(f"{sign_str}{hours_part}H")
        if minutes_part:
            time_parts.append(f"{sign_str}{minutes_part}M")
        if seconds_part or nanos_part:
            if nanos_part:
                frac = str(nanos_part).rjust(9, "0").rstrip("0")
                time_parts.append(f"{sign_str}{seconds_part}.{frac}S")
            else:
                time_parts.append(f"{sign_str}{seconds_part}S")
        if time_parts:
            parts.append("T")
            parts.extend(time_parts)

    return "".join(parts)


def _extract_time_text_parts(text: str) -> Optional[tuple[str, Optional[str], Optional[str]]]:
    candidate = text.strip()
    if "T" in candidate:
        candidate = candidate.split("T", 1)[1]
    zone_name: Optional[str] = None
    zone_match = re.fullmatch(r"(?P<core>.+)\[(?P<zone>[^\]]+)\]$", candidate)
    if zone_match is not None:
        candidate = zone_match.group("core")
        zone_name = zone_match.group("zone")
    match = _NORMALIZED_TIME_PART_RE.fullmatch(candidate)
    if match is None:
        return None
    return match.group("local"), match.group("tz"), zone_name


def _cast_temporal_to_localtime_string(text: str) -> Optional[str]:
    normalized = (
        _normalize_datetime_string(text)
        or _normalize_localdatetime_string(text)
        or _normalize_time_string(text)
        or _normalize_localtime_string(text)
    )
    if normalized is None:
        return None
    parts = _extract_time_text_parts(normalized)
    return None if parts is None else parts[0]


def _cast_temporal_to_time_string(text: str) -> Optional[str]:
    normalized = (
        _normalize_datetime_string(text)
        or _normalize_time_string(text)
        or _normalize_localdatetime_string(text)
        or _normalize_localtime_string(text)
    )
    if normalized is None:
        return None
    parts = _extract_time_text_parts(normalized)
    if parts is None:
        return None
    local_text, _, _ = parts
    parsed = _parse_temporal_value(normalized)
    preserved_suffix = _preserve_source_timezone_suffix(
        parsed,
        _zone_compatible_local_datetime_text(parsed.date_value if parsed is not None and parsed.date_value is not None else py_date(1970, 1, 1), local_text),
        keep_zone_name=False,
    )
    return local_text + (preserved_suffix or "Z")


def _cast_temporal_to_localdatetime_string(text: str) -> Optional[str]:
    normalized = _normalize_datetime_string(text) or _normalize_localdatetime_string(text) or _normalize_date_string(text)
    if normalized is None:
        return None
    if "T" not in normalized:
        return normalized + "T00:00"
    date_text, _ = normalized.split("T", 1)
    local_time_text = _cast_temporal_to_localtime_string(normalized)
    if local_time_text is None:
        return None
    return f"{date_text}T{local_time_text}"


def _cast_temporal_to_datetime_string(text: str) -> Optional[str]:
    normalized = _normalize_datetime_string(text) or _normalize_localdatetime_string(text) or _normalize_date_string(text)
    if normalized is None:
        return None
    if "T" not in normalized:
        return normalized + "T00:00Z"
    if _normalize_datetime_string(normalized) is not None:
        return normalized
    return normalized + "Z"


def _current_temporal_literal(fn_name: str, current_dt: py_datetime) -> Optional[str]:
    local_dt = current_dt.astimezone()
    local_time_text = _format_localtime_parts(
        local_dt.hour,
        local_dt.minute,
        local_dt.second,
        local_dt.microsecond * 1_000,
    )
    if fn_name == "date":
        return _format_date(local_dt.year, local_dt.month, local_dt.day)
    if fn_name == "localtime":
        return local_time_text
    if fn_name == "time":
        offset = local_dt.utcoffset()
        suffix = "Z" if offset is None or offset == timedelta(0) else _format_offset(offset)
        return local_time_text + suffix
    local_datetime_text = _format_localdatetime_parts(
        local_dt.date(),
        local_dt.hour,
        local_dt.minute,
        local_dt.second,
        local_dt.microsecond * 1_000,
    )
    if fn_name == "localdatetime":
        return local_datetime_text
    if fn_name == "datetime":
        offset = local_dt.utcoffset()
        suffix = "Z" if offset is None or offset == timedelta(0) else _format_offset(offset)
        return local_datetime_text + suffix
    return None


def normalize_temporal_constructor_text(text: str) -> Optional[str]:
    stripped = text.strip()
    match = _TEMPORAL_FUNC_RE.fullmatch(stripped)
    if match is None:
        return None
    fn = match.group("fn")
    arg_text = match.group("arg").strip()

    if arg_text.startswith("{") and arg_text.endswith("}"):
        fields = _parse_map_fields(arg_text)
        if fields is None:
            return None
        if fn == "date":
            return _normalize_date_map(fields)
        if fn == "localtime":
            return _normalize_localtime_map(fields)
        if fn == "time":
            return _normalize_time_map(fields)
        if fn == "localdatetime":
            return _normalize_localdatetime_map(fields)
        if fn == "datetime":
            return _normalize_datetime_map(fields)
        if fn == "duration":
            return _normalize_duration_map(fields)
        return None

    literal = _parse_quoted(arg_text)
    if literal is None:
        return None
    if fn == "date":
        return _extract_date_text(literal)
    if fn == "localtime":
        return _cast_temporal_to_localtime_string(literal)
    if fn == "time":
        return _cast_temporal_to_time_string(literal)
    if fn == "localdatetime":
        return _cast_temporal_to_localdatetime_string(literal)
    if fn == "datetime":
        return _cast_temporal_to_datetime_string(literal)
    if fn == "duration":
        return _normalize_duration_string(literal)
    return None


from graphistry.compute.gfql.temporal.values import (  # noqa: E402
    _TemporalValue,
    _comparable_datetime,
    _days_in_month,
    _format_localdatetime_parts,
    _format_localtime_parts,
    _parse_temporal_value,
    _split_zone_name,
    py_timedelta_from_offset,
)
from graphistry.compute.gfql.temporal.durations import _DURATION_TOKEN_RE  # noqa: E402
from graphistry.compute.gfql.temporal.truncation import _zone_compatible_local_datetime_text  # noqa: E402
