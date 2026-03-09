from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import timedelta
from datetime import timezone as py_timezone
import re
from typing import Optional, cast
from zoneinfo import ZoneInfo

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


def _date_from_fields(fields: dict[str, str]) -> Optional[py_date]:
    base = _base_date_from_text(_first_present_field_text(fields, "date", "datetime", "localdatetime"))
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
        iso_year_default = base.isocalendar().year if base is not None else None
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


def _normalize_date_map(fields: dict[str, str]) -> Optional[str]:
    value = _date_from_fields(fields)
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


def _normalize_localtime_map(fields: dict[str, str]) -> Optional[str]:
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


def _normalize_time_map(fields: dict[str, str]) -> Optional[str]:
    base = _base_time_parts_from_text(_first_present_field_text(fields, "time", "datetime", "localdatetime"))
    tz_text = fields.get("timezone")
    local = _normalize_localtime_map(fields)
    if local is None:
        return None
    if tz_text is None:
        base_tz = None if base is None else cast(Optional[str], base.get("tz"))
        return local + (base_tz or "Z")
    zone = _parse_quoted(tz_text)
    if zone is None:
        return local + _normalize_offset_text(tz_text)
    if re.fullmatch(r"Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?", zone):
        return local + _normalize_offset_text(zone)
    return local + f"[{zone}]"


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
        zone = ZoneInfo(zone_name)
    except Exception:
        return None
    try:
        local_dt = py_datetime.fromisoformat(local_text)
    except ValueError:
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
    date_part = _normalize_date_map(fields)
    time_part = _normalize_localtime_map(fields)
    if date_part is None or time_part is None:
        return None
    out = f"{date_part}T{time_part}"
    tz_text = fields.get("timezone")
    if tz_text is None:
        base = _base_time_parts_from_text(_first_present_field_text(fields, "time", "datetime"))
        base_tz = None if base is None else cast(Optional[str], base.get("tz"))
        return out + (base_tz or "Z")
    zone_name = _parse_quoted(tz_text)
    if zone_name is not None:
        if re.fullmatch(r"Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?", zone_name):
            return out + _normalize_offset_text(zone_name)
        suffix = _zone_suffix(zone_name, out)
        return out + suffix if suffix is not None else None
    return out + _normalize_offset_text(tz_text)


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
    def _decimal_value(key: str) -> Decimal:
        raw = fields.get(key)
        return Decimal(raw) if raw is not None else Decimal(0)

    years_total = _decimal_value("years")
    months_total = _decimal_value("months")
    years = int(years_total)
    months = int(months_total)
    total_nanoseconds = (
        (years_total - Decimal(years)) * Decimal("365.2425") * Decimal(24 * 60 * 60 * 1_000_000_000)
        + (months_total - Decimal(months)) * Decimal("30.436875") * Decimal(24 * 60 * 60 * 1_000_000_000)
        + _decimal_value("weeks") * Decimal(7 * 24 * 60 * 60 * 1_000_000_000)
        + _decimal_value("days") * Decimal(24 * 60 * 60 * 1_000_000_000)
        + _decimal_value("hours") * Decimal(60 * 60 * 1_000_000_000)
        + _decimal_value("minutes") * Decimal(60 * 1_000_000_000)
        + _decimal_value("seconds") * Decimal(1_000_000_000)
        + Decimal(_nanos_from_fields(fields))
    )
    day_time_text = _format_signed_day_time_duration(int(total_nanoseconds.to_integral_value()))

    parts = ["P"]
    if years:
        parts.append(f"{years}Y")
    if months:
        parts.append(f"{months}M")
    if day_time_text != "PT0S":
        parts.append(day_time_text[1:])
    if parts == ["P"]:
        return "PT0S"
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
    local_text, tz_text, zone_name = parts
    if tz_text is None and zone_name is None:
        return local_text + "Z"
    suffix = (tz_text or "") + (f"[{zone_name}]" if zone_name is not None else "")
    return local_text + suffix


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
_DURATION_TOKEN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)([YMWDHMS])")
_DATE_TRUNCATION_UNITS = frozenset({"millennium", "century", "decade", "year", "weekYear", "quarter", "month", "week", "day"})
_DAY_TIME_DURATION_TOKEN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)([WDHMS])")


def _format_localtime_parts(hour: int, minute: int, second: int, nanosecond: int) -> str:
    out = f"{hour:02d}:{minute:02d}"
    if second != 0 or nanosecond != 0:
        out += f":{second:02d}{_normalize_fraction(nanosecond)}"
    return out


def _format_localdatetime_parts(
    value_date: py_date,
    hour: int,
    minute: int,
    second: int,
    nanosecond: int,
) -> str:
    return f"{_format_date(value_date.year, value_date.month, value_date.day)}T{_format_localtime_parts(hour, minute, second, nanosecond)}"


def parse_temporal_sort_duration_components(text: str) -> Optional[tuple[int, int]]:
    stripped = text.strip()
    prefix_sign = 1
    if stripped.startswith("-P"):
        prefix_sign = -1
        body = stripped[2:]
    elif stripped.startswith("P"):
        body = stripped[1:]
    else:
        return None
    if body == "":
        return None

    if "T" in body:
        date_part, time_part = body.split("T", 1)
    else:
        date_part, time_part = body, ""

    month_shift = 0
    total = Decimal(0)

    def _consume(part: str, allowed_units: set[str]) -> Optional[list[tuple[Decimal, str]]]:
        if part == "":
            return []
        pos = 0
        out: list[tuple[Decimal, str]] = []
        for match in _DAY_TIME_DURATION_TOKEN_RE.finditer(part):
            if match.start() != pos:
                return None
            value_txt, unit = match.groups()
            if unit not in allowed_units:
                return None
            out.append((Decimal(value_txt), unit))
            pos = match.end()
        if pos != len(part):
            return None
        return out

    date_tokens = _consume(date_part, {"Y", "M", "W", "D"})
    time_tokens = _consume(time_part, {"H", "M", "S"})
    if date_tokens is None or time_tokens is None:
        return None

    for value, unit in date_tokens:
        if unit == "Y":
            if value != int(value):
                return None
            month_shift += int(value) * 12
        elif unit == "M":
            if value != int(value):
                return None
            month_shift += int(value)
        elif unit == "W":
            total += value * Decimal(7 * 24 * 60 * 60 * 1_000_000_000)
        elif unit == "D":
            total += value * Decimal(24 * 60 * 60 * 1_000_000_000)

    for value, unit in time_tokens:
        if unit == "H":
            total += value * Decimal(60 * 60 * 1_000_000_000)
        elif unit == "M":
            total += value * Decimal(60 * 1_000_000_000)
        elif unit == "S":
            total += value * Decimal(1_000_000_000)

    return month_shift * prefix_sign, int((total * prefix_sign).to_integral_value())


def parse_day_time_duration_nanoseconds(text: str) -> Optional[int]:
    parsed = parse_temporal_sort_duration_components(text)
    if parsed is None:
        return None
    month_shift, nanosecond_shift = parsed
    if month_shift != 0:
        return None
    return nanosecond_shift


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
        value_date = _base_date_from_text(date_text)
        if value_date is None:
            return None
        time_core, zone_name = _split_zone_name(time_text)
        parts = _base_time_parts_from_text(time_core)
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
        parts = _base_time_parts_from_text(time_core)
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
    value_date = _base_date_from_text(stripped)
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
    localdatetime_match = _WIDE_LOCALDATETIME_TEXT_RE.fullmatch(stripped)
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
            nanosecond=_parse_fraction_to_nanos(localdatetime_match.group("frac")),
        )
    date_match = _WIDE_DATE_TEXT_RE.fullmatch(stripped)
    if date_match is None:
        return None
    year = int(date_match.group("year"))
    month = int(date_match.group("month"))
    day = int(date_match.group("day"))
    if not 1 <= month <= 12 or not 1 <= day <= _days_in_month(year, month):
        return None
    return _WideTemporalValue(kind="date", year=year, month=month, day=day)


def _format_large_time_only_duration(total_nanoseconds: int) -> str:
    if total_nanoseconds == 0:
        return "PT0S"
    sign = -1 if total_nanoseconds < 0 else 1
    remaining = abs(total_nanoseconds)
    hours, remaining = divmod(remaining, 3_600_000_000_000)
    minutes, remaining = divmod(remaining, 60_000_000_000)
    seconds, nanoseconds = divmod(remaining, 1_000_000_000)

    def _signed(value: int) -> str:
        return f"{'-' if sign < 0 else ''}{value}"

    parts = ["PT"]
    if hours:
        parts.append(f"{_signed(hours)}H")
    if minutes:
        parts.append(f"{_signed(minutes)}M")
    if seconds or nanoseconds or len(parts) == 1:
        if nanoseconds:
            frac = str(nanoseconds).rjust(9, "0").rstrip("0")
            parts.append(f"{_signed(seconds)}.{frac}S")
        else:
            parts.append(f"{_signed(seconds)}S")
    return "".join(parts)


def _fold_large_year_duration_function_call(
    fn_name: str,
    start_text: str,
    end_text: str,
) -> Optional[str]:
    start_value = _parse_wide_temporal_value(start_text)
    end_value = _parse_wide_temporal_value(end_text)
    if start_value is None or end_value is None:
        return None
    if fn_name == "duration.between":
        if start_value.kind != "date" or end_value.kind != "date":
            return None
        if (end_value.year, end_value.month, end_value.day) < (start_value.year, start_value.month, start_value.day):
            return None
        years = end_value.year - start_value.year
        months = end_value.month - start_value.month
        days = end_value.day - start_value.day
        if days < 0:
            months -= 1
            prev_year = end_value.year
            prev_month = end_value.month - 1
            if prev_month == 0:
                prev_year -= 1
                prev_month = 12
            days += _days_in_month(prev_year, prev_month)
        if months < 0:
            years -= 1
            months += 12
        return _format_duration_components(years=years, months=months, days=days)
    if fn_name == "duration.inseconds":
        if start_value.kind != "localdatetime" or end_value.kind != "localdatetime":
            return None
        start_days = _days_from_civil(start_value.year, start_value.month, start_value.day)
        end_days = _days_from_civil(end_value.year, end_value.month, end_value.day)
        start_total = (
            start_days * 86_400_000_000_000
            + start_value.hour * 3_600_000_000_000
            + start_value.minute * 60_000_000_000
            + start_value.second * 1_000_000_000
            + start_value.nanosecond
        )
        end_total = (
            end_days * 86_400_000_000_000
            + end_value.hour * 3_600_000_000_000
            + end_value.minute * 60_000_000_000
            + end_value.second * 1_000_000_000
            + end_value.nanosecond
        )
        return _format_large_time_only_duration(end_total - start_total)
    return None


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
        base = py_date.fromisocalendar(source_date.isocalendar().year, 1, 1)
        if "day" in overrides:
            return py_date(base.year, 1, 1) + timedelta(days=int(overrides["day"]) - 1)
        if "dayOfWeek" in overrides:
            return base + timedelta(days=int(overrides["dayOfWeek"]) - 1)
    elif unit == "day":
        base = source_date
    else:
        return None
    fields = {"date": _format_date(base.year, base.month, base.day)}
    fields.update(overrides)
    return _date_from_fields(fields)


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
        millisecond = _parse_int(
            overrides.get("millisecond", overrides.get("milliseconds")),
            truncated_millisecond,
        )
        microsecond = _parse_int(
            overrides.get("microsecond", overrides.get("microseconds")),
            truncated_microsecond,
        )
        sub_nanosecond = _parse_int(
            overrides.get("nanosecond", overrides.get("nanoseconds")),
            truncated_nanosecond,
        )
        if millisecond is None or microsecond is None or sub_nanosecond is None:
            return hour, minute, second, nanosecond
        nanosecond = (millisecond * 1_000_000) + (microsecond * 1_000) + sub_nanosecond

    return hour, minute, second, nanosecond


def _zone_compatible_local_datetime_text(date_value: py_date, local_time_text: str) -> str:
    local_dt_text = f"{_format_date(date_value.year, date_value.month, date_value.day)}T{local_time_text}"
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
    zone_name = _parse_quoted(timezone_text)
    if zone_name is None and not re.fullmatch(r"Z|[+-]\d{2}(?::?\d{2})?(?::?\d{2})?", timezone_text):
        zone_name = timezone_text
    if zone_name is None:
        return _normalize_offset_text(timezone_text)
    zone_base_date = target_date or source_value.date_value or py_date(1970, 1, 1)
    suffix = _zone_suffix(
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
        parsed = _parse_quoted(rendered)
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
        return Literal(_format_date(target_date.year, target_date.month, target_date.day))

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
    local_text = _format_localdatetime_parts(
        value_date,
        value.hour,
        value.minute,
        value.second,
        value.nanosecond,
    )
    effective_tz_suffix = value.tz_suffix or anchor_tz_suffix
    if keep_timezone and effective_tz_suffix is not None:
        offset = effective_tz_suffix.split("[", 1)[0]
        zone_match = re.search(r"\[(?P<zone>[^\]]+)\]$", effective_tz_suffix)
        if zone_match is not None:
            try:
                return py_datetime.fromisoformat(local_text).replace(tzinfo=ZoneInfo(zone_match.group("zone")))
            except Exception:
                pass
        if offset == "Z":
            return py_datetime.fromisoformat(local_text).replace(tzinfo=py_timezone.utc)
        offset_delta = py_timedelta_from_offset(offset)
        if offset_delta is not None:
            return py_datetime.fromisoformat(local_text).replace(tzinfo=py_timezone(offset_delta))
        return py_datetime.fromisoformat(local_text + offset)
    return py_datetime.fromisoformat(local_text)


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


def _format_time_only_duration(delta: timedelta) -> str:
    total_microseconds = _timedelta_total_microseconds(delta)
    if total_microseconds == 0:
        return "PT0S"
    sign = -1 if total_microseconds < 0 else 1
    remaining = abs(total_microseconds)
    hours, remaining = divmod(remaining, 3_600_000_000)
    minutes, remaining = divmod(remaining, 60_000_000)
    seconds, microseconds = divmod(remaining, 1_000_000)

    def _signed(value: int) -> str:
        return f"{'-' if sign < 0 else ''}{value}"

    parts = ["PT"]
    if hours:
        parts.append(f"{_signed(hours)}H")
    if minutes:
        parts.append(f"{_signed(minutes)}M")
    if seconds or microseconds or len(parts) == 1:
        if microseconds:
            frac = str(microseconds).rjust(6, "0").rstrip("0")
            parts.append(f"{_signed(seconds)}.{frac}S")
        else:
            parts.append(f"{_signed(seconds)}S")
    return "".join(parts)


def _format_duration_components(
    *,
    years: int = 0,
    months: int = 0,
    days: int = 0,
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    microseconds: int = 0,
) -> str:
    parts = ["P"]
    if years:
        parts.append(f"{years}Y")
    if months:
        parts.append(f"{months}M")
    if days:
        parts.append(f"{days}D")
    time_parts: list[str] = []
    if hours:
        time_parts.append(f"{hours}H")
    if minutes:
        time_parts.append(f"{minutes}M")
    if seconds or microseconds:
        if microseconds:
            frac = str(abs(microseconds)).rjust(6, "0").rstrip("0")
            time_parts.append(f"{seconds}.{frac}S")
        else:
            time_parts.append(f"{seconds}S")
    if time_parts:
        parts.append("T")
        parts.extend(time_parts)
    if parts == ["P"]:
        return "PT0S"
    return "".join(parts)


def _fold_duration_between(
    start_value: _TemporalValue,
    end_value: _TemporalValue,
) -> str:
    include_date = start_value.date_value is not None and end_value.date_value is not None
    keep_timezone = start_value.tz_suffix is not None and end_value.tz_suffix is not None
    anchor_date = start_value.date_value or end_value.date_value
    anchor_tz_suffix = start_value.tz_suffix or end_value.tz_suffix
    effective_keep_timezone = keep_timezone or anchor_tz_suffix is not None
    start_dt = _comparable_datetime(
        start_value,
        include_date=include_date,
        keep_timezone=effective_keep_timezone,
        anchor_date=anchor_date,
        anchor_tz_suffix=anchor_tz_suffix if start_value.tz_suffix is None else None,
    )
    end_dt = _comparable_datetime(
        end_value,
        include_date=include_date,
        keep_timezone=effective_keep_timezone,
        anchor_date=anchor_date,
        anchor_tz_suffix=anchor_tz_suffix if end_value.tz_suffix is None else None,
    )
    delta = _absolute_temporal_delta(start_dt, end_dt)
    if not include_date:
        return _format_time_only_duration(delta)

    from dateutil.relativedelta import relativedelta  # type: ignore[import]

    rel = relativedelta(end_dt, start_dt)
    if rel.years == 0 and rel.months == 0:
        return _format_signed_day_time_duration(_timedelta_total_microseconds(delta) * 1_000)
    return _format_duration_components(
        years=rel.years,
        months=rel.months,
        days=rel.days,
        hours=rel.hours,
        minutes=rel.minutes,
        seconds=rel.seconds,
        microseconds=rel.microseconds,
    )


def _fold_duration_in_months(
    start_value: _TemporalValue,
    end_value: _TemporalValue,
) -> str:
    if start_value.date_value is None or end_value.date_value is None:
        return "PT0S"
    keep_timezone = start_value.tz_suffix is not None and end_value.tz_suffix is not None
    start_dt = _comparable_datetime(start_value, include_date=True, keep_timezone=keep_timezone)
    end_dt = _comparable_datetime(end_value, include_date=True, keep_timezone=keep_timezone)
    from dateutil.relativedelta import relativedelta  # type: ignore[import]

    rel = relativedelta(end_dt, start_dt)
    return _format_duration_components(years=rel.years, months=rel.months)


def _fold_duration_in_days(
    start_value: _TemporalValue,
    end_value: _TemporalValue,
) -> str:
    if start_value.date_value is None or end_value.date_value is None:
        return "PT0S"
    keep_timezone = start_value.tz_suffix is not None and end_value.tz_suffix is not None
    start_dt = _comparable_datetime(start_value, include_date=True, keep_timezone=keep_timezone)
    end_dt = _comparable_datetime(end_value, include_date=True, keep_timezone=keep_timezone)
    total_microseconds = _timedelta_total_microseconds(_absolute_temporal_delta(start_dt, end_dt))
    days = int(total_microseconds / (24 * 60 * 60 * 1_000_000))
    return _format_duration_components(days=days)


def _fold_duration_in_seconds(
    start_value: _TemporalValue,
    end_value: _TemporalValue,
) -> str:
    include_date = start_value.date_value is not None and end_value.date_value is not None
    keep_timezone = start_value.tz_suffix is not None and end_value.tz_suffix is not None
    anchor_date = start_value.date_value or end_value.date_value
    anchor_tz_suffix = start_value.tz_suffix or end_value.tz_suffix
    effective_keep_timezone = keep_timezone or anchor_tz_suffix is not None
    start_dt = _comparable_datetime(
        start_value,
        include_date=include_date,
        keep_timezone=effective_keep_timezone,
        anchor_date=anchor_date,
        anchor_tz_suffix=anchor_tz_suffix if start_value.tz_suffix is None else None,
    )
    end_dt = _comparable_datetime(
        end_value,
        include_date=include_date,
        keep_timezone=effective_keep_timezone,
        anchor_date=anchor_date,
        anchor_tz_suffix=anchor_tz_suffix if end_value.tz_suffix is None else None,
    )
    return _format_time_only_duration(_absolute_temporal_delta(start_dt, end_dt))


def _fold_duration_function_call(
    fn_name: str,
    args: tuple[ExprNode, ...],
) -> Optional[Literal]:
    if len(args) != 2:
        return None
    if any(isinstance(arg, Literal) and arg.value is None for arg in args):
        return Literal(None)
    if not all(isinstance(arg, Literal) and isinstance(arg.value, str) for arg in args):
        return None
    start_text = cast(str, cast(Literal, args[0]).value)
    end_text = cast(str, cast(Literal, args[1]).value)
    try:
        start_value = _parse_temporal_value(start_text)
        end_value = _parse_temporal_value(end_text)
    except ValueError:
        start_value = None
        end_value = None
    if start_value is None or end_value is None:
        large_year = _fold_large_year_duration_function_call(fn_name, start_text, end_text)
        if large_year is not None:
            return Literal(large_year)
        return None
    if fn_name == "duration.between":
        return Literal(_fold_duration_between(start_value, end_value))
    if fn_name == "duration.inmonths":
        return Literal(_fold_duration_in_months(start_value, end_value))
    if fn_name == "duration.indays":
        return Literal(_fold_duration_in_days(start_value, end_value))
    if fn_name == "duration.inseconds":
        return Literal(_fold_duration_in_seconds(start_value, end_value))
    return None


def _fold_datetime_epoch_function_call(
    fn_name: str,
    args: tuple[ExprNode, ...],
) -> Optional[Literal]:
    if fn_name not in {"datetime.fromepoch", "datetime.fromepochmillis"}:
        return None
    if any(isinstance(arg, Literal) and arg.value is None for arg in args):
        return Literal(None)
    if not all(isinstance(arg, Literal) and isinstance(arg.value, int) and not isinstance(arg.value, bool) for arg in args):
        return None

    epoch = py_datetime(1970, 1, 1)
    if fn_name == "datetime.fromepochmillis":
        if len(args) != 1:
            return None
        total_nanoseconds = cast(int, cast(Literal, args[0]).value) * 1_000_000
    else:
        if len(args) not in {1, 2}:
            return None
        seconds_value = cast(int, cast(Literal, args[0]).value)
        nanoseconds_value = cast(int, cast(Literal, args[1]).value) if len(args) == 2 else 0
        total_nanoseconds = (seconds_value * 1_000_000_000) + nanoseconds_value

    seconds_part, nanoseconds_part = divmod(total_nanoseconds, 1_000_000_000)
    dt = epoch + timedelta(seconds=seconds_part, microseconds=nanoseconds_part // 1_000)
    rendered = _format_localdatetime_parts(
        dt.date(),
        dt.hour,
        dt.minute,
        dt.second,
        int(nanoseconds_part),
    )
    return Literal(rendered + "Z")


def resolve_duration_text_property(duration_text: str, prop: str) -> Optional[str]:
    if not duration_text.startswith(("P", "-P")):
        return None
    days_value = 0
    time_ns = 0
    body = duration_text[1:] if duration_text.startswith("P") else duration_text[2:]
    date_part, _, time_part = body.partition("T")
    for number_text, unit in _DURATION_TOKEN_RE.findall(date_part):
        if unit == "D":
            days_value = int(Decimal(number_text))
    for number_text, unit in _DURATION_TOKEN_RE.findall(time_part):
        if unit == "H":
            time_ns += int(Decimal(number_text) * Decimal(3_600_000_000_000))
        elif unit == "M":
            time_ns += int(Decimal(number_text) * Decimal(60_000_000_000))
        elif unit == "S":
            time_ns += int(Decimal(number_text) * Decimal(1_000_000_000))
    if prop == "days":
        return str(days_value)
    if prop == "seconds":
        return str(time_ns // 1_000_000_000)
    if prop == "nanosecondsOfSecond":
        seconds_value = time_ns // 1_000_000_000
        return str(time_ns - (seconds_value * 1_000_000_000))
    return None


def rewrite_temporal_constructors_in_expr(expr_text: str) -> str:
    current_dt = py_datetime.now().astimezone()

    def _replace_current(match: re.Match[str]) -> str:
        normalized = _current_temporal_literal(match.group("fn"), current_dt)
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    def _replace(match: re.Match[str]) -> str:
        normalized = normalize_temporal_constructor_text(match.group(0))
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    rewritten = CURRENT_TEMPORAL_CALL_EXPR_RE.sub(_replace_current, expr_text)
    return TEMPORAL_CALL_EXPR_RE.sub(_replace, rewritten)


def _render_temporal_arg(node: ExprNode) -> Optional[str]:
    if isinstance(node, Literal):
        value = node.value
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace("'", "\\'")
            return f"'{escaped}'"
        return None
    if isinstance(node, MapLiteral):
        parts: list[str] = []
        for key, value in node.items:
            rendered = _render_temporal_arg(value)
            if rendered is None:
                return None
            parts.append(f"{key}: {rendered}")
        return "{" + ", ".join(parts) + "}"
    if isinstance(node, ListLiteral):
        rendered_items: list[str] = []
        for item in node.items:
            rendered = _render_temporal_arg(item)
            if rendered is None:
                return None
            rendered_items.append(rendered)
        return "[" + ", ".join(rendered_items) + "]"
    return None


def fold_temporal_constructor_ast(node: ExprNode) -> ExprNode:
    current_dt = py_datetime.now().astimezone()

    def _fold(inner: ExprNode) -> ExprNode:
        if isinstance(inner, (Identifier, Literal, Wildcard)):
            return inner
        if isinstance(inner, UnaryOp):
            return UnaryOp(inner.op, _fold(inner.operand))
        if isinstance(inner, BinaryOp):
            return BinaryOp(
                inner.op,
                _fold(inner.left),
                _fold(inner.right),
            )
        if isinstance(inner, IsNullOp):
            return IsNullOp(_fold(inner.value), negated=inner.negated)
        if isinstance(inner, FunctionCall):
            args = tuple(_fold(arg) for arg in inner.args)
            rewritten = FunctionCall(inner.name, args, distinct=inner.distinct)
            if not inner.distinct and len(args) == 0 and inner.name in {
                "date",
                "localtime",
                "time",
                "localdatetime",
                "datetime",
            }:
                current_literal = _current_temporal_literal(inner.name, current_dt)
                if current_literal is not None:
                    return Literal(current_literal)
            if not inner.distinct and inner.name == "tostring" and len(args) == 1 and isinstance(args[0], Literal):
                value = args[0].value
                if value is None:
                    return Literal(None)
                if isinstance(value, bool):
                    return Literal("true" if value else "false")
                return Literal(str(value))
            if not inner.distinct and len(args) == 1 and inner.name in {
                "date",
                "localtime",
                "time",
                "localdatetime",
                "datetime",
                "duration",
            } and isinstance(args[0], Literal) and args[0].value is None:
                return Literal(None)
            if not inner.distinct and len(args) == 1 and inner.name in {"date", "localtime", "time", "localdatetime", "datetime", "duration"}:
                rendered_arg = _render_temporal_arg(args[0])
                if rendered_arg is not None:
                    normalized = normalize_temporal_constructor_text(f"{inner.name}({rendered_arg})")
                    if normalized is not None:
                        return Literal(normalized)
            if not inner.distinct and inner.name in {
                "date.truncate",
                "localtime.truncate",
                "time.truncate",
                "localdatetime.truncate",
                "datetime.truncate",
            }:
                folded = _fold_temporal_truncate_call(inner.name, args)
                if folded is not None:
                    return folded
            if not inner.distinct and inner.name in {
                "duration.between",
                "duration.inmonths",
                "duration.indays",
                "duration.inseconds",
            }:
                folded = _fold_duration_function_call(inner.name, args)
                if folded is not None:
                    return folded
            if not inner.distinct and inner.name in {
                "datetime.fromepoch",
                "datetime.fromepochmillis",
            }:
                folded = _fold_datetime_epoch_function_call(inner.name, args)
                if folded is not None:
                    return folded
            return rewritten
        if isinstance(inner, CaseWhen):
            return CaseWhen(
                _fold(inner.condition),
                _fold(inner.when_true),
                _fold(inner.when_false),
            )
        if isinstance(inner, QuantifierExpr):
            return QuantifierExpr(
                inner.fn,
                inner.var,
                _fold(inner.source),
                _fold(inner.predicate),
            )
        if isinstance(inner, ListComprehension):
            return ListComprehension(
                inner.var,
                _fold(inner.source),
                predicate=None if inner.predicate is None else _fold(inner.predicate),
                projection=None if inner.projection is None else _fold(inner.projection),
            )
        if isinstance(inner, ListLiteral):
            return ListLiteral(tuple(_fold(item) for item in inner.items))
        if isinstance(inner, MapLiteral):
            return MapLiteral(tuple((key, _fold(value)) for key, value in inner.items))
        if isinstance(inner, SubscriptExpr):
            return SubscriptExpr(
                _fold(inner.value),
                _fold(inner.key),
            )
        if isinstance(inner, SliceExpr):
            return SliceExpr(
                _fold(inner.value),
                None if inner.start is None else _fold(inner.start),
                None if inner.stop is None else _fold(inner.stop),
            )
        return inner

    return _fold(node)
