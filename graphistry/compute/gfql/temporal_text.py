from __future__ import annotations

from datetime import date as py_date
from datetime import datetime as py_datetime
from datetime import timedelta
import re
from typing import Optional
from zoneinfo import ZoneInfo


_TEMPORAL_FUNC_RE = re.compile(
    r"^(?P<fn>date|localtime|time|localdatetime|datetime|duration)\((?P<arg>.*)\)$"
)
TEMPORAL_CALL_EXPR_RE = re.compile(
    r"(?:localdatetime|localtime|datetime|time|date|duration)\((?:\{[^()]*\}|'[^']*')\)"
)
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


def _parse_int(value: Optional[str], default: Optional[int] = None) -> Optional[int]:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _nanos_from_fields(fields: dict[str, str]) -> int:
    if "nanosecond" in fields:
        return int(fields["nanosecond"])
    if "microsecond" in fields:
        return int(fields["microsecond"]) * 1_000
    if "millisecond" in fields:
        return int(fields["millisecond"]) * 1_000_000
    return 0


def _normalize_fraction(nanos: int) -> str:
    if nanos == 0:
        return ""
    frac = str(nanos).zfill(9).rstrip("0")
    return f".{frac}" if frac else ""


def _normalize_offset_text(tz_text: str) -> str:
    if tz_text == "Z":
        return "Z"
    match = re.fullmatch(r"([+-])(\d{2})(?::?(\d{2}))?(?::?(\d{2}))?", tz_text)
    if match is None:
        return tz_text
    sign, hour, minute, second = match.groups()
    minute = minute or "00"
    second = second or "00"
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


def _date_from_fields(fields: dict[str, str]) -> Optional[py_date]:
    year = _parse_int(fields.get("year"))
    if year is None:
        return None
    if "month" in fields or "day" in fields:
        month = _parse_int(fields.get("month"), 1)
        day = _parse_int(fields.get("day"), 1)
        if month is None or day is None:
            return None
        return py_date(year, month, day)
    if "week" in fields:
        week = _parse_int(fields.get("week"))
        day_of_week = _parse_int(fields.get("dayOfWeek"), 1)
        if week is None or day_of_week is None:
            return None
        return py_date.fromisocalendar(year, week, day_of_week)
    if "ordinalDay" in fields:
        ordinal = _parse_int(fields.get("ordinalDay"))
        if ordinal is None:
            return None
        return py_date(year, 1, 1) + timedelta(days=ordinal - 1)
    if "quarter" in fields:
        quarter = _parse_int(fields.get("quarter"))
        day_of_quarter = _parse_int(fields.get("dayOfQuarter"), 1)
        if quarter is None or day_of_quarter is None:
            return None
        start = py_date(year, ((quarter - 1) * 3) + 1, 1)
        return start + timedelta(days=day_of_quarter - 1)
    return py_date(year, 1, 1)


def _normalize_date_map(fields: dict[str, str]) -> Optional[str]:
    value = _date_from_fields(fields)
    if value is None:
        return None
    return _format_date(value.year, value.month, value.day)


def _normalize_date_string(text: str) -> Optional[str]:
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
    hour = _parse_int(fields.get("hour"), 0)
    minute = _parse_int(fields.get("minute"), 0)
    second = _parse_int(fields.get("second"))
    nanos = _nanos_from_fields(fields)
    if hour is None or minute is None:
        return None
    out = f"{hour:02d}:{minute:02d}"
    if second is not None or nanos:
        out += f":{int(second or 0):02d}{_normalize_fraction(nanos)}"
    return out


def _normalize_time_string(text: str) -> Optional[str]:
    match = _TIME_RE.fullmatch(text)
    if match is None:
        return None
    local = _normalize_localtime_string(match.group("local"))
    if local is None:
        return None
    return local + _normalize_offset_text(match.group("tz"))


def _normalize_time_map(fields: dict[str, str]) -> Optional[str]:
    tz_text = fields.get("timezone")
    if tz_text is None:
        return None
    local = _normalize_localtime_map(fields)
    if local is None:
        return None
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
    return out


def _normalize_datetime_map(fields: dict[str, str]) -> Optional[str]:
    date_part = _normalize_date_map(fields)
    time_part = _normalize_localtime_map(fields)
    if date_part is None or time_part is None:
        return None
    out = f"{date_part}T{time_part}"
    tz_text = fields.get("timezone")
    if tz_text is None:
        return out
    zone_name = _parse_quoted(tz_text)
    if zone_name is not None:
        suffix = _zone_suffix(zone_name, out)
        return out + suffix if suffix is not None else None
    return out + _normalize_offset_text(tz_text)


def _normalize_duration_string(text: str) -> Optional[str]:
    return text if text.startswith("P") or text.startswith("-P") else None


def _normalize_duration_map(fields: dict[str, str]) -> str:
    years = _parse_int(fields.get("years"), 0) or 0
    months = _parse_int(fields.get("months"), 0) or 0
    days = _parse_int(fields.get("days"), 0) or 0
    hours = _parse_int(fields.get("hours"), 0) or 0
    minutes = _parse_int(fields.get("minutes"), 0) or 0
    seconds = _parse_int(fields.get("seconds"), 0) or 0
    nanos = _nanos_from_fields(fields)

    minutes += seconds // 60
    seconds = seconds % 60
    hours += minutes // 60
    minutes = minutes % 60
    days += hours // 24
    hours = hours % 24

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
    if seconds or nanos:
        sec_text = f"{seconds}{_normalize_fraction(nanos)}S"
        time_parts.append(sec_text)
    if time_parts:
        parts.append("T")
        parts.extend(time_parts)
    if parts == ["P"]:
        return "PT0S"
    return "".join(parts)


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
        return _normalize_date_string(literal)
    if fn == "localtime":
        return _normalize_localtime_string(literal)
    if fn == "time":
        return _normalize_time_string(literal)
    if fn == "localdatetime":
        return _normalize_localdatetime_string(literal)
    if fn == "datetime":
        return _normalize_datetime_string(literal)
    if fn == "duration":
        return _normalize_duration_string(literal)
    return None


def rewrite_temporal_constructors_in_expr(expr_text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        normalized = normalize_temporal_constructor_text(match.group(0))
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    return TEMPORAL_CALL_EXPR_RE.sub(_replace, expr_text)
