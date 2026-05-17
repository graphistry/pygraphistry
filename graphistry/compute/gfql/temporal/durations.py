from __future__ import annotations

from decimal import Decimal
from datetime import datetime as py_datetime
from datetime import timedelta
from datetime import timezone as py_timezone
import re
from typing import Optional, cast

from graphistry.compute.gfql import temporal_text as _tt
from graphistry.compute.gfql.expr_parser import ExprNode, Literal
from graphistry.compute.gfql.temporal.values import (
    _TemporalValue,
    _comparable_datetime,
    _days_from_civil,
    _days_in_month,
    _parse_temporal_value,
    _parse_wide_temporal_value,
)

_DURATION_TOKEN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)([YMWDHMS])")
_DAY_TIME_DURATION_TOKEN_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)([WDHMS])")

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
        return _tt._format_signed_day_time_duration(_timedelta_total_microseconds(delta) * 1_000)
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
