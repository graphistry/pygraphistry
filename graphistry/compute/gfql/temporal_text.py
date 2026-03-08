from __future__ import annotations

import re
from typing import Optional


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

TEMPORAL_CALL_EXPR_RE = re.compile(
    r"(?:localdatetime|localtime|datetime|time|date)\(\{[^()]*\}\)"
)


def _normalize_fraction(nanos: Optional[str]) -> str:
    if nanos is None:
        return ""
    frac = nanos.zfill(9).rstrip("0")
    return f".{frac}" if frac else ""


def _normalize_time_text(
    *,
    hour: str,
    minute: str,
    second: Optional[str],
    nanos: Optional[str],
    timezone: Optional[str] = None,
) -> str:
    out = f"{int(hour):02d}:{int(minute):02d}"
    if second is not None or nanos is not None:
        out += f":{int(second or '0'):02d}{_normalize_fraction(nanos)}"
    if timezone is not None:
        out += timezone
    return out


def normalize_temporal_constructor_text(text: str) -> Optional[str]:
    stripped = text.strip()

    date_match = DATE_CALL_TEXT_RE.fullmatch(stripped)
    if date_match is not None:
        return (
            f"{int(date_match.group('year')):04d}-"
            f"{int(date_match.group('month')):02d}-"
            f"{int(date_match.group('day')):02d}"
        )

    localtime_match = LOCALTIME_CALL_TEXT_RE.fullmatch(stripped)
    if localtime_match is not None:
        return _normalize_time_text(
            hour=localtime_match.group("hour"),
            minute=localtime_match.group("minute"),
            second=localtime_match.group("second"),
            nanos=localtime_match.group("nano"),
        )

    time_match = TIME_CALL_TEXT_RE.fullmatch(stripped)
    if time_match is not None:
        return _normalize_time_text(
            hour=time_match.group("hour"),
            minute=time_match.group("minute"),
            second=time_match.group("second"),
            nanos=time_match.group("nano"),
            timezone=time_match.group("tz"),
        )

    localdatetime_match = LOCALDATETIME_CALL_TEXT_RE.fullmatch(stripped)
    if localdatetime_match is not None:
        return (
            f"{int(localdatetime_match.group('year')):04d}-"
            f"{int(localdatetime_match.group('month')):02d}-"
            f"{int(localdatetime_match.group('day')):02d}T"
            + _normalize_time_text(
                hour=localdatetime_match.group("hour"),
                minute=localdatetime_match.group("minute"),
                second=localdatetime_match.group("second"),
                nanos=localdatetime_match.group("nano"),
            )
        )

    datetime_match = DATETIME_CALL_TEXT_RE.fullmatch(stripped)
    if datetime_match is not None:
        return (
            f"{int(datetime_match.group('year')):04d}-"
            f"{int(datetime_match.group('month')):02d}-"
            f"{int(datetime_match.group('day')):02d}T"
            + _normalize_time_text(
                hour=datetime_match.group("hour"),
                minute=datetime_match.group("minute"),
                second=datetime_match.group("second"),
                nanos=datetime_match.group("nano"),
                timezone=datetime_match.group("tz"),
            )
        )

    return None


def rewrite_temporal_constructors_in_expr(expr_text: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        normalized = normalize_temporal_constructor_text(match.group(0))
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    return TEMPORAL_CALL_EXPR_RE.sub(_replace, expr_text)
