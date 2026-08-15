from __future__ import annotations

from datetime import datetime as py_datetime
from datetime import timedelta
import re
from typing import Optional, cast

from graphistry.compute.gfql.temporal import constructors as _tt
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    ExprNode,
    FunctionCall,
    Literal,
    _rebuild_expr_node,
)
from graphistry.compute.gfql.temporal.durations import (
    _NANOS_PER_DAY,
    _fold_duration_function_call,
    format_duration_calendar_components,
    parse_duration_calendar_components,
)
from graphistry.compute.gfql.temporal.rendering import _render_temporal_arg
from graphistry.compute.gfql.temporal.truncation import _fold_temporal_truncate_call
from graphistry.compute.gfql.temporal.values import (
    _TemporalValue,
    _days_in_month,
    _format_localdatetime_parts,
    _format_localtime_parts,
    _parse_temporal_value,
)


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


def _shift_temporal_value(value: _TemporalValue, months: int, days: int, time_nanos: int) -> Optional[str]:
    """Apply a duration offset to a temporal value and re-render it as ISO text.

    openCypher: months are applied first with end-of-month clamping, then days, then
    the seconds group. A DATE drops the seconds group entirely (``date + PT25H`` is
    a no-op); a TIME/LOCALTIME has no date to carry into, so it wraps mod 24h.
    """
    if value.kind in {"time", "localtime"}:
        total = (
            value.hour * 3_600_000_000_000
            + value.minute * 60_000_000_000
            + value.second * 1_000_000_000
            + value.nanosecond
            + time_nanos
        )
        total = ((total % _NANOS_PER_DAY) + _NANOS_PER_DAY) % _NANOS_PER_DAY
        hour, rest = divmod(total, 3_600_000_000_000)
        minute, rest = divmod(rest, 60_000_000_000)
        second, nanosecond = divmod(rest, 1_000_000_000)
        rendered = _format_localtime_parts(int(hour), int(minute), int(second), int(nanosecond))
        return rendered + (value.tz_suffix or "") if value.kind == "time" else rendered

    if value.date_value is None:
        return None

    year = value.date_value.year
    month = value.date_value.month
    day = value.date_value.day
    if months:
        total_months = year * 12 + (month - 1) + months
        year, month = divmod(total_months, 12)
        month += 1
        day = min(day, _days_in_month(year, month))
    try:
        shifted_date = py_datetime(year, month, day).date() + timedelta(days=days)
    except ValueError:
        return None

    if value.kind == "date":
        # Date + Duration ignores the duration's seconds group (openCypher).
        return _tt._format_date(shifted_date.year, shifted_date.month, shifted_date.day)

    total_time = (
        value.hour * 3_600_000_000_000
        + value.minute * 60_000_000_000
        + value.second * 1_000_000_000
        + value.nanosecond
        + time_nanos
    )
    day_carry, nanos_of_day = divmod(total_time, _NANOS_PER_DAY)
    shifted_date = shifted_date + timedelta(days=int(day_carry))
    hour, rest = divmod(nanos_of_day, 3_600_000_000_000)
    minute, rest = divmod(rest, 60_000_000_000)
    second, nanosecond = divmod(rest, 1_000_000_000)
    rendered = _format_localdatetime_parts(
        shifted_date, int(hour), int(minute), int(second), int(nanosecond)
    )
    if value.kind == "datetime":
        return rendered + (value.tz_suffix or "")
    return rendered


def _scale_duration(components: tuple[int, int, int], factor: float, divide: bool) -> Optional[str]:
    months, days, time_nanos = components
    scaled_months = (months / factor) if divide else (months * factor)
    if scaled_months != int(scaled_months):
        return None
    total_nanos = days * _NANOS_PER_DAY + time_nanos
    scaled_nanos = (total_nanos / factor) if divide else (total_nanos * factor)
    return format_duration_calendar_components(int(scaled_months), 0, int(round(scaled_nanos)))


def _fold_temporal_arithmetic(node: BinaryOp) -> Optional[Literal]:
    """Constant-fold ``<temporal> ± <duration>``, ``<duration> ± <duration>`` and
    ``<duration> * | / <number>`` over already-lowered ISO literals.

    Temporal literals lower to ISO TEXT before the AST exists, so without this fold
    ``date('2020-01-02') + duration('P1D')`` reached Python ``str + str`` and silently
    produced ``'2020-01-02P1D'`` — including inside WHERE, where the concatenated text
    then changed the row set (#1915 B-2). Only ISO-duration-shaped operands engage, so
    ordinary string concatenation is untouched.
    """
    op = str(node.op).lower()
    if op not in {"+", "-", "*", "/"}:
        return None
    if not (isinstance(node.left, Literal) and isinstance(node.right, Literal)):
        return None
    left_value = node.left.value
    right_value = node.right.value

    def _duration_of(value: object) -> Optional[tuple[int, int, int]]:
        return parse_duration_calendar_components(value) if isinstance(value, str) else None

    def _number_of(value: object) -> Optional[float]:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return float(value)

    left_duration = _duration_of(left_value)
    right_duration = _duration_of(right_value)
    if left_duration is None and right_duration is None:
        return None

    if op in {"*", "/"}:
        if left_duration is not None and right_duration is None:
            factor = _number_of(right_value)
            if factor is None or factor == 0:
                return None
            scaled = _scale_duration(left_duration, factor, divide=(op == "/"))
            return None if scaled is None else Literal(scaled)
        if right_duration is not None and left_duration is None and op == "*":
            factor = _number_of(left_value)
            if factor is None:
                return None
            scaled = _scale_duration(right_duration, factor, divide=False)
            return None if scaled is None else Literal(scaled)
        return None

    sign = -1 if op == "-" else 1
    if left_duration is not None and right_duration is not None:
        return Literal(
            format_duration_calendar_components(
                left_duration[0] + sign * right_duration[0],
                left_duration[1] + sign * right_duration[1],
                left_duration[2] + sign * right_duration[2],
            )
        )

    if right_duration is not None:
        # <temporal> ± <duration>
        if not isinstance(left_value, str):
            return None
        try:
            temporal_value = _parse_temporal_value(left_value)
        except ValueError:
            return None
        if temporal_value is None:
            return None
        shifted = _shift_temporal_value(
            temporal_value,
            sign * right_duration[0],
            sign * right_duration[1],
            sign * right_duration[2],
        )
        return None if shifted is None else Literal(shifted)

    # <duration> + <temporal>: only addition commutes (duration - temporal is invalid)
    if op != "+" or not isinstance(right_value, str):
        return None
    try:
        temporal_value = _parse_temporal_value(right_value)
    except ValueError:
        return None
    if temporal_value is None:
        return None
    assert left_duration is not None
    shifted = _shift_temporal_value(temporal_value, left_duration[0], left_duration[1], left_duration[2])
    return None if shifted is None else Literal(shifted)


def rewrite_temporal_constructors_in_expr(expr_text: str) -> str:
    current_dt = py_datetime.now().astimezone()

    def _replace_current(match: re.Match[str]) -> str:
        normalized = _tt._current_temporal_literal(match.group("fn"), current_dt)
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    def _replace(match: re.Match[str]) -> str:
        normalized = _tt.normalize_temporal_constructor_text(match.group(0))
        if normalized is None:
            return match.group(0)
        escaped = normalized.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{escaped}'"

    rewritten = _tt.CURRENT_TEMPORAL_CALL_EXPR_RE.sub(_replace_current, expr_text)
    return _tt.TEMPORAL_CALL_EXPR_RE.sub(_replace, rewritten)


def fold_temporal_constructor_ast(node: ExprNode) -> ExprNode:
    current_dt = py_datetime.now().astimezone()

    def _fold(inner: ExprNode) -> ExprNode:
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
                current_literal = _tt._current_temporal_literal(inner.name, current_dt)
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
                    normalized = _tt.normalize_temporal_constructor_text(f"{inner.name}({rendered_arg})")
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
        rebuilt = _rebuild_expr_node(inner, rewrite=_fold, error_context="temporal constructor folding")
        if isinstance(rebuilt, BinaryOp):
            arithmetic = _fold_temporal_arithmetic(rebuilt)
            if arithmetic is not None:
                return arithmetic
        return rebuilt

    return _fold(node)
