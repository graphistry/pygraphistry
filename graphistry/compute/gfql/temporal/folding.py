from __future__ import annotations

from datetime import datetime as py_datetime
from datetime import timedelta
import re
from typing import Optional, cast

from graphistry.compute.gfql.temporal import constructors as _tt
from graphistry.compute.gfql.expr_parser import (
    ExprNode,
    FunctionCall,
    Literal,
    _rebuild_expr_node,
)
from graphistry.compute.gfql.temporal.durations import _fold_duration_function_call
from graphistry.compute.gfql.temporal.rendering import _render_temporal_arg
from graphistry.compute.gfql.temporal.truncation import _fold_temporal_truncate_call
from graphistry.compute.gfql.temporal.values import _format_localdatetime_parts


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
        return _rebuild_expr_node(inner, rewrite=_fold, error_context="temporal constructor folding")

    return _fold(node)
