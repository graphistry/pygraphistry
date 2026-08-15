"""Direct tests for temporal constant folding and duration components (#1915).

``test_temporal_and_union_semantics_1915.py`` pins the observable Cypher
semantics end-to-end; this file pins the branch structure of the fold itself,
in the style of ``test_aggregate_identity_branches.py``: every helper here is
pure (literal in -> literal or ``None`` out), so the inputs are built directly
as ``Literal``/``BinaryOp``/``FunctionCall`` nodes and the result is asserted
structurally, no engine execution involved.

``None`` from a fold helper means DECLINE (leave the node alone for the engine);
``Literal(None)`` means the fold produced a NULL. The two are distinguished
explicitly below because conflating them is exactly how B-2 shipped.
"""
from __future__ import annotations

from datetime import date as py_date
from typing import Any, Optional

import pytest

from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    Identifier,
    ExprNode,
    FunctionCall,
    Literal,
    MapLiteral,
)
from graphistry.compute.gfql.temporal.durations import (
    _NANOS_PER_DAY,
    _fold_duration_function_call,
    _fold_large_year_duration_function_call,
    _format_day_time_duration_nanoseconds,
    _format_large_time_only_duration,
    format_duration_calendar_components,
    parse_duration_calendar_components,
    parse_temporal_sort_duration_components,
    resolve_duration_text_property,
)
from graphistry.compute.gfql.temporal.folding import (
    _fold_datetime_epoch_function_call,
    _fold_temporal_arithmetic,
    _scale_duration,
    _shift_temporal_value,
    fold_temporal_constructor_ast,
    rewrite_temporal_constructors_in_expr,
)
from graphistry.compute.gfql.temporal.values import _TemporalValue, _parse_temporal_value

_HOUR_NS = 3_600_000_000_000
_MINUTE_NS = 60_000_000_000
_SECOND_NS = 1_000_000_000

DECLINE = object()
"""Sentinel: the helper returned ``None`` (no fold), as opposed to ``Literal(None)``."""


def _folded(node: ExprNode) -> Any:
    """``DECLINE`` when the fold declines, else the folded literal's value."""
    assert isinstance(node, BinaryOp)
    out = _fold_temporal_arithmetic(node)
    if out is None:
        return DECLINE
    assert isinstance(out, Literal)
    return out.value


def _arith(left: Any, op: str, right: Any) -> Any:
    return _folded(BinaryOp(op, Literal(left), Literal(right)))


def _call(name: str, *args: Any) -> Any:
    """``DECLINE`` when the duration-function fold declines, else its value."""
    out = _fold_duration_function_call(name, tuple(Literal(a) for a in args))
    if out is None:
        return DECLINE
    assert isinstance(out, Literal)
    return out.value


def _epoch(name: str, *args: Any) -> Any:
    """``DECLINE`` when the epoch fold declines, else its value."""
    out = _fold_datetime_epoch_function_call(name, tuple(Literal(a) for a in args))
    if out is None:
        return DECLINE
    assert isinstance(out, Literal)
    return out.value


# ===========================================================================
# 1. parse_duration_calendar_components: the three component groups
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # The group separation that makes `date + PT25H` a no-op: hours never
        # become days, and days never become hours.
        ("P1D", (0, 1, 0)),
        ("PT25H", (0, 0, 25 * _HOUR_NS)),
        ("P1M", (1, 0, 0)),
        ("P1Y", (12, 0, 0)),
        ("P1Y2M", (14, 0, 0)),
        ("P1W", (0, 7, 0)),
        ("PT0S", (0, 0, 0)),
        ("PT1M", (0, 0, _MINUTE_NS)),
        ("P1DT2H3M4.5S", (0, 1, 2 * _HOUR_NS + 3 * _MINUTE_NS + 4 * _SECOND_NS + 500_000_000)),
        # Fractional days spill into the seconds group (Neo4j: P0.5D == PT12H).
        ("P0.5D", (0, 0, 12 * _HOUR_NS)),
        ("P1.5D", (0, 1, 12 * _HOUR_NS)),
        ("P1.5W", (0, 10, 12 * _HOUR_NS)),
        ("PT1.5S", (0, 0, 1_500_000_000)),
        # A leading '-' negates every group at once.
        ("-P1D", (0, -1, 0)),
        ("-P1Y2M3D", (-14, -3, 0)),
        ("-PT1H", (0, 0, -_HOUR_NS)),
        # A per-token sign is honoured without the prefix form.
        ("PT-1H", (0, 0, -_HOUR_NS)),
        ("P-1D", (0, -1, 0)),
        # Surrounding whitespace is stripped.
        ("  P1D  ", (0, 1, 0)),
    ],
)
def test_parse_duration_calendar_components_groups(text: str, expected: tuple[int, int, int]) -> None:
    assert parse_duration_calendar_components(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",  # no designator at all
        "P",  # designator with an empty body
        "-P",
        "X1D",  # not a duration
        "2020-01-02",  # a temporal, not a duration
        "p1d",  # lower case is not ISO-8601
        "P1D2",  # trailing garbage after the last token
        "P1H",  # hours belong to the time part, not the date part
        "PT1D",  # days belong to the date part, not the time part
        "P1.5Y",  # fractional years/months have no fixed length -> rejected
        "P1.5M",
    ],
)
def test_parse_duration_calendar_components_declines(text: str) -> None:
    assert parse_duration_calendar_components(text) is None


def test_parse_duration_calendar_components_bare_t_is_zero() -> None:
    """``PT`` lexes to an empty time part rather than declining."""
    assert parse_duration_calendar_components("PT") == (0, 0, 0)


# ===========================================================================
# 2. format_duration_calendar_components: rendering back to ISO text
# ===========================================================================


@pytest.mark.parametrize(
    ("components", "expected"),
    [
        ((0, 0, 0), "PT0S"),
        # months == 0 takes the day/time-only renderer
        ((0, 1, 0), "P1D"),
        ((0, -1, 0), "P-1D"),
        ((0, 0, _HOUR_NS), "PT1H"),
        ((0, 0, _MINUTE_NS), "PT1M"),
        ((0, 0, 1_500_000_000), "PT1.5S"),
        ((0, 0, -1_500_000_000), "PT-1.5S"),
        ((0, 0, 1), "PT0.000000001S"),
        ((0, 2, _HOUR_NS), "P2DT1H"),
        # days and the seconds group are summed only once rendered
        ((0, 0, _NANOS_PER_DAY + _HOUR_NS), "P1DT1H"),
        ((0, 0, -_NANOS_PER_DAY), "P-1D"),
        # months != 0 takes the year/month renderer
        ((1, 0, 0), "P1M"),
        ((12, 0, 0), "P1Y"),
        ((13, 0, 0), "P1Y1M"),
        ((24, 0, 0), "P2Y"),
        ((-1, 0, 0), "P-1M"),
        ((-12, 0, 0), "P-1Y"),
        ((-13, 0, 0), "P-1Y-1M"),
        # ... and appends the day/time tail only when it is non-zero
        ((1, 1, 0), "P1M1D"),
        ((1, 0, _HOUR_NS), "P1MT1H"),
        ((3, 0, -_HOUR_NS), "P3MT-1H"),
        ((1, -1, 0), "P1M-1D"),
        # whole seconds render without a fractional part, signed either way
        ((0, 0, _SECOND_NS), "PT1S"),
        ((0, 0, _MINUTE_NS + 30 * _SECOND_NS), "PT1M30S"),
        ((0, 0, -(_MINUTE_NS + 30 * _SECOND_NS)), "PT-1M-30S"),
    ],
)
def test_format_duration_calendar_components(components: tuple[int, int, int], expected: str) -> None:
    assert format_duration_calendar_components(*components) == expected


@pytest.mark.parametrize(
    "text",
    ["P1D", "P1M", "P1Y1M", "P1DT2H3M4.5S", "-P1D", "PT0S", "P1M1D", "P1MT1H"],
)
def test_duration_calendar_components_round_trip(text: str) -> None:
    components = parse_duration_calendar_components(text)
    assert components is not None
    rendered = format_duration_calendar_components(*components)
    assert parse_duration_calendar_components(rendered) == components


def test_format_normalizes_an_overlong_seconds_group_into_days() -> None:
    """Rendering is where 25 hours becomes a day: the round trip is one-way."""
    parsed = parse_duration_calendar_components("PT25H")
    assert parsed == (0, 0, 25 * _HOUR_NS)
    assert format_duration_calendar_components(*parsed) == "P1DT1H"
    assert parse_duration_calendar_components("P1DT1H") == (0, 1, _HOUR_NS)


def test_day_time_and_time_only_renderers_differ_on_the_day_group() -> None:
    """``include_days`` is the only difference: one hoists days, one does not."""
    assert _format_day_time_duration_nanoseconds(_NANOS_PER_DAY + _HOUR_NS) == "P1DT1H"
    assert _format_large_time_only_duration(_NANOS_PER_DAY + _HOUR_NS) == "PT25H"
    assert _format_day_time_duration_nanoseconds(0) == "PT0S"
    assert _format_large_time_only_duration(0) == "PT0S"


# ===========================================================================
# 3. _scale_duration: duration * | / number
# ===========================================================================


def test_scale_duration_multiplies_every_group() -> None:
    assert _scale_duration((0, 1, 0), 3.0, divide=False) == "P3D"
    assert _scale_duration((1, 0, 0), 2.0, divide=False) == "P2M"
    assert _scale_duration((0, 0, _SECOND_NS), 3.0, divide=True) == "PT0.333333333S"


def test_scale_duration_declines_a_fractional_month_result() -> None:
    """Half a month has no fixed length, so the fold must decline rather than round."""
    assert _scale_duration((1, 0, 0), 2.0, divide=True) is None
    assert _scale_duration((1, 0, 0), 0.5, divide=False) is None
    assert _scale_duration((2, 0, 0), 2.0, divide=True) == "P1M"


def test_scale_duration_rounds_the_seconds_group_to_whole_nanoseconds() -> None:
    assert _scale_duration((0, 0, 1), 3.0, divide=True) == "PT0S"


# ===========================================================================
# 4. _shift_temporal_value: which groups a given temporal kind consumes
# ===========================================================================


def test_shift_date_drops_the_seconds_group() -> None:
    """openCypher: a DATE has no time-of-day for the seconds group to land in."""
    value = _TemporalValue(kind="date", date_value=py_date(2020, 1, 2))
    assert _shift_temporal_value(value, 0, 0, 25 * _HOUR_NS) == "2020-01-02"
    assert _shift_temporal_value(value, 0, 1, 25 * _HOUR_NS) == "2020-01-03"
    assert _shift_temporal_value(value, 1, 0, 0) == "2020-02-02"


def test_shift_time_wraps_modulo_a_day_in_both_directions() -> None:
    """A TIME/LOCALTIME has no date to carry into, so it wraps."""
    localtime = _TemporalValue(kind="localtime", date_value=None, hour=12)
    assert _shift_temporal_value(localtime, 0, 0, 25 * _HOUR_NS) == "13:00"
    assert _shift_temporal_value(localtime, 0, 0, -25 * _HOUR_NS) == "11:00"
    assert _shift_temporal_value(localtime, 0, 0, -13 * _HOUR_NS) == "23:00"
    # The month/day groups have nothing to act on and are ignored outright.
    assert _shift_temporal_value(localtime, 5, 9, 0) == "12:00"


def test_shift_time_keeps_its_zone_suffix_but_localtime_has_none() -> None:
    zoned = _TemporalValue(kind="time", date_value=None, hour=12, tz_suffix="Z")
    assert _shift_temporal_value(zoned, 0, 0, _HOUR_NS) == "13:00Z"
    naive = _TemporalValue(kind="localtime", date_value=None, hour=12)
    assert _shift_temporal_value(naive, 0, 0, _HOUR_NS) == "13:00"


def test_shift_datetime_carries_the_seconds_group_into_the_date() -> None:
    value = _TemporalValue(kind="localdatetime", date_value=py_date(2020, 1, 2), hour=23)
    assert _shift_temporal_value(value, 0, 0, 2 * _HOUR_NS) == "2020-01-03T01:00"
    assert _shift_temporal_value(value, 0, 0, -24 * _HOUR_NS) == "2020-01-01T23:00"
    zoned = _TemporalValue(
        kind="datetime", date_value=py_date(2020, 1, 2), hour=23, tz_suffix="+01:00"
    )
    assert _shift_temporal_value(zoned, 0, 0, 2 * _HOUR_NS) == "2020-01-03T01:00+01:00"


def test_shift_declines_when_the_shifted_year_leaves_the_representable_range() -> None:
    value = _TemporalValue(kind="date", date_value=py_date(9999, 12, 1))
    assert _shift_temporal_value(value, 12, 0, 0) is None


# ===========================================================================
# 5. _fold_temporal_arithmetic: the operand-type matrix
# ===========================================================================


@pytest.mark.parametrize(
    ("left", "op", "right", "expected"),
    [
        # <date> +/- <duration>
        ("2020-01-02", "+", "P1D", "2020-01-03"),
        ("2020-01-02", "-", "P1D", "2020-01-01"),
        # the day/seconds group split, observed end to end
        ("2020-01-02", "+", "PT25H", "2020-01-02"),
        ("2020-01-02", "-", "PT25H", "2020-01-02"),
        # month arithmetic clamps to the end of the target month
        ("2020-01-31", "+", "P1M", "2020-02-29"),  # leap year
        ("2019-01-31", "+", "P1M", "2019-02-28"),  # common year
        ("2020-03-31", "-", "P1M", "2020-02-29"),
        ("2020-01-31", "+", "P1Y", "2021-01-31"),
        ("2020-02-29", "+", "P1Y", "2021-02-28"),  # leap day into a common year
        # <localdatetime>/<datetime>/<time>/<localtime> +/- <duration>
        ("2020-01-02T00:00:00", "+", "PT25H", "2020-01-03T01:00"),
        ("2020-01-02T00:00:00Z", "+", "PT25H", "2020-01-03T01:00Z"),
        ("12:00:00", "+", "PT25H", "13:00"),
        ("12:00:00Z", "+", "PT25H", "13:00Z"),
        ("12:00", "-", "PT25H", "11:00"),
        # <duration> + <temporal> commutes
        ("P1D", "+", "2020-01-02", "2020-01-03"),
        # <duration> +/- <duration>, per component group
        ("P1D", "+", "P1D", "P2D"),
        ("P1D", "-", "P2D", "P-1D"),
        ("P1M", "+", "P1D", "P1M1D"),
        ("P1M", "-", "P1M", "PT0S"),
        ("PT1H", "+", "PT30M", "PT1H30M"),
        # <duration> * | / <number>
        ("P1D", "*", 2, "P2D"),
        ("P1D", "/", 2, "PT12H"),
        ("P1D", "*", 0.5, "PT12H"),
        ("P1D", "*", -1, "P-1D"),
        ("P1D", "/", -2, "PT-12H"),
        # <number> * <duration> commutes
        (2, "*", "P1D", "P2D"),
        (0.5, "*", "P1D", "PT12H"),
    ],
)
def test_fold_temporal_arithmetic_folds(left: Any, op: str, right: Any, expected: str) -> None:
    assert _arith(left, op, right) == expected


@pytest.mark.parametrize(
    ("left", "op", "right"),
    [
        # Neither operand is duration-shaped: ordinary string concatenation is
        # left completely alone (the reason `'a' + 'b'` still concatenates).
        ("a", "+", "b"),
        ("2020-01-02", "+", "2020-01-03"),
        (1, "+", 2),
        # A non-foldable operator never engages, even on two durations.
        ("P1D", "%", "P1D"),
        ("P1D", "<", "P1D"),
        # Division does not commute: a number divided by a duration is invalid.
        (2, "/", "P1D"),
        # Two durations multiplied/divided is invalid.
        ("P1D", "*", "P1D"),
        ("P1D", "/", "P1D"),
        # Duration minus temporal is invalid (only addition commutes).
        ("P1D", "-", "2020-01-02"),
        # Duration +/- a bare number is not duration arithmetic.
        ("P1D", "+", 1),
        (1, "+", "P1D"),
        ("P1D", "-", 1),
        # Booleans are not Cypher numbers, so they never scale a duration.
        ("P1D", "*", True),
        (True, "*", "P1D"),
        ("P1D", "/", False),
        # Scaling by zero: multiplication by 0 and any division by 0 decline
        # rather than emit PT0S / raise.
        ("P1D", "/", 0),
        ("P1D", "*", 0),
        # A fractional month result has no fixed length.
        ("P1M", "*", 0.5),
        ("P1M", "/", 2),
        # The left operand is not a parseable temporal.
        ("foo", "+", "P1D"),
        ("P1D", "+", "foo"),
        # A year outside the representable range raises inside the parse and is
        # swallowed into a decline.
        ("0000-01-01", "-", "P1Y"),
        ("P1Y", "+", "0000-01-01"),
        # ... and one that only leaves the range after the shift.
        ("9999-12-01", "+", "P1Y"),
        # A non-string operand opposite a duration cannot be a temporal.
        (3.5, "+", "P1D"),
    ],
)
def test_fold_temporal_arithmetic_declines(left: Any, op: str, right: Any) -> None:
    assert _arith(left, op, right) is DECLINE


def test_fold_temporal_arithmetic_needs_two_literals() -> None:
    """A residual COLUMN operand is not foldable; the engine decides instead."""
    assert _fold_temporal_arithmetic(BinaryOp("+", Identifier("ts"), Literal("P1D"))) is None
    assert _fold_temporal_arithmetic(BinaryOp("+", Literal("2020-01-02"), Identifier("d"))) is None


def test_fold_temporal_arithmetic_zero_duration_is_identity() -> None:
    assert _arith("2020-01-02", "+", "PT0S") == "2020-01-02"
    assert _arith("2020-01-02", "-", "PT0S") == "2020-01-02"
    assert _arith("P1D", "+", "PT0S") == "P1D"


# ===========================================================================
# 6. _fold_datetime_epoch_function_call
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "args", "expected"),
    [
        ("datetime.fromepoch", (0,), "1970-01-01T00:00Z"),
        ("datetime.fromepoch", (1, 2), "1970-01-01T00:00:01.000000002Z"),
        ("datetime.fromepoch", (-1,), "1969-12-31T23:59:59Z"),
        ("datetime.fromepochmillis", (1500,), "1970-01-01T00:00:01.5Z"),
        ("datetime.fromepochmillis", (0,), "1970-01-01T00:00Z"),
    ],
)
def test_fold_datetime_epoch(name: str, args: tuple[Any, ...], expected: str) -> None:
    assert _epoch(name, *args) == expected


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("datetime.fromepoch", ()),  # arity
        ("datetime.fromepoch", (1, 2, 3)),
        ("datetime.fromepochmillis", (1, 2)),
        ("datetime.fromepoch", ("x",)),  # not an integer
        ("datetime.fromepoch", (1.5,)),
        ("datetime.fromepoch", (True,)),  # bool is not a Cypher integer
        ("datetime.notanepoch", (1,)),  # not an epoch constructor at all
    ],
)
def test_fold_datetime_epoch_declines(name: str, args: tuple[Any, ...]) -> None:
    assert _epoch(name, *args) is DECLINE


@pytest.mark.parametrize("name", ["datetime.fromepoch", "datetime.fromepochmillis"])
def test_fold_datetime_epoch_propagates_null(name: str) -> None:
    """A NULL argument folds to NULL -- which is not the same as declining."""
    assert _epoch(name, None) is None


def test_fold_datetime_epoch_null_wins_over_a_bad_arity() -> None:
    assert _epoch("datetime.fromepoch", None, 1, 2) is None


# ===========================================================================
# 7. _fold_duration_function_call / the wide-year fallback
# ===========================================================================


@pytest.mark.parametrize(
    ("name", "start", "end", "expected"),
    [
        ("duration.between", "2020-01-01", "2021-03-05", "P1Y2M4D"),
        ("duration.between", "2020-01-01", "2020-01-05", "P4D"),
        ("duration.between", "12:00", "13:30", "PT1H30M"),
        ("duration.between", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z", "P1D"),
        ("duration.inmonths", "2020-01-01", "2021-03-05", "P1Y2M"),
        ("duration.indays", "2020-01-01", "2020-03-05", "P64D"),
        ("duration.inseconds", "2020-01-01", "2020-01-02", "PT24H"),
        ("duration.inseconds", "12:00", "13:00", "PT1H"),
        # A time-only pair has no date component for the month/day answers.
        ("duration.inmonths", "12:00", "13:00", "PT0S"),
        ("duration.indays", "12:00", "13:00", "PT0S"),
        # A year/month answer carries the whole time tail, fractional or not.
        ("duration.between", "2020-01-01T10:00:00", "2021-03-05T12:34:56.5", "P1Y2M4DT2H34M56.5S"),
        ("duration.between", "2020-01-01T10:00:00", "2021-03-05T12:34:56", "P1Y2M4DT2H34M56S"),
        ("duration.between", "2020-01-01T10:00:00", "2021-03-01T11:00:00", "P1Y2MT1H"),
    ],
)
def test_fold_duration_function_call(name: str, start: str, end: str, expected: str) -> None:
    assert _call(name, start, end) == expected


@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("duration.between", ("2020-01-01",)),  # arity
        ("duration.between", ("2020-01-01", "2020-01-02", "2020-01-03")),
        ("duration.between", ("2020-01-01", 5)),  # not text
        ("duration.between", ("foo", "bar")),  # unparseable either way
        ("duration.nope", ("2020-01-01", "2020-01-02")),  # unknown function
    ],
)
def test_fold_duration_function_call_declines(name: str, args: tuple[Any, ...]) -> None:
    assert _call(name, *args) is DECLINE


def test_fold_duration_function_call_propagates_null() -> None:
    assert _call("duration.between", "2020-01-01", None) is None
    assert _call("duration.between", None, None) is None


@pytest.mark.parametrize(
    ("start", "end"),
    [
        ("0000-01-01", "0000-02-01"),  # year 0 is outside the representable range
        ("2020-02-30", "2020-03-01"),  # a syntactically valid but impossible date
        ("2020-13-01", "2020-13-02"),
    ],
)
def test_fold_duration_function_call_swallows_an_unrepresentable_date(start: str, end: str) -> None:
    """The parse raises rather than returning None; the fold must still decline."""
    assert _call("duration.between", start, end) is DECLINE


@pytest.mark.parametrize(
    ("name", "start", "end", "expected"),
    [
        # Years outside the narrow parser's range fall through to the wide path.
        ("duration.between", "+10000-01-01", "+10002-03-05", "P2Y2M4D"),
        ("duration.inseconds", "+10000-01-01T00:00:00", "+10000-01-01T01:00:00", "PT1H"),
    ],
)
def test_fold_wide_year_duration(name: str, start: str, end: str, expected: str) -> None:
    assert _call(name, start, end) == expected


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [
        # duration.between over wide years is date-only ...
        ("duration.between", "+10000-01-01T00:00:00", "+10002-01-01T00:00:00"),
        # ... and is not defined backwards.
        ("duration.between", "+10002-01-01", "+10000-03-05"),
        # duration.inseconds over wide years is localdatetime-only ...
        ("duration.inseconds", "+10000-01-01", "+10002-03-05"),
        # ... and the month/day answers have no wide-year implementation.
        ("duration.inmonths", "+10000-01-01", "+10002-03-05"),
        ("duration.indays", "+10000-01-01", "+10002-03-05"),
        # A single unparseable side still declines.
        ("duration.between", "+10000-01-01", "nonsense"),
    ],
)
def test_fold_wide_year_duration_declines(name: str, start: str, end: str) -> None:
    assert _fold_large_year_duration_function_call(name, start, end) is None
    assert _call(name, start, end) is DECLINE


def test_fold_wide_year_duration_borrows_across_the_month_boundary() -> None:
    """A negative day difference borrows the length of the preceding month."""
    assert _fold_large_year_duration_function_call("duration.between", "+10000-03-01", "+10000-04-30") == "P1M29D"
    # February of +10000 is 29 days long (it is a leap year), so the borrow is 29.
    assert _fold_large_year_duration_function_call("duration.between", "+10000-01-15", "+10000-03-10") == "P1M24D"
    # ... and a negative month difference borrows a year.
    assert _fold_large_year_duration_function_call("duration.between", "+10000-12-01", "+10001-01-01") == "P1M"
    # An identical pair has no components at all.
    assert _fold_large_year_duration_function_call("duration.between", "+10000-01-01", "+10000-01-01") == "PT0S"
    # Borrowing in January walks back into the previous December.
    assert _fold_large_year_duration_function_call("duration.between", "+10000-12-15", "+10001-01-10") == "P26D"
    assert _fold_large_year_duration_function_call("duration.between", "+10000-12-31", "+10001-01-01") == "P1D"


# ===========================================================================
# 8. parse_temporal_sort_duration_components (the ORDER BY sort-key variant)
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Unlike the calendar variant, days collapse INTO the nanosecond total.
        ("P1D", (0, _NANOS_PER_DAY)),
        ("PT25H", (0, 25 * _HOUR_NS)),
        ("P1W", (0, 7 * _NANOS_PER_DAY)),
        ("P0.5D", (0, 12 * _HOUR_NS)),
        ("P1DT2H", (0, _NANOS_PER_DAY + 2 * _HOUR_NS)),
        ("PT1M", (0, _MINUTE_NS)),
        ("PT1.5S", (0, 1_500_000_000)),
        ("-P1M", (-1, 0)),
        ("-P1D", (0, -_NANOS_PER_DAY)),
    ],
)
def test_parse_temporal_sort_duration_components(text: str, expected: tuple[int, int]) -> None:
    assert parse_temporal_sort_duration_components(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "",
        "P",
        "X1D",
        "p1d",
        "P1D2",
        # A fractional month has no fixed length here either.
        "P1.5M",
        # This variant lexes with the day/time-only token pattern, so a YEAR
        # unit is not lexable at all here (the calendar variant accepts it) --
        # which is why the ``Y`` arm of its token loop is unreachable.
        "P1Y",
        "P1.5Y",
        "P1Y2M3DT4H5M6S",
    ],
)
def test_parse_temporal_sort_duration_components_declines(text: str) -> None:
    assert parse_temporal_sort_duration_components(text) is None


def test_sort_and_calendar_variants_disagree_only_on_the_day_group() -> None:
    """The whole reason both exist: PT25H and P1D sort equal, but add differently."""
    assert parse_temporal_sort_duration_components("P1D") == parse_temporal_sort_duration_components("PT24H")
    assert parse_duration_calendar_components("P1D") != parse_duration_calendar_components("PT24H")


# ===========================================================================
# 9. resolve_duration_text_property
# ===========================================================================


@pytest.mark.parametrize(
    ("text", "prop", "expected"),
    [
        ("P1DT2H", "days", "1"),
        ("P1DT2H", "seconds", "7200"),
        ("P1DT2H", "nanosecondsOfSecond", "0"),
        ("PT1H", "days", "0"),
        ("PT1.5S", "seconds", "1"),
        ("PT1.5S", "nanosecondsOfSecond", "500000000"),
        ("PT1M", "seconds", "60"),
        # The month group is deliberately not reachable through the day/second
        # properties: P1M contributes no days and no seconds.
        ("P1M", "days", "0"),
        ("P1M", "seconds", "0"),
        # A '-P' prefix is only stripped, so the properties read the magnitude.
        ("-P1DT2H", "days", "1"),
    ],
)
def test_resolve_duration_text_property(text: str, prop: str, expected: str) -> None:
    assert resolve_duration_text_property(text, prop) == expected


@pytest.mark.parametrize(
    ("text", "prop"),
    [
        ("P1DT2H", "months"),  # not a supported property
        ("P1DT2H", "nanoseconds"),
        ("2020-01-02", "days"),  # not duration-shaped
        ("", "days"),
    ],
)
def test_resolve_duration_text_property_declines(text: str, prop: str) -> None:
    assert resolve_duration_text_property(text, prop) is None


# ===========================================================================
# 10. rewrite_temporal_constructors_in_expr (the pre-parse TEXT rewriter)
# ===========================================================================


@pytest.mark.parametrize(
    ("expr_text", "expected"),
    [
        ("date('2020-01-02')", "'2020-01-02'"),
        ("duration('P1D')", "'P1D'"),
        ("time('12:00')", "'12:00Z'"),
        ("n.ts > datetime('2020-01-02T03:04:05Z')", "n.ts > '2020-01-02T03:04:05Z'"),
        # Both operands of an arithmetic expression are rewritten in place.
        ("date('2020-01-02') + duration('P1D')", "'2020-01-02' + 'P1D'"),
    ],
)
def test_rewrite_temporal_constructors_in_expr(expr_text: str, expected: str) -> None:
    assert rewrite_temporal_constructors_in_expr(expr_text) == expected


@pytest.mark.parametrize(
    "expr_text",
    [
        "date('not-a-date')",  # a constructor that does not normalize is left alone
        "foo('x')",  # not a temporal constructor
        "n.a + 1",  # nothing to rewrite
        "'date(x)'",  # an unquoted argument is not a temporal-constructor call
    ],
)
def test_rewrite_temporal_constructors_in_expr_leaves_text_alone(expr_text: str) -> None:
    assert rewrite_temporal_constructors_in_expr(expr_text) == expr_text


@pytest.mark.parametrize("fn", ["date", "localtime", "time", "localdatetime", "datetime"])
def test_rewrite_zero_arg_constructors_to_a_quoted_literal(fn: str) -> None:
    """``date()`` etc. become a quoted ISO literal of the matching temporal kind."""
    rewritten = rewrite_temporal_constructors_in_expr(f"{fn}()")
    assert rewritten.startswith("'") and rewritten.endswith("'")
    parsed = _parse_temporal_value(rewritten[1:-1])
    assert parsed is not None and parsed.kind == fn


# ===========================================================================
# 11. fold_temporal_constructor_ast: the arithmetic reaches the AST walker
# ===========================================================================


def _fold_value(node: ExprNode) -> Any:
    folded = fold_temporal_constructor_ast(node)
    assert isinstance(folded, Literal), folded
    return folded.value


def test_ast_fold_evaluates_constructor_arithmetic() -> None:
    """The whole B-2 defect in one shape: this used to be '2020-01-02P1D'."""
    node = BinaryOp(
        "+",
        FunctionCall("date", (Literal("2020-01-02"),)),
        FunctionCall("duration", (Literal("P1D"),)),
    )
    assert _fold_value(node) == "2020-01-03"


def test_ast_fold_evaluates_nested_arithmetic() -> None:
    node = BinaryOp(
        "+",
        BinaryOp(
            "+",
            FunctionCall("date", (Literal("2020-01-31"),)),
            FunctionCall("duration", (Literal("P1M"),)),
        ),
        FunctionCall("duration", (Literal("P1D"),)),
    )
    assert _fold_value(node) == "2020-03-01"


def test_ast_fold_leaves_a_column_operand_alone() -> None:
    node = BinaryOp("+", Identifier("ts"), FunctionCall("duration", (Literal("P1D"),)))
    folded = fold_temporal_constructor_ast(node)
    assert isinstance(folded, BinaryOp)
    assert isinstance(folded.left, Identifier)
    assert folded.right == Literal("P1D")


def test_ast_fold_leaves_ordinary_string_concatenation_alone() -> None:
    node = BinaryOp("+", Literal("hello "), Literal("world"))
    folded = fold_temporal_constructor_ast(node)
    assert isinstance(folded, BinaryOp)
    assert folded.left == Literal("hello ")
    assert folded.right == Literal("world")


def test_ast_fold_null_constructor_argument() -> None:
    for name in ("date", "localtime", "time", "localdatetime", "datetime", "duration"):
        assert _fold_value(FunctionCall(name, (Literal(None),))) is None


def test_ast_fold_tostring() -> None:
    assert _fold_value(FunctionCall("tostring", (Literal(1),))) == "1"
    assert _fold_value(FunctionCall("tostring", (Literal(True),))) == "true"
    assert _fold_value(FunctionCall("tostring", (Literal(False),))) == "false"
    assert _fold_value(FunctionCall("tostring", (Literal(None),))) is None


def test_ast_fold_zero_arg_constructors_produce_a_literal() -> None:
    for name in ("date", "localtime", "time", "localdatetime", "datetime"):
        folded = fold_temporal_constructor_ast(FunctionCall(name, ()))
        assert isinstance(folded, Literal)
        assert isinstance(folded.value, str)
        parsed = _parse_temporal_value(folded.value)
        assert parsed is not None and parsed.kind == name


def test_ast_fold_declines_distinct_calls() -> None:
    """DISTINCT is aggregate syntax; a DISTINCT-marked call is never a constructor."""
    node: ExprNode = FunctionCall("date", (Literal("2020-01-02"),), distinct=True)
    folded = fold_temporal_constructor_ast(node)
    assert isinstance(folded, FunctionCall)
    assert folded.distinct is True


def test_ast_fold_declines_an_unparseable_constructor_argument() -> None:
    folded = fold_temporal_constructor_ast(FunctionCall("date", (Literal("not-a-date"),)))
    assert isinstance(folded, FunctionCall)
    assert folded.name == "date"


def test_ast_fold_folds_truncate_and_duration_helpers() -> None:
    truncated = fold_temporal_constructor_ast(
        FunctionCall("date.truncate", (Literal("month"), Literal("2020-05-17"), MapLiteral(())))
    )
    assert truncated == Literal("2020-05-01")
    between = fold_temporal_constructor_ast(
        FunctionCall("duration.between", (Literal("2020-01-01"), Literal("2020-01-05")))
    )
    assert between == Literal("P4D")
    epoch = fold_temporal_constructor_ast(FunctionCall("datetime.fromepoch", (Literal(0),)))
    assert epoch == Literal("1970-01-01T00:00Z")


def test_ast_fold_recurses_into_call_arguments() -> None:
    """A non-temporal call keeps its shape but its argument still folds."""
    node = FunctionCall(
        "length",
        (
            BinaryOp(
                "+",
                FunctionCall("date", (Literal("2020-01-02"),)),
                FunctionCall("duration", (Literal("P1D"),)),
            ),
        ),
    )
    folded = fold_temporal_constructor_ast(node)
    assert isinstance(folded, FunctionCall)
    assert folded.name == "length"
    inner: Optional[ExprNode] = folded.args[0]
    assert inner == Literal("2020-01-03")


def test_temporal_shift_past_date_max_declines_not_overflowerror():
    """A shift past ``date.max`` raises OverflowError from ``timedelta`` (not the
    ValueError the constructors raise), so both carry sites must catch it or a raw
    OverflowError escapes the fold. Found by the coverage sweep of this module."""
    import graphistry
    import pandas as pd
    from graphistry.compute.exceptions import GFQLTypeError

    g = (graphistry.nodes(pd.DataFrame({"id": [0]}), "id")
         .edges(pd.DataFrame({"s": [0], "d": [0]}), "s", "d"))
    for q in [
        "RETURN date('9999-12-31') + duration('P1D') AS d",              # day carry
        "RETURN localdatetime('9999-12-31T23:00:00') + duration('PT2H') AS d",  # seconds carry
    ]:
        with pytest.raises(GFQLTypeError):
            g.gfql(q, engine="pandas")
