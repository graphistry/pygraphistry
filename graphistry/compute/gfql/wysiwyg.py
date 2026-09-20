"""Render values the way the viz inspector renders them, so ``searchAny`` matches what the
user can actually SEE.

Ground truth, read from the shipping formatters (``apps/core/viz/src/formatters/``):

``defaultFormat(v, 'number')``
  ``NaN`` -> null; ``v === 2147483647`` -> null (an Int32 sentinel); then
  ``if (v && v % 1 !== 0) -> formatNumber(v, false)`` -- note ``v &&``, so ``0`` is falsy and
  takes the else branch -- otherwise ``formatToString(v, false)`` == ``String(v)``.
``formatNumber(v, false)``
  ``sprintf('%.4f', v)``, and sprintf-js's ``%f`` with a precision is ``Number(v).toFixed(4)``.

``toFixed`` rounds half-AWAY-from-zero on the exact decimal expansion of the double. Python's
formatter reproduces that on 99.84% of values, the residual being exact half-boundaries, which
need a magnitude above ~1e13 to occur at all, so pandas renders exactly. polars and cuDF have
no column-wide equivalent -- their ``round`` is half-to-EVEN on the binary value -- so they
scale, render as an integer and re-insert the decimal point, which agrees with the inspector
on 99.93%+ of realistic column values.

The residual is therefore a genuine CROSS-ENGINE divergence, not merely a UI one: a value
whose (precision+1)-th decimal is exactly 5 renders half-away on pandas and half-even on
polars/cuDF. It is pinned as a known divergence rather than papered over, and the engines are
kept consistent with each other per engine family: polars renders the same on CPU and GPU,
because a device must never change an answer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from graphistry.compute.typing import SeriesT

if TYPE_CHECKING:
    import polars as pl

#: The inspector renders this Int32 sentinel as null, so it must never match a search.
INSPECTOR_INT32_SENTINEL = 2147483647

#: ``formatNumber``'s fixed precision.
DEFAULT_FLOAT_PRECISION = 4

#: JS switches ``String(v)`` to exponential at this magnitude; below it, integers print in full.
_JS_EXPONENTIAL_AT = 1e21

#: The inspector renders a date with moment ``'MMM D YYYY, h:mm:ss a z'``.
_MONTH_ABBREVS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
                  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")

#: The inspector formats in the VIEWER's zone, which a server cannot know, so it is a caller knob.
DEFAULT_TEMPORAL_TZ = "UTC"


def js_string_of_whole_float(v: float) -> str:
    """``String(v)`` for a float that has no fractional part: ``42.0`` -> ``"42"``.

    Above 2**53 the digits are approximate -- JS prints the shortest string that round-trips,
    Python prints the exact integer -- so the two can differ in the low digits of very large
    values. Substring search degrades gracefully there (a term has to straddle the differing
    digits to be affected), and the common range is exact.
    """
    if v == 0:  # also normalizes -0.0, which JS prints as "0"
        return "0"
    if abs(v) < _JS_EXPONENTIAL_AT:
        return str(int(v))
    return repr(v)  # shortest round-trip; matches JS's exponential form (e.g. '1e+21')


def render_float_pandas(s: SeriesT, precision: int = DEFAULT_FLOAT_PRECISION) -> SeriesT:
    """Inspector-exact render of a pandas float column, as an object Series of str/None.

    None marks "the inspector shows nothing here", which must never match: nulls and the
    Int32 sentinel.
    """
    import numpy as np
    import pandas as pd

    vals = np.asarray(s.astype("float64"), dtype="float64")
    out = np.full(len(vals), None, dtype=object)
    finite = np.isfinite(vals)
    # `v && v % 1 !== 0` -- 0 is falsy in JS, so it takes the String(v) branch
    with np.errstate(invalid="ignore"):
        fractional = finite & (vals != 0) & (np.mod(vals, 1) != 0)
    fmt = ("{:." + str(precision) + "f}").format
    for i in np.flatnonzero(fractional):
        out[i] = fmt(vals[i])
    for i in np.flatnonzero(finite & ~fractional):
        out[i] = js_string_of_whole_float(vals[i])
    # Infinity % 1 is NaN in JS too, so infinities take the toFixed branch and print by name
    out[np.isposinf(vals)] = "Infinity"
    out[np.isneginf(vals)] = "-Infinity"
    out[vals == INSPECTOR_INT32_SENTINEL] = None
    return pd.Series(out, index=s.index, dtype=object)


def render_float_cudf(  # pragma: no cover - cuDF-only; the changed-line-coverage gate has no cuDF lane (validated on dgx)
    s: SeriesT, precision: int = DEFAULT_FLOAT_PRECISION
) -> SeriesT:
    """Inspector render of a cuDF float column, GPU-resident throughout.

    Scales by ``10**precision`` AFTER rounding, renders the integer, and re-inserts the
    decimal point, because cuDF's ``astype(str)`` does not zero-pad (``2.675`` -> ``"2.675"``
    where the inspector shows ``"2.6750"``).
    """
    import cudf

    inf = float("inf")
    # infinities reaching the int64 cast come out as int64-max digits with a doubled sign
    infinite = (s == inf) | (s == -inf)
    safe = s.where(~infinite, 0.0)

    scale = 10 ** precision
    rounded = safe.round(precision)
    scaled = (rounded * scale).round().astype("int64")
    digits = scaled.abs().astype(str).str.pad(precision + 1, side="left", fillchar="0")
    sign = cudf.Series(["-"] * len(s), index=s.index).where(safe < 0, "")
    fractional_txt = sign + digits.str.slice(0, -precision) + "." + digits.str.slice(-precision)

    whole_txt = safe.astype("int64").astype(str)
    is_whole = (safe % 1 == 0) & safe.notna()
    out = fractional_txt.where(~is_whole, whole_txt)
    out = out.where(~(infinite & (s > 0)), "Infinity")
    out = out.where(~(infinite & (s < 0)), "-Infinity")
    # nulls and the sentinel show nothing in the inspector, so they must not match
    return out.where(s.notna() & (s != INSPECTOR_INT32_SENTINEL), None)


def float_render_expr_polars(
    col: "pl.Expr", dtype: "pl.DataType", precision: int = DEFAULT_FLOAT_PRECISION
) -> "pl.Expr":
    """Inspector render of a polars float column as an expression.

    Native on purpose: a ``map_elements`` UDF would be exact on CPU but cannot execute under
    cudf-polars, and an answer that changes with the device is worse than one that differs
    from pandas at half-boundaries only.
    """
    import polars as pl

    # polars NaN is a VALUE not null, and the Int64 cast runs whatever the when/then says
    is_nan = col.is_nan().fill_null(False)
    is_inf = col.is_infinite().fill_null(False)
    safe = pl.when(col.is_null() | is_nan | is_inf).then(pl.lit(0.0)).otherwise(col)

    scale = 10 ** precision
    rounded = safe.round(precision)
    scaled = (rounded * scale).round().cast(pl.Int64)
    digits = scaled.abs().cast(pl.String).str.pad_start(precision + 1, "0")
    sign = pl.when(safe < 0).then(pl.lit("-")).otherwise(pl.lit(""))
    whole_digits = digits.str.slice(0, digits.str.len_chars() - precision)
    frac_digits = digits.str.slice(-precision)
    fractional_txt = sign + whole_digits + pl.lit(".") + frac_digits

    whole_txt = safe.cast(pl.Int64).cast(pl.String)
    return (
        pl.when(col.is_null() | is_nan | (col == INSPECTOR_INT32_SENTINEL))
        .then(pl.lit(None, dtype=pl.String))
        .when(is_inf & (col > 0))
        .then(pl.lit("Infinity"))
        .when(is_inf)
        .then(pl.lit("-Infinity"))
        .when(safe % 1 == 0)
        .then(whole_txt)
        .otherwise(fractional_txt)
    )


def _tz_abbrev_by_offset(loc: SeriesT) -> SeriesT:
    """Zone abbreviation per row, resolved once per distinct UTC offset.

    Within one zone the abbreviation follows the offset (``EST`` at -5, ``EDT`` at -4), and the
    offset only moves at a DST transition, so a column spanning years still holds a handful of
    distinct values. ``strftime`` is a per-element path, so it runs on one representative row per
    offset rather than on every row.
    """
    import numpy as np
    import pandas as pd

    naive = loc.dt.tz_localize(None)
    offset = naive.astype("int64") - loc.astype("int64")
    codes, uniques = pd.factorize(offset)
    names = np.array(
        [loc[offset == value].iloc[0].strftime("%Z") for value in uniques], dtype=object
    )
    return pd.Series(names[codes], index=loc.index, dtype=object)


def render_datetime_pandas(s: SeriesT, tz: str = DEFAULT_TEMPORAL_TZ) -> SeriesT:
    """Inspector-exact render of a pandas datetime column, as an object Series of str/None.

    Reproduces moment's ``'MMM D YYYY, h:mm:ss a z'``: month abbreviation, day and 12-hour hour
    without a leading zero, minute and second with one, lowercase am/pm, zone abbreviation last.
    Midnight and noon are ``12`` rather than ``0``.

    ``tz`` decides which day and hour a timestamp lands on, so it changes what matches, not just
    how a row looks. None marks a row the inspector shows nothing for, which must never match.

    The zone ABBREVIATION comes from the installed tz database, which can disagree with the one a
    browser bundles for historical or contested zones (``Africa/Juba`` reads ``CAST`` here and
    ``EAT`` in moment). Search terms are numeric, so an alphabetic abbreviation cannot be matched
    either way.
    """
    import numpy as np
    import pandas as pd

    if len(s) == 0:
        return pd.Series([], index=s.index, dtype=object)

    localized = (
        s.dt.tz_localize("UTC").dt.tz_convert(tz) if s.dt.tz is None else s.dt.tz_convert(tz)
    )
    present = localized.notna()
    # NaT turns every extracted component into a float, so stand a real timestamp in for it
    localized = localized.fillna(
        localized[present].iloc[0] if present.any() else pd.Timestamp(0, tz="UTC")
    )

    hour_24 = localized.dt.hour
    hour_12 = hour_24 % 12
    hour_12 = hour_12.where(hour_12 != 0, 12)
    meridiem = pd.Series(np.where(hour_24 < 12, "am", "pm"), index=localized.index)
    month = pd.Series(
        np.array(_MONTH_ABBREVS, dtype=object)[localized.dt.month.to_numpy() - 1],
        index=localized.index,
    )

    rendered = (
        month + " " + localized.dt.day.astype(str) + " " + localized.dt.year.astype(str) + ", "
        + hour_12.astype(str) + ":" + localized.dt.minute.astype(str).str.zfill(2)
        + ":" + localized.dt.second.astype(str).str.zfill(2) + " " + meridiem + " "
        + _tz_abbrev_by_offset(localized)
    )
    return pd.Series(rendered, index=s.index, dtype=object).where(present, None)


class CudfTemporalTzUnsupported(NotImplementedError):
    """cuDF cannot name a zone other than UTC; the caller should decline rather than guess."""


def render_datetime_cudf(  # pragma: no cover - cuDF-only; the changed-line-coverage gate has no cuDF lane (validated on dgx)
    s: SeriesT, tz: str = DEFAULT_TEMPORAL_TZ
) -> SeriesT:
    """Inspector render of a cuDF datetime column, GPU-resident throughout.

    Matches :func:`render_datetime_pandas` for ``tz='UTC'``, and raises for any other zone.

    Measured on cudf 26.02.01: ``.dt`` components follow ``tz_convert`` but ``strftime`` does
    not. ``strftime('%Z')`` answers ``UTC`` whatever the zone, and ``%p`` reads the underlying
    UTC instant, so an ``Asia/Kolkata`` column renders the right hour beside the wrong meridiem
    and the wrong zone name. The meridiem is therefore taken from ``.dt.hour``; the zone name has
    no component to take it from, so a non-UTC zone declines instead of rendering a false one.
    ``strftime`` also rejects the no-leading-zero specifiers moment's ``D`` and ``h`` need.
    """
    import cudf

    if tz != "UTC":
        raise CudfTemporalTzUnsupported(
            "cuDF renders every zone abbreviation as UTC, so temporal_tz=%r would produce a "
            "wrong label; use engine='pandas' for non-UTC temporal search" % (tz,)
        )

    if len(s) == 0:
        return cudf.Series([], dtype="object")

    localized = s.dt.tz_localize("UTC") if s.dt.tz is None else s.dt.tz_convert("UTC")
    present = localized.notna()

    hour_24 = localized.dt.hour
    hour_12 = hour_24 % 12
    hour_12 = hour_12.where(hour_12 != 0, 12)
    meridiem = cudf.Series(["am"] * len(s), index=s.index).where(hour_24 < 12, "pm")

    month_num = localized.dt.month
    month = cudf.Series([_MONTH_ABBREVS[0]] * len(s), index=s.index)
    for ordinal, abbrev in enumerate(_MONTH_ABBREVS[1:], start=2):
        month = month.where(month_num != ordinal, abbrev)

    rendered = (
        month + " " + localized.dt.day.astype(str) + " " + localized.dt.year.astype(str)
        + ", " + hour_12.astype(str) + ":" + localized.dt.minute.astype(str).str.zfill(2)
        + ":" + localized.dt.second.astype(str).str.zfill(2) + " " + meridiem + " UTC"
    )
    return rendered.where(present, None)
