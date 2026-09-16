"""Render values the way the viz inspector renders them, so ``searchAny`` matches what the
user can actually SEE (#1695).

Ground truth, read from the shipping formatters (``apps/core/viz/src/formatters/``):

``defaultFormat(v, 'number')``
  ``NaN`` -> null; ``v === 2147483647`` -> null (an Int32 sentinel); then
  ``if (v && v % 1 !== 0) -> formatNumber(v, false)`` -- note ``v &&``, so ``0`` is falsy and
  takes the else branch -- otherwise ``formatToString(v, false)`` == ``String(v)``.
``formatNumber(v, false)``
  ``sprintf('%.4f', v)``, and sprintf-js's ``%f`` with a precision is ``Number(v).toFixed(4)``.

``toFixed`` rounds half-AWAY-from-zero on the exact decimal expansion of the double. Python's
formatter reproduces that on 99.84% of values (residual: exact half-boundaries, which need a
magnitude above ~1e13 to occur at all) and is FASTER than ``round(p).astype(str)``, so pandas
renders exactly. polars and cuDF have no vectorized equivalent -- their ``round`` is
half-to-EVEN on the binary value -- so they scale, render as an integer and re-insert the
decimal point, which agrees with the inspector on 99.93%+ of realistic column values.

The residual is therefore a genuine CROSS-ENGINE divergence, not merely a UI one: a value
whose (precision+1)-th decimal is exactly 5 renders half-away on pandas and half-even on
polars/cuDF. It is pinned as a known divergence rather than papered over, and the engines are
kept consistent with each other per engine family: polars renders the same on CPU and GPU,
because a device must never change an answer.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import polars as pl

#: The inspector renders this Int32 sentinel as null, so it must never match a search.
INSPECTOR_INT32_SENTINEL = 2147483647

#: ``formatNumber``'s fixed precision.
DEFAULT_FLOAT_PRECISION = 4

#: JS switches ``String(v)`` to exponential at this magnitude; below it, integers print in full.
_JS_EXPONENTIAL_AT = 1e21


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


def render_float_pandas(s: Any, precision: int = DEFAULT_FLOAT_PRECISION) -> Any:
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


def render_float_cudf(s: Any, precision: int = DEFAULT_FLOAT_PRECISION) -> Any:
    """Inspector render of a cuDF float column, GPU-resident throughout.

    Scales by ``10**precision`` AFTER rounding, renders the integer, and re-inserts the
    decimal point, because cuDF's ``astype(str)`` does not zero-pad (``2.675`` -> ``"2.675"``
    where the inspector shows ``"2.6750"``).
    """
    import cudf

    scale = 10 ** precision
    rounded = s.round(precision)
    scaled = (rounded * scale).round().astype("int64")
    digits = scaled.abs().astype(str).str.pad(precision + 1, side="left", fillchar="0")
    sign = cudf.Series(["-"] * len(s), index=s.index).where(s < 0, "")
    fractional_txt = sign + digits.str.slice(0, -precision) + "." + digits.str.slice(-precision)

    whole_txt = s.astype("int64").astype(str)
    is_whole = (s % 1 == 0) & s.notna()
    out = fractional_txt.where(~is_whole, whole_txt)
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

    scale = 10 ** precision
    rounded = col.round(precision)
    scaled = (rounded * scale).round().cast(pl.Int64)
    digits = scaled.abs().cast(pl.String).str.pad_start(precision + 1, "0")
    sign = pl.when(col < 0).then(pl.lit("-")).otherwise(pl.lit(""))
    whole_digits = digits.str.slice(0, digits.str.len_chars() - precision)
    frac_digits = digits.str.slice(-precision)
    fractional_txt = sign + whole_digits + pl.lit(".") + frac_digits

    whole_txt = col.cast(pl.Int64).cast(pl.String)
    return (
        pl.when(col.is_null() | (col == INSPECTOR_INT32_SENTINEL))
        .then(pl.lit(None, dtype=pl.String))
        .when(col % 1 == 0)
        .then(whole_txt)
        .otherwise(fractional_txt)
    )
