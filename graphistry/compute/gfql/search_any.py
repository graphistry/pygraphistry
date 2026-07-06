"""Cross-column search kernel (viz-filter L2, panel-algebra D2): OR-across-columns
substring/regex match, dtype-gated AS SEMANTICS — string columns always; integer
columns iff the term is a numeric literal (inspector gate); float/date/bool are
auto-gated OUT (kept symmetric across engines for cross-engine parity — #1695
decision A). Explicit ``columns=`` reaches bool on both engines and float/datetime
on **pandas only**: pandas renders floats to the inspector's WYSIWYG search text
via ``_canonical_float_str`` (``%.4f``-direct, dgx-verified byte-identical to the
viz ``sprintf-js``) and datetimes via ``_canonical_datetime_str`` (the inspector's
moment date format, localized to a caller ``tz``); cuDF/polars honestly decline
both (their native decimal / datetime render can't reproduce pandas — cuDF float
truncates, polars diverges at ties — and a per-element UDF would be a host-bridge;
dgx-probed 2026-07-05). Per-column matching delegates to the parity-hardened
``Contains`` predicate, so every pandas/cuDF quirk and honest decline gate carries
over; cuDF regex obeys the same decline rules as ``=~``."""
import re
from typing import List, Optional

from graphistry.compute.typing import DataFrameT, SeriesT

# inspector's numeric-term gate (streamgl-viz sortAndFilterRowsByQuery.js)
_NUMERIC_TERM_RE = re.compile(r"^[0-9.\-]+$")


def is_numeric_term(term: str) -> bool:
    return bool(_NUMERIC_TERM_RE.match(term))


def _is_searchable_string_dtype(dtype: object) -> bool:
    import pandas.api.types as pat  # cuDF mirrors the pandas dtype API
    return bool(pat.is_string_dtype(dtype)) or bool(pat.is_object_dtype(dtype))


def _is_int_dtype(dtype: object) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_integer_dtype(dtype))


def _is_float_dtype(dtype: object) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_float_dtype(dtype))


def _is_datetime_dtype(dtype: object) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_datetime64_any_dtype(dtype))


# The inspector's default date render (moment `'MMM D YYYY, h:mm:ss a z'`) expressed
# as its strftime equivalent — moment MMM=%b, D=%-d (no leading zero), YYYY=%Y,
# h=%-I (12h no leading zero), mm=%M, ss=%S, a=am/pm (lowercase), z=%Z (tz abbrev).
_INSPECTOR_TEMPORAL_STRFTIME = "%b %-d %Y, %-I:%M:%S %p %Z"


def _canonical_datetime_str(
    s: SeriesT, temporal_format: Optional[str] = None, tz: str = "UTC"
) -> SeriesT:
    """WYSIWYG datetime render matching the streamgl-viz inspector's date search text
    (``formatDate`` -> moment ``'MMM D YYYY, h:mm:ss a z'``), LOCALIZED to ``tz``.

    The inspector formats a date in the VIEWER's timezone (``moment.unix`` local),
    which is not knowable server-side — so ``tz`` is a caller knob (Leo's
    localization call-param): pass the viewer's zone (e.g. ``'America/Los_Angeles'``)
    for byte parity with what that viewer sees. Default ``'UTC'`` is chosen for
    DETERMINISM (a server-local default would vary by deployment and break the
    parity oracle); flip via ``tz=``.

    PANDAS-ONLY (like the float render): cuDF/polars decline datetime search — their
    native datetime->string rendering and tz handling diverge from pandas strftime,
    and a per-cell UDF would be a host-bridge. Null cells render ``""`` (masked out).

    ``temporal_format`` is a **strftime** pattern; its default reproduces the
    inspector's moment default. moment's lowercase am/pm (``a``) is reproduced by
    lowercasing the ``%p`` output (tz abbrevs / month names never contain AM/PM).
    """
    import pandas as pd
    fmt = temporal_format or _INSPECTOR_TEMPORAL_STRFTIME
    ser = s if isinstance(s, pd.Series) else pd.Series(s)
    # localize: naive columns are assumed UTC then converted; tz-aware are converted.
    if getattr(ser.dtype, "tz", None) is not None:
        localized = ser.dt.tz_convert(tz)
    else:
        localized = ser.dt.tz_localize("UTC").dt.tz_convert(tz)
    notna = ser.notna()
    txt = localized.dt.strftime(fmt)
    if "%p" in fmt:
        # moment `a` is lowercase am/pm; strftime %p is upper. Lowercase ONLY the
        # standalone AM/PM token (word-bounded) so we don't corrupt an alpha tz
        # abbreviation (e.g. America/Manaus -> "AMT") or literal text in a custom
        # temporal_format — a blind global replace turned "AMT" into "amT".
        txt = txt.str.replace(r"\bAM\b", "am", regex=True).str.replace(r"\bPM\b", "pm", regex=True)
    return txt.where(notna, "")


def _canonical_float_str(s: SeriesT, precision: int = 4) -> SeriesT:
    """WYSIWYG float render matching the streamgl-viz inspector's search text —
    verified byte-identical to the inspector's real ``sprintf-js`` ``%.4f`` on
    dgx-spark (float parity fuzzer, 2026-07-05).

    - WHOLE value (``v % 1 == 0``, incl. whole floats like ``5.0``, ``|v| < 1e21``)
      -> bare integer string (``"5"``/``"-3"`` — no decimals, no exponent;
      matches JS ``String(v)`` / ``formatToString``).
    - FRACTIONAL value -> ``"%.<precision>f" % v`` applied DIRECTLY to the raw
      double (single correctly-rounded conversion; matches ``sprintf('%.4f', v)``,
      e.g. ``0.12345`` -> ``"0.1235"``, ``-3.74825`` -> ``"-3.7483"``).

    PANDAS-ONLY. The GPU engines cannot reproduce this natively and honestly
    decline float search upstream (dgx-proven: cuDF ``float64 -> decimal128``
    TRUNCATES — wrong on generic values, not just ties; polars' decimal cast
    rounds differently at 5th-decimal ties and drops the sign on ``-0.0000``;
    there is no native GPU ``printf`` and a per-element UDF would be a forbidden
    host-bridge). So this is only ever called on a pandas float column.

    NOTE: an earlier draft pre-rounded with ``np.round(v, precision)`` before
    ``%.*f`` — that DOUBLE-ROUNDS and mis-renders half-ties (``-3.74825`` ->
    ``-3.7482`` instead of the browser's ``-3.7483``). We round exactly once, in
    ``%.*f``, on the raw double. Null cells render as ``""`` and never match
    (the caller masks them out).
    """
    import numpy as np
    notna = np.asarray(s.notna())
    vals = np.asarray(s.to_numpy(dtype="float64", na_value=np.nan))
    out = np.empty(len(vals), dtype=object)
    for i in range(len(vals)):
        v = vals[i]
        if not notna[i] or v != v:  # null / NaN -> "" (masked out by caller)
            out[i] = ""
        elif v == np.inf:  # JS String(Infinity)/sprintf('%.4f',Inf) == "Infinity"
            out[i] = "Infinity"
        elif v == -np.inf:
            out[i] = "-Infinity"
        elif v % 1 == 0:
            # whole -> plain integer digits (JS String); the >= 1e21 JS-exponent
            # regime is unsupported (astronomically rare for a search term) -> "".
            # (Whole floats >= 2**53 also fall here: str(int(v)) is the exact-integer
            # decimal, which can differ from JS String()'s shortest round-trip for a
            # few magnitudes — a documented, astronomically-rare search limitation.)
            out[i] = str(int(v)) if abs(v) < 1e21 else ""
        else:
            out[i] = "%.*f" % (precision, v)  # printf on the raw double == sprintf-js
    return cast_series_like(s, out.tolist())


def cast_series_like(template: SeriesT, data: list) -> SeriesT:
    import pandas as pd
    return pd.Series(data, index=template.index, dtype="object")


def _has_string_content(df: DataFrameT, c: object) -> bool:
    """True iff column ``c`` holds STRINGS (a real string dtype, or an object column whose
    values are actually strings). An object column of lists/dicts/mixed is NOT string
    content — the streamgl-viz inspector skips such columns (``shouldSearch`` only fires on
    ``dataType === 'string'``), so the auto gate skips them too rather than include-then-
    silently-never-match on a ``Contains``-over-lists path (viz-filter searchAny 2a)."""
    import pandas.api.types as pat
    s = df[c]
    is_cudf = "cudf" in type(s).__module__
    # The list/dict ambiguity is PANDAS-only (numpy `object` can hold arbitrary python
    # objects); cuDF/polars use typed columns (a list is a typed List, not object), and
    # infer_dtype does NOT accept a cuDF Series — so only inspect contents for numpy-object.
    if not is_cudf and s.dtype == object:
        try:
            return pat.infer_dtype(s, skipna=True) in ("string", "empty")
        except Exception:
            return False
    return bool(pat.is_string_dtype(s.dtype))  # StringDtype / cuDF str; cuDF List -> False


def search_candidate_columns(
    df: DataFrameT, term: str, columns: Optional[List[str]]
) -> Optional[List[str]]:
    """Columns to search: the explicit list (None if any is missing — caller declines
    loudly) or the auto dtype gate (mirrors the streamgl-viz inspector's ``shouldSearch``:
    string cols always; integer cols iff the term is a numeric literal; nested/bool/other
    skipped — see research/searchany-inspector-parity.md)."""
    if columns is not None:
        return list(columns) if all(c in df.columns for c in columns) else None
    numeric_ok = is_numeric_term(term)
    out: List[str] = []
    for c in df.columns:
        dt = df[c].dtype
        if _has_string_content(df, c):
            out.append(c)
        elif numeric_ok and _is_int_dtype(dt):
            out.append(c)
    return out


def validate_format_opts(
    float_precision: int, temporal_format: Optional[str], tz: str
) -> None:
    """Validate the #1695 WYSIWYG format options at the kernel — the SINGLE choke
    point every surface passes through (cypher op, ast op, and the python twins,
    which bypass the call safelist). Consistent ``GFQLValidationError`` regardless
    of entry point, instead of a downstream ``"%.*f" % -1`` silent clamp or an
    opaque ``tz_convert('')`` IndexError."""
    from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
    if not isinstance(float_precision, int) or isinstance(float_precision, bool) or float_precision < 0:
        raise GFQLValidationError(
            ErrorCode.E108, "searchAny floatPrecision must be a non-negative integer",
            field="float_precision", value=float_precision,
            suggestion="Use an integer >= 0 (default 4).")
    if temporal_format is not None and (not isinstance(temporal_format, str) or temporal_format == ""):
        raise GFQLValidationError(
            ErrorCode.E108, "searchAny temporalFormat must be a non-empty strftime string",
            field="temporal_format", value=temporal_format,
            suggestion="Omit for the inspector default, or pass a non-empty strftime pattern.")
    if not isinstance(tz, str) or tz == "":
        raise GFQLValidationError(
            ErrorCode.E108, "searchAny tz must be a non-empty timezone string",
            field="tz", value=tz, suggestion="Use an IANA zone like 'UTC' or 'America/Los_Angeles'.")


def search_any_mask(
    df: DataFrameT,
    term: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
    columns: Optional[List[str]] = None,
    float_precision: int = 4,
    temporal_format: Optional[str] = None,
    tz: str = "UTC",
) -> Optional[SeriesT]:
    """Boolean row mask over ``df`` (pandas or cuDF), or None to decline (an explicit
    column is missing). Null cells never match; no candidate columns -> all-False."""
    validate_format_opts(float_precision, temporal_format, tz)
    from graphistry.compute.predicates.str import (
        Contains, _cudf_casefold_or_decline, _cudf_regex_prep,
    )
    cols = search_candidate_columns(df, term, columns)
    if cols is None:
        return None
    if columns is not None and "cudf" in type(df).__module__:
        # Explicit columns= reaches beyond the auto gate; cuDF's astype(str) float
        # rendering DIVERGES from pandas (dgx-probed: 0.1+0.2 -> '0.3' vs
        # '0.30000000000000004'; 1e16 -> '1.0e+16' vs '1e+16'; long mantissas
        # truncate) and temporal is unverified — decline honestly rather than
        # silently mismatch the pandas oracle (wave-3 W3-1). string/int/bool render
        # identically (bool parity is pinned) and stay native.
        import pandas.api.types as pat
        for c in cols:
            dt = df[c].dtype
            if not (_is_searchable_string_dtype(dt) or _is_int_dtype(dt)
                    or bool(pat.is_bool_dtype(dt))):
                raise NotImplementedError(
                    "cuDF searchAny explicit columns support string/int/bool dtypes "
                    "only (float/temporal stringification diverges from pandas); "
                    "use engine='pandas'"
                )
    if not cols or len(df) == 0:
        if len(df.columns) == 0:
            return None
        return df[df.columns[0]].isna() & False  # engine-safe all-False
    pat, case = term, case_sensitive
    if regex and "cudf" in type(df).__module__:
        # Same decline rules as =~ (Match/Fullmatch): NIE on lookaround/backrefs/
        # inline flags instead of a libcudf crash, and refuse unsound casefolds
        # instead of Contains' blind pat.lower() (\D -> \d INVERTS) — wave-1 B2.
        pat, case = _cudf_regex_prep(pat, case)
        if not case:
            pat = _cudf_casefold_or_decline(pat)  # pre-folded; Contains' .lower() is a no-op
    pred = Contains(pat, case=case, regex=regex, na=False)
    mask: Optional[SeriesT] = None
    for c in cols:
        s = df[c]
        m: SeriesT
        if _is_float_dtype(s.dtype):
            # FLOAT (pandas-only; cuDF float declined above, polars uses its own
            # lowering): render to the inspector's WYSIWYG search text (%.4f-direct
            # / whole->int-string, dgx-verified byte-identical to sprintf-js) rather
            # than astype(str), whose exponent/half-tie rendering diverges (#1695).
            nulls = s.isna()
            m = pred(_canonical_float_str(s, float_precision)) & ~nulls
        elif _is_datetime_dtype(s.dtype):
            # DATETIME (pandas-only; cuDF/polars declined above/in their lowering):
            # render the inspector's moment date format localized to `tz` (#1695).
            nulls = s.isna()
            m = pred(_canonical_datetime_str(s, temporal_format, tz)) & ~nulls
        elif not _is_searchable_string_dtype(s.dtype):
            # canonical toString for int / explicit columns; pandas astype(str)
            # stringifies nulls ("nan"/"<NA>") so mask them back out — null cells
            # never match on any engine (wave-1 I1)
            nulls = s.isna()
            m = pred(s.astype(str)) & ~nulls
        else:
            m = pred(s)
        mask = m if mask is None else (mask | m)
    assert mask is not None  # cols is non-empty here
    return mask.fillna(False)
