"""Cross-column search kernel (viz-filter L2, panel-algebra D2): OR-across-columns
substring/regex match, dtype-gated AS SEMANTICS — string columns always; integer
columns iff the term is a numeric literal (inspector gate); float/date/bool are
auto-gated OUT (float stringification is engine-divergent — explicit ``columns=``
reaches bool on both engines, float/date on pandas only: cuDF declines them, its
astype(str) rendering diverges from pandas — dgx-probed). Per-column matching
delegates to the parity-hardened ``Contains`` predicate, so every pandas/cuDF
quirk and honest decline gate carries over; cuDF regex obeys the same decline
rules as ``=~``."""
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


def _canonical_float_str(s: SeriesT, precision: int = 4) -> SeriesT:
    """WYSIWYG float render matching the streamgl-viz inspector's search text
    (``defaultFormat`` number/integer branch): a WHOLE value (``v % 1 == 0``,
    incl. whole floats like ``5.0``) renders as a bare integer string
    (``"5"``/``"-3"`` — no decimals, no exponent, matching JS ``String()``); a
    FRACTIONAL value renders fixed to ``precision`` decimals (``0.12345`` ->
    ``"0.1235"``, matching ``sprintf('%.4f')``).

    The naive ``astype(str)`` was excluded because it diverges cross-engine in
    the exponent regime (``1e16`` -> ``'1e16'`` vs ``'1e+16'``) and at
    half-boundaries (true-float-bit ``f"{v:.4f}"`` vs a decimal render). We route
    BOTH branches through an exponent-free, decimal-based render so pandas /
    cuDF / polars agree byte-for-byte; pandas is the oracle. Null cells render as
    the empty string here and are masked out by the caller (null never matches).
    """
    is_cudf = "cudf" in type(s).__module__
    notna = s.notna()
    # whole vs fractional split (nulls treated as non-whole; masked out anyway)
    whole = (s == s.round(0)) & notna  # type: ignore[operator]
    if is_cudf:
        # fractional: fixed-`precision` decimal via Decimal128 (exponent-free,
        # GPU-native, half-even like pandas' decimal round); whole: int string.
        frac_txt = s.round(precision).astype(  # type: ignore[union-attr]
            f"decimal128[38,{precision}]"
        ).astype("str")
        whole_txt = s.round(0).astype("int64").astype("str")  # type: ignore[union-attr]
        out = whole_txt.where(whole, frac_txt)
        return out.where(notna, "")  # type: ignore[union-attr]
    # pandas
    import numpy as np
    vals = s.to_numpy()
    frac = np.round(vals.astype("float64"), precision)
    # fixed-`precision` decimal render (no exponent) for fractional values
    frac_txt = np.array([("%.*f" % (precision, v)) for v in frac], dtype=object)
    # whole values -> integer string (round to kill float noise, then int)
    whole_arr = np.asarray(whole)
    whole_txt = np.where(
        whole_arr,
        np.array([str(int(round(v))) if not (v != v) else "" for v in vals], dtype=object),
        "",
    )
    result = np.where(whole_arr, whole_txt, frac_txt)
    result = np.where(np.asarray(notna), result, "")
    return cast_series_like(s, [str(x) for x in result])


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


def search_any_mask(
    df: DataFrameT,
    term: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
    columns: Optional[List[str]] = None,
) -> Optional[SeriesT]:
    """Boolean row mask over ``df`` (pandas or cuDF), or None to decline (an explicit
    column is missing). Null cells never match; no candidate columns -> all-False."""
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
        if not _is_searchable_string_dtype(s.dtype):
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
