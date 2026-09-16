"""Cross-column search kernel (viz-filter L2, panel-algebra D2): OR-across-columns
substring/regex match, dtype-gated AS SEMANTICS — string columns always; integer AND
FLOAT columns iff the term is a numeric literal (inspector gate); date/bool are
auto-gated OUT. Floats are rendered the way the inspector renders them rather than by
``astype(str)`` (see ``wysiwyg.py``): fixed ``precision`` decimals for fractional values,
``String(v)`` for whole ones, and nothing at all for NaN or the Int32 sentinel. pandas
reproduces that exactly; polars/cuDF differ only where the (precision+1)-th decimal is
exactly 5, which is pinned as a known cross-engine divergence (#1695). Per-column matching
delegates to the parity-hardened ``Contains`` predicate, so every pandas/cuDF
quirk and honest decline gate carries over; cuDF regex obeys the same decline
rules as ``=~``."""
import re
from typing import List, Optional

from graphistry.compute.gfql.wysiwyg import DEFAULT_FLOAT_PRECISION
from graphistry.compute.typing import DataFrameT, DType, SeriesT

# inspector's numeric-term gate (streamgl-viz sortAndFilterRowsByQuery.js)
_NUMERIC_TERM_RE = re.compile(r"^[0-9.\-]+$")


def is_numeric_term(term: str) -> bool:
    return bool(_NUMERIC_TERM_RE.match(term))


def _is_searchable_string_dtype(dtype: DType) -> bool:
    import pandas.api.types as pat  # cuDF mirrors the pandas dtype API
    return bool(pat.is_string_dtype(dtype)) or bool(pat.is_object_dtype(dtype))


def _is_int_dtype(dtype: DType) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_integer_dtype(dtype))


def _is_float_dtype(dtype: DType) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_float_dtype(dtype))


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
    string cols always; integer AND float cols iff the term is a numeric literal;
    nested/bool/other skipped — see research/searchany-inspector-parity.md)."""
    if columns is not None:
        return list(columns) if all(c in df.columns for c in columns) else None
    numeric_ok = is_numeric_term(term)
    out: List[str] = []
    for c in df.columns:
        dt = df[c].dtype
        if _has_string_content(df, c):
            out.append(c)
        elif numeric_ok and (_is_int_dtype(dt) or _is_float_dtype(dt)):
            out.append(c)
    return out


def search_any_mask(
    df: DataFrameT,
    term: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
    columns: Optional[List[str]] = None,
    float_precision: int = DEFAULT_FLOAT_PRECISION,
) -> Optional[SeriesT]:
    """Boolean row mask over ``df`` (pandas or cuDF), or None to decline (an explicit
    column is missing). Null cells never match; no candidate columns -> all-False.

    ``float_precision`` is the inspector's fixed decimal count for fractional floats."""
    from graphistry.compute.predicates.str import (
        Contains, _cudf_casefold_or_decline, _cudf_regex_prep,
    )
    cols = search_candidate_columns(df, term, columns)
    if cols is None:
        return None
    if columns is not None and "cudf" in type(df).__module__:
        # Explicit columns= reaches beyond the auto gate. Float is now RENDERED rather than
        # astype(str)'d (wysiwyg.py), so it no longer diverges wildly from pandas and is
        # supported; temporal stringification is still unverified and declines honestly.
        # NB: aliased ``pd_types``, not ``pat`` — ``pat`` is rebound below to the search
        # PATTERN string, and a module/str double-binding in one scope is one statement
        # reorder away from ``AttributeError: 'str' object has no attribute ...``.
        import pandas.api.types as pd_types
        for c in cols:
            dt = df[c].dtype
            if not (_is_searchable_string_dtype(dt) or _is_int_dtype(dt)
                    or _is_float_dtype(dt) or bool(pd_types.is_bool_dtype(dt))):
                raise NotImplementedError(
                    "cuDF searchAny explicit columns support string/int/float/bool dtypes "
                    "only (temporal stringification diverges from pandas); "
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
            # WYSIWYG: match what the inspector DISPLAYS, not repr() — see wysiwyg.py.
            # The renderer emits null where the inspector shows nothing (NaN, sentinel),
            # and Contains(na=False) already refuses to match a null.
            from graphistry.compute.gfql.wysiwyg import render_float_cudf, render_float_pandas
            rendered = (render_float_cudf(s, float_precision)
                        if "cudf" in type(s).__module__
                        else render_float_pandas(s, float_precision))
            m = pred(rendered) & rendered.notna()
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
