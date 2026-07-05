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
from typing import Any, List, Optional

# inspector's numeric-term gate (streamgl-viz sortAndFilterRowsByQuery.js)
_NUMERIC_TERM_RE = re.compile(r"^[0-9.\-]+$")


def is_numeric_term(term: str) -> bool:
    return bool(_NUMERIC_TERM_RE.match(term))


def _is_searchable_string_dtype(dtype: Any) -> bool:
    import pandas.api.types as pat  # cuDF mirrors the pandas dtype API
    return bool(pat.is_string_dtype(dtype)) or bool(pat.is_object_dtype(dtype))


def _is_int_dtype(dtype: Any) -> bool:
    import pandas.api.types as pat
    return bool(pat.is_integer_dtype(dtype))


def search_candidate_columns(
    df: Any, term: str, columns: Optional[List[str]]
) -> Optional[List[str]]:
    """Columns to search: the explicit list (None if any is missing — caller declines
    loudly) or the auto dtype gate."""
    if columns is not None:
        return list(columns) if all(c in df.columns for c in columns) else None
    numeric_ok = is_numeric_term(term)
    out: List[str] = []
    for c in df.columns:
        dt = df[c].dtype
        if _is_searchable_string_dtype(dt):
            out.append(c)
        elif numeric_ok and _is_int_dtype(dt):
            out.append(c)
    return out


def search_any_mask(
    df: Any,
    term: str,
    *,
    case_sensitive: bool = False,
    regex: bool = False,
    columns: Optional[List[str]] = None,
) -> Optional[Any]:
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
    mask: Optional[Any] = None
    for c in cols:
        s = df[c]
        m: Any
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
