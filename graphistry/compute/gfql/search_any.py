"""Cross-column search kernel (viz-filter L2, panel-algebra D2): OR-across-columns
substring/regex match, dtype-gated AS SEMANTICS — string columns always; integer
columns iff the term is a numeric literal (inspector gate); float/date/bool are
auto-gated OUT (float stringification is engine-divergent — reach them via explicit
``columns=``). Per-column matching delegates to the parity-hardened ``Contains``
predicate, so every pandas/cuDF quirk and honest decline gate carries over."""
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
    from graphistry.compute.predicates.str import Contains
    cols = search_candidate_columns(df, term, columns)
    if cols is None:
        return None
    if not cols or len(df) == 0:
        if len(df.columns) == 0:
            return None
        return df[df.columns[0]].isna() & False  # engine-safe all-False
    pred = Contains(term, case=case_sensitive, regex=regex, na=False)
    mask: Optional[Any] = None
    for c in cols:
        s = df[c]
        if not _is_searchable_string_dtype(s.dtype):
            s = s.astype(str)  # canonical toString for int / explicit columns
        m = pred(s)
        mask = m if mask is None else (mask | m)
    assert mask is not None  # cols is non-empty here
    return mask.fillna(False)
