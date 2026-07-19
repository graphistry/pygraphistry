from typing import Any, Optional, Tuple, Union, cast
import re

import pandas as pd

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


def _series_supports_str_ops(s: Any) -> bool:
    """True if ``s`` has a usable string accessor (object / pandas ``string`` / categorical-of-str).

    A numeric, temporal, or boolean column does NOT: pandas and cuDF both raise on ``s.str``
    attribute access for those dtypes. Used to make the string predicates value-safe instead of
    surfacing an opaque ``AttributeError: Can only use .str accessor with string values!``.
    """
    try:
        s.str  # the accessor validates dtype on attribute access
    except (AttributeError, TypeError):
        return False
    return True


def _nonstring_null_result(s: Any, na: Optional[bool]) -> Any:
    """Result of a string predicate applied to a NON-string column.

    openCypher treats a string operation over a non-string value as null (a non-string value is
    not a string, so the predicate is unknown → null → excluded from a ``WHERE``). This mirrors the
    established per-cell behavior on an *object* column holding non-strings (those cells already
    become null), and — crucially — NEVER stringifies the column (which would diverge pandas↔cuDF
    and wrongly match, e.g. ``5 CONTAINS '5'``). ``na`` overrides the fill only when the caller
    pinned it (``na=True``/``na=False``).
    """
    fill = None if na is None else na
    n = len(s)
    is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__
    if is_cudf:
        import cudf
        out = cudf.Series([fill] * n)
        try:
            out.index = s.index
        except Exception:
            pass
        return out
    return pd.Series([fill] * n, index=getattr(s, 'index', None))


def _cudf_mask_value(result: Any, mask: Any, value: Any) -> Any:
    try:
        return result.mask(mask, value)
    except Exception:
        try:
            result[mask] = value
            return result
        except Exception:
            try:
                mask_arr = mask.to_pandas().to_numpy()
            except Exception:
                mask_arr = mask
            import cudf
            result_pd = result.to_pandas()
            if value is None:
                result_pd = result_pd.astype('object')
            result_pd.iloc[mask_arr] = value
            return cudf.from_pandas(result_pd)


def _cudf_handle_na(
    result: Any,
    source: Any,
    na: Optional[bool]
) -> Any:
    mask = None
    try:
        mask = source.isna()
        has_mask = bool(mask.any())
    except Exception:
        has_mask = False

    if na is None:
        if not has_mask:
            return result
        return _cudf_mask_value(result, mask, None)

    if isinstance(na, bool):
        if has_mask:
            return _cudf_mask_value(result, mask, na)
        try:
            return result.fillna(na)
        except Exception:
            return result

    return result


def _pandas_handle_na(
    result: pd.Series,
    source: pd.Series,
    na: Optional[bool]
) -> pd.Series:
    mask = source.isna()
    if na is None:
        if mask.any():
            result = result.astype('object')
            result[mask] = None
        return result

    if mask.any():
        result = result.copy()
        result[mask] = na
        if result.dtype == object:
            result = result.infer_objects()
    return result


class Contains(ASTPredicate):
    def __init__(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[bool] = None,
        regex: bool = True
    ) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na
        self.regex = regex

    def __call__(self, s: SeriesT) -> SeriesT:
        if not _series_supports_str_ops(s):
            return _nonstring_null_result(s, self.na)
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        # workaround cuDF not supporting 'case' and 'na' parameters
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.contains/
        if is_cudf:
            if not self.case:
                s_modified = s.str.lower()
                pat_modified = (
                    self.pat.lower()
                    if isinstance(self.pat, str)
                    else self.pat
                )
                result = s_modified.str.contains(
                    pat_modified,
                    regex=self.regex,
                    flags=self.flags
                )
            else:
                result = s.str.contains(
                    self.pat,
                    regex=self.regex,
                    flags=self.flags
                )

            return _cudf_handle_na(result, s, self.na)
        else:
            result = s.str.contains(
                self.pat,
                case=self.case,
                flags=self.flags,
                regex=self.regex
            )
            return _pandas_handle_na(result, s, self.na)

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        if not isinstance(self.pat, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "pat must be string",
                field="pat",
                value=type(self.pat).__name__
            )

        if not isinstance(self.case, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "case must be boolean",
                field="case",
                value=type(self.case).__name__
            )

        if not isinstance(self.flags, int):
            raise GFQLTypeError(
                ErrorCode.E201,
                "flags must be integer",
                field="flags",
                value=type(self.flags).__name__
            )

        if not isinstance(self.na, (bool, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be boolean or None",
                field="na",
                value=type(self.na).__name__
            )

        if not isinstance(self.regex, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "regex must be boolean",
                field="regex",
                value=type(self.regex).__name__
            )


def contains(
    pat: str,
    case: bool = True,
    flags: int = 0,
    na: Optional[bool] = None,
    regex: bool = True
) -> Contains:
    """
    Return whether a given pattern or regex is contained within a string
    """
    return Contains(pat, case, flags, na, regex)


class NeverMatch(ASTPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isna() & False


def never_match() -> NeverMatch:
    return NeverMatch()


class _BoundaryStringPredicate(ASTPredicate):
    def __init__(
        self,
        pat: Union[str, tuple],
        case: bool = True,
        na: Optional[bool] = None
    ) -> None:
        # Convert list to tuple for JSON deserialization compatibility
        self.pat = tuple(pat) if isinstance(pat, list) else pat
        self.case = case
        self.na = na

    def _match_boundary(self, s: SeriesT, pat: Union[str, tuple]) -> SeriesT:
        raise NotImplementedError

    def _compute_result(self, s: SeriesT, is_cudf: bool) -> SeriesT:
        # workaround: pandas and cuDF don't support 'case' parameter
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.startswith/
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/
        # pandas.Series.str.startswith.html
        # pandas.Series.str.endswith.html

        # Handle tuple patterns
        # Workaround for cuDF bug: docs claim tuple support but implementation fails
        # See: https://github.com/rapidsai/cudf/issues/20237
        if isinstance(self.pat, tuple):
            # pandas supports tuples natively (OR logic), cuDF doesn't
            if not is_cudf and self.case:
                # Use pandas native tuple support for case-sensitive
                return self._match_boundary(s, self.pat)
            if not is_cudf and not self.case:
                # pandas tuple with case-insensitive - need workaround
                if len(self.pat) == 0:
                    return pd.Series(False, index=s.index)
                s_lower = s.str.lower()
                patterns_lower = tuple(p.lower() for p in self.pat)
                # Use pandas native tuple support on lowercased data
                return self._match_boundary(s_lower, patterns_lower)

            # cuDF - need manual OR logic (workaround for bug #20237)
            if len(self.pat) == 0:
                import cudf
                # Create False for all values - scalar broadcast, not Python list
                return cudf.Series(False, index=s.index)
            if not self.case:
                s_modified = s.str.lower()
                patterns = [p.lower() for p in self.pat]
            else:
                s_modified = s
                patterns = list(self.pat)
            # Start with first pattern
            result = self._match_boundary(s_modified, patterns[0])
            # OR with remaining patterns
            for pat in patterns[1:]:
                result = result | self._match_boundary(s_modified, pat)
            return result

        if not self.case:
            # Use str.lower() workaround for case-insensitive matching
            s_modified = s.str.lower()
            pat_modified = self.pat.lower()
            return self._match_boundary(s_modified, pat_modified)

        return self._match_boundary(s, self.pat)

    def __call__(self, s: SeriesT) -> SeriesT:
        if not _series_supports_str_ops(s):
            return _nonstring_null_result(s, self.na)
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__
        result = self._compute_result(s, is_cudf)
        if is_cudf:
            return _cudf_handle_na(result, s, self.na)
        return _pandas_handle_na(result, s, self.na)

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        if not isinstance(self.pat, (str, tuple)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "pat must be string or tuple of strings",
                field="pat",
                value=type(self.pat).__name__
            )

        # If tuple, validate all elements are strings
        if isinstance(self.pat, tuple):
            for i, p in enumerate(self.pat):
                if not isinstance(p, str):
                    raise GFQLTypeError(
                        ErrorCode.E201,
                        f"pat tuple element {i} must be string",
                        field="pat",
                        value=type(p).__name__
                    )

        if not isinstance(self.case, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "case must be boolean",
                field="case",
                value=type(self.case).__name__
            )

        if not isinstance(self.na, (bool, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be boolean or None",
                field="na",
                value=type(self.na).__name__
            )


class Startswith(_BoundaryStringPredicate):
    def _match_boundary(self, s: SeriesT, pat: Union[str, tuple]) -> SeriesT:
        return s.str.startswith(pat)


def startswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Startswith:
    """
    Return whether a given pattern or tuple of patterns is at the start of a string.

    :param pat: Pattern (str) or tuple of patterns to match at start of string. When tuple,
        returns True if the string starts with ANY pattern (OR logic).
    :param case: If True, case-sensitive matching (default: True).
    :param na: Fill value for missing values (default: None).
    :returns: Startswith predicate.

    Examples
    --------
    >>> n({"name": startswith("John")})
    >>> n({"name": startswith("john", case=False)})
    >>> n({"filename": startswith(("test_", "demo_"))})
    >>> n({"filename": startswith(("TEST", "DEMO"), case=False)})
    """
    return Startswith(pat, case, na)


class Endswith(_BoundaryStringPredicate):
    def _match_boundary(self, s: SeriesT, pat: Union[str, tuple]) -> SeriesT:
        return s.str.endswith(pat)


def endswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Endswith:
    """
    Return whether a given pattern or tuple of patterns is at the end of a string.

    :param pat: Pattern (str) or tuple of patterns to match at end of string. When tuple,
        returns True if the string ends with ANY pattern (OR logic).
    :param case: If True, case-sensitive matching (default: True).
    :param na: Fill value for missing values (default: None).
    :returns: Endswith predicate.

    Examples
    --------
    >>> n({"email": endswith(".com")})
    >>> n({"email": endswith(".COM", case=False)})
    >>> n({"filename": endswith((".txt", ".csv"))})
    >>> n({"filename": endswith((".TXT", ".CSV"), case=False)})
    """
    return Endswith(pat, case, na)


class _RegexStringPredicate(ASTPredicate):
    def __init__(
        self,
        pat: str,
        case: bool = True,
        flags: int = 0,
        na: Optional[bool] = None
    ) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na

    def _compute_result(self, s: SeriesT, is_cudf: bool) -> SeriesT:
        raise NotImplementedError

    def __call__(self, s: SeriesT) -> SeriesT:
        if not _series_supports_str_ops(s):
            return _nonstring_null_result(s, self.na)
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__
        result = self._compute_result(s, is_cudf)
        if is_cudf:
            return _cudf_handle_na(result, s, self.na)
        return _pandas_handle_na(result, s, self.na)

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

        if not isinstance(self.pat, str):
            raise GFQLTypeError(
                ErrorCode.E201,
                "pat must be string",
                field="pat",
                value=type(self.pat).__name__
            )

        if not isinstance(self.case, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "case must be boolean",
                field="case",
                value=type(self.case).__name__
            )

        if not isinstance(self.flags, int):
            raise GFQLTypeError(
                ErrorCode.E201,
                "flags must be integer",
                field="flags",
                value=type(self.flags).__name__
            )

        if not isinstance(self.na, (bool, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be boolean or None",
                field="na",
                value=type(self.na).__name__
            )


# Lookaround ((?=), (?!), (?<…)) and pattern backreferences (\1-\9, (?P=…)).
# NOT a cuDF version-lag — these need a BACKTRACKING engine, and libcudf's regex is
# non-backtracking (finite-automata, GPU-parallel, linear-time), so they are excluded
# by construction and NO RAPIDS version restores them (verified vs the cuDF 26.02 regex
# feature doc: docs.rapids.ai/api/cudf/stable/libcudf_docs/md_regex/). Same limitation,
# same reason, on polars + polars-gpu (Rust `regex` crate, also non-backtracking) — see
# the twin guard `_regex_rust_incompatible` in lazy/engine/polars/predicates.py. Only
# pandas (Python `re`, backtracking) supports them, so pandas is the honest fallback.
_CUDF_REGEX_UNSUPPORTED = re.compile(r'\(\?=|\(\?!|\(\?<|\\[1-9]|\(\?P[=<]')


def _cudf_regex_prep(pat: str, case: bool) -> Tuple[str, bool]:
    """Adapt a regex for libcudf. Two distinct, both PERMANENT (not version-lag), limits:

    1. Lookaround / backreferences (``_CUDF_REGEX_UNSUPPORTED`` above): architecturally
       impossible on a non-backtracking engine — decline with ``NotImplementedError``.
       polars/polars-gpu share this (Rust regex); pandas is the fallback.
    2. Inline flag groups (``(?i)``, ``(?m)``, ``(?s)`` … at ANY position): unsupported
       inline (26.02 doc verbatim: "The inline (?i...) ignore case format pattern is not
       supported"); flags go via the out-of-band ``flags`` arg, which the Python
       ``str.contains`` binding exposes for MULTILINE/DOTALL only — NOT IGNORECASE (dgx-
       probed; see ``_cudf_casefold_or_decline``). So a leading ``(?i)`` is translated to
       the caller's lowercase-folding workaround (returns the flag-stripped pattern +
       ``case=False``); any other inline flag has no equivalent, so decline honestly.

    Surfaced by the openCypher ``=~`` operator, which lowers ``=~ '(?i)…'`` to
    Match/Fullmatch (viz-filter #1673).
    """
    if _CUDF_REGEX_UNSUPPORTED.search(pat):
        raise NotImplementedError(
            "cuDF regex does not support lookaround or backreferences; use engine='pandas'"
        )
    m = re.match(r'\(\?([aiLmsux]+)\)', pat)
    if not m:
        return pat, case
    flag_chars = m.group(1)
    rest = pat[m.end():]
    if set(flag_chars) <= {'i'}:
        return rest, False  # case-insensitive -> lowercase-folding path
    raise NotImplementedError(
        f"cuDF regex does not support inline flags '(?{flag_chars})'; use engine='pandas'"
    )


def _cudf_casefold_or_decline(pat: str) -> str:
    """Lowercase-fold a pattern for the cuDF case-insensitive regex workaround (data is
    lowercased, so the pattern must be too). This manual fold is the ONLY case-insensitive
    REGEX mechanism cuDF's Python API offers, so the declines below are PERMANENT — not a
    workaround pending a native flag, no RAPIDS version to wait for. Confirmed by dgx probe
    on cudf 26.02.01: ``str.contains``/``match`` reject ``flags=re.IGNORECASE``
    ("unsupported value for flags"; only MULTILINE/DOTALL are accepted) and ``case=False``
    raises "only supported when regex=False". (libcudf's C++ ``regex_flags`` DOES have an
    IGNORECASE bit, but the Python StringMethods binding does not expose it — so it is
    unreachable from ``s.str.contains``.)

    Folding is UNSOUND for exactly three shapes, declined honestly (each a silent wrong
    answer or crash otherwise):
    uppercase escape classes (``.lower()`` turns ``\\D`` into ``\\d`` — INVERTS the
    predicate; dgx-repro'd, wave 1); case-crossing character ranges (``(?i)[A-z]``
    silently narrows, ``[X-b]`` folds to the invalid ``[x-b]``; wave 2); and
    non-ASCII (Python ``str.lower`` vs libcudf lowercasing diverge, e.g. ``İ``).
    Lowercase escapes (``\\d``, ``\\.``, ``\\w``) are ``.lower()`` no-ops and stay
    allowed — they worked before this guard and must not regress to NIE (wave 2)."""
    unsafe = (
        # [A-Z]: uppercase escape classes invert under fold; x: hex escapes can
        # spell uppercase letters invisibly to .lower() ((?i)\\x41 — wave-4).
        re.search(r'\\[A-Zx]', pat) is not None
        or not pat.isascii()
        # Any x-y range where exactly ONE endpoint is an uppercase letter shifts
        # under fold: [A-z] narrows, [?-Z] widens, [X-^] goes invalid (wave-3:
        # letter-letter-only scanning missed the mixed ones). Both-upper shifts
        # consistently ([A-Z]->[a-z]); neither-upper is a fold no-op. Ranges only
        # exist inside classes, so gate on '[' — a class-free literal hyphen
        # ((?i)e-MAIL) keeps folding; in-class false positives DECLINE (safe).
        or ('[' in pat and any(a.isupper() != b.isupper()
                               for a, b in re.findall(r'([!-~])-([!-~])', pat)))
    )
    if unsafe:
        raise NotImplementedError(
            "cuDF case-insensitive regex cannot safely fold this pattern (uppercase "
            "escape class, case-crossing range, or non-ASCII); use engine='pandas'"
        )
    return pat.lower()


class Match(_RegexStringPredicate):
    def _compute_result(self, s: SeriesT, is_cudf: bool) -> SeriesT:
        # workaround cuDF not supporting 'case' and 'na' parameters
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.match/
        if is_cudf:
            pat, case = _cudf_regex_prep(self.pat, self.case)
            if not case:
                return s.str.lower().str.match(_cudf_casefold_or_decline(pat), flags=self.flags)
            return s.str.match(pat, flags=self.flags)

        effective_flags = self.flags
        if not self.case:
            effective_flags |= re.IGNORECASE
        if effective_flags:
            return s.str.match(self.pat, flags=effective_flags)
        return s.str.match(self.pat)


def match(
    pat: str,
    case: bool = True,
    flags: int = 0,
    na: Optional[bool] = None
) -> Match:
    """
    Return whether a given pattern is at the start of a string
    """
    return Match(pat, case, flags, na)


class Fullmatch(_RegexStringPredicate):
    def _compute_result(self, s: SeriesT, is_cudf: bool) -> SeriesT:
        if is_cudf:
            # cuDF has no fullmatch; emulate with match() + ``^(…)$`` anchors. The
            # group wrap makes a top-level alternation anchor as a WHOLE (``ab|cd``
            # must not become ``^ab|cd$``, which matched 'abXXX' — dgx-repro'd);
            # libcudf lacks ``(?:``, so a plain capture group (match is boolean —
            # numbering is irrelevant). libcudf also rejects inline flag groups
            # (``(?i)`` …) entirely, so translate them first (``(?i)`` ->
            # lowercase-folding; others NIE). Surfaced by openCypher ``=~`` (#1673).
            pat, case = _cudf_regex_prep(self.pat, self.case)
            anchored_pat = f'^({pat})$'
            if not case:
                return s.str.lower().str.match(
                    _cudf_casefold_or_decline(anchored_pat), flags=self.flags)
            return s.str.match(anchored_pat, flags=self.flags)

        # pandas has native fullmatch support
        effective_flags = self.flags
        if not self.case:
            effective_flags |= re.IGNORECASE
        if effective_flags:
            return s.str.fullmatch(self.pat, flags=effective_flags)
        return s.str.fullmatch(self.pat)


def fullmatch(
    pat: str,
    case: bool = True,
    flags: int = 0,
    na: Optional[bool] = None
) -> Fullmatch:
    """
    Return whether a given pattern matches the entire string

    Unlike match() which matches from the start, fullmatch() requires the
    pattern to match the entire string. This is useful for exact validation
    of formats like emails, phone numbers, or IDs.

    Args:
        pat: Regular expression pattern to match against entire string
        case: If True, case-sensitive matching (default: True)
        flags: Regex flags (e.g., re.IGNORECASE, re.MULTILINE)
        na: Fill value for missing values (default: None)

    Returns:
        Fullmatch predicate

    Examples:
        >>> # Exact digit match
        >>> n({"code": fullmatch(r"\\d{3}")})  # Matches "123" but not "123abc"
        >>>
        >>> # Case-insensitive email validation
        >>> n({"email": fullmatch(r"[a-z]+@[a-z]+\\.com", case=False)})
        >>>
        >>> # With regex flags
        >>> import re
        >>> n({"id": fullmatch(r"[A-Z]{3}-\\d{4}", flags=re.IGNORECASE)})
    """
    return Fullmatch(pat, case, flags, na)


class _CallablePredicate(ASTPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        raise NotImplementedError()

    def __call__(self, s: SeriesT) -> SeriesT:
        return cast(SeriesT, type(self).predicate(s))


class IsNumeric(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isnumeric()


def isnumeric() -> IsNumeric:
    """
    Return whether a given string is numeric
    """
    return IsNumeric()


class IsAlpha(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isalpha()


def isalpha() -> IsAlpha:
    """
    Return whether a given string is alphabetic
    """
    return IsAlpha()


class IsDigit(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isdigit()


def isdigit() -> IsDigit:
    """
    Return whether a given string is numeric
    """
    return IsDigit()


class IsLower(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.islower()


def islower() -> IsLower:
    """
    Return whether a given string is lowercase
    """
    return IsLower()


class IsUpper(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isupper()


def isupper() -> IsUpper:
    """
    Return whether a given string is uppercase
    """
    return IsUpper()


class IsSpace(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isspace()


def isspace() -> IsSpace:
    """
    Return whether a given string is whitespace
    """
    return IsSpace()


class IsAlnum(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isalnum()


def isalnum() -> IsAlnum:
    """
    Return whether a given string is alphanumeric
    """
    return IsAlnum()


class IsDecimal(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.isdecimal()


def isdecimal() -> IsDecimal:
    """
    Return whether a given string is decimal
    """
    return IsDecimal()


class IsTitle(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.str.istitle()


def istitle() -> IsTitle:
    """
    Return whether a given string is title case
    """
    return IsTitle()


class IsNull(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.isnull()


def isnull() -> IsNull:
    """
    Return whether a given string is null
    """
    return IsNull()


class NotNull(_CallablePredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.notnull()


def notnull() -> NotNull:
    """
    Return whether a given string is not null
    """
    return NotNull()
