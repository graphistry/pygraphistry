from typing import Optional, Union

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


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

            if self.na is not None and isinstance(self.na, bool):
                result = result.fillna(self.na)

            return result
        else:
            return s.str.contains(
                self.pat,
                self.case,
                self.flags,
                self.na,
                self.regex
            )

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


class Startswith(ASTPredicate):
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

    def __call__(self, s: SeriesT) -> SeriesT:
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        # workaround: pandas and cuDF don't support 'case' parameter
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.startswith/
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/
        # pandas.Series.str.startswith.html

        # Handle tuple patterns
        # Workaround for cuDF bug: docs claim tuple support but implementation fails
        # See: https://github.com/rapidsai/cudf/issues/20237
        if isinstance(self.pat, tuple):
            # pandas supports tuples natively (OR logic), cuDF doesn't
            if not is_cudf and self.case:
                # Use pandas native tuple support for case-sensitive
                result = s.str.startswith(self.pat)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
            elif not is_cudf and not self.case:
                # pandas tuple with case-insensitive - need workaround
                if len(self.pat) == 0:
                    import pandas as pd
                    # Create False for all values
                    result = pd.Series([False] * len(s), index=s.index)
                    # Preserve NA values when na=None (default)
                    if self.na is None:
                        result = result.astype(object)
                        result[s.isna()] = None
                else:
                    s_lower = s.str.lower()
                    patterns_lower = tuple(p.lower() for p in self.pat)
                    # Use pandas native tuple support on lowercased data
                    result = s_lower.str.startswith(patterns_lower)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
            else:
                # cuDF - need manual OR logic (workaround for bug #20237)
                if len(self.pat) == 0:
                    import cudf
                    import pandas as pd
                    # Create False for all values
                    result = cudf.Series([False] * len(s), index=s.index)
                    # Preserve NA values when na=None (default) - match pandas behavior
                    if self.na is None:
                        # cuDF bool dtype can't hold None, so check if we need object dtype
                        has_na: bool = bool(s.isna().any())
                        if has_na:
                            # Convert to object dtype to preserve None values
                            result_pd = result.to_pandas().astype('object')  # type: ignore[operator]
                            result_pd[s.to_pandas().isna()] = None
                            result = cudf.from_pandas(result_pd)
                else:
                    if not self.case:
                        s_modified = s.str.lower()
                        patterns = [p.lower() for p in self.pat]
                    else:
                        s_modified = s
                        patterns = list(self.pat)
                    # Start with first pattern
                    result = s_modified.str.startswith(patterns[0])
                    # OR with remaining patterns
                    for pat in patterns[1:]:
                        result = result | s_modified.str.startswith(pat)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
        elif not self.case:
            # Use str.lower() workaround for case-insensitive matching
            s_modified = s.str.lower()
            pat_modified = self.pat.lower()
            result = s_modified.str.startswith(pat_modified)
        else:
            result = s.str.startswith(self.pat)

        # Handle na parameter for non-tuple cases
        if is_cudf:
            # cuDF doesn't support na parameter, use fillna
            if self.na is not None:
                return result.fillna(self.na)
            else:
                return result
        else:
            # pandas supports na parameter for case-sensitive str patterns
            if not self.case:
                if self.na is not None:
                    return result.fillna(self.na)
                else:
                    return result
            else:
                return s.str.startswith(self.pat, self.na)

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


def startswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Startswith:
    """
    Return whether a given pattern or tuple of patterns is at the
    start of a string

    Args:
        pat: Pattern (str) or tuple of patterns to match at start of
             string. When tuple, returns True if string starts with ANY
             pattern (OR logic)
        case: If True, case-sensitive matching (default: True)
        na: Fill value for missing values (default: None)

    Returns:
        Startswith predicate

    Examples:
        >>> # Single pattern, case-sensitive (default)
        >>> n({"name": startswith("John")})
        >>> # Single pattern, case-insensitive
        >>> n({"name": startswith("john", case=False)})
        >>> # Multiple patterns (OR logic)
        >>> n({"filename": startswith(("test_", "demo_"))})
        >>> # Multiple patterns, case-insensitive
        >>> n({"filename": startswith(("TEST", "DEMO"), case=False)})
    """
    return Startswith(pat, case, na)


class Endswith(ASTPredicate):
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

    def __call__(self, s: SeriesT) -> SeriesT:
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        # workaround: pandas and cuDF don't support 'case' parameter
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.endswith/
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/
        # pandas.Series.str.endswith.html

        # Handle tuple patterns
        # Workaround for cuDF bug: docs claim tuple support but implementation fails
        # See: https://github.com/rapidsai/cudf/issues/20237
        if isinstance(self.pat, tuple):
            # pandas supports tuples natively (OR logic), cuDF doesn't
            if not is_cudf and self.case:
                # Use pandas native tuple support for case-sensitive
                result = s.str.endswith(self.pat)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
            elif not is_cudf and not self.case:
                # pandas tuple with case-insensitive - need workaround
                if len(self.pat) == 0:
                    import pandas as pd
                    # Create False for all values
                    result = pd.Series([False] * len(s), index=s.index)
                    # Preserve NA values when na=None (default)
                    if self.na is None:
                        result = result.astype(object)
                        result[s.isna()] = None
                else:
                    s_lower = s.str.lower()
                    patterns_lower = tuple(p.lower() for p in self.pat)
                    # Use pandas native tuple support on lowercased data
                    result = s_lower.str.endswith(patterns_lower)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
            else:
                # cuDF - need manual OR logic (workaround for bug #20237)
                if len(self.pat) == 0:
                    import cudf
                    import pandas as pd
                    # Create False for all values
                    result = cudf.Series([False] * len(s), index=s.index)
                    # Preserve NA values when na=None (default) - match pandas behavior
                    if self.na is None:
                        # cuDF bool dtype can't hold None, so check if we need object dtype
                        has_na: bool = bool(s.isna().any())
                        if has_na:
                            # Convert to object dtype to preserve None values
                            result_pd = result.to_pandas().astype('object')  # type: ignore[operator]
                            result_pd[s.to_pandas().isna()] = None
                            result = cudf.from_pandas(result_pd)
                else:
                    if not self.case:
                        s_modified = s.str.lower()
                        patterns = [p.lower() for p in self.pat]
                    else:
                        s_modified = s
                        patterns = list(self.pat)
                    # Start with first pattern
                    result = s_modified.str.endswith(patterns[0])
                    # OR with remaining patterns
                    for pat in patterns[1:]:
                        result = result | s_modified.str.endswith(pat)
                if self.na is not None:
                    return result.fillna(self.na)
                return result
        elif not self.case:
            # Use str.lower() workaround for case-insensitive matching
            s_modified = s.str.lower()
            pat_modified = self.pat.lower()
            result = s_modified.str.endswith(pat_modified)
        else:
            result = s.str.endswith(self.pat)

        # Handle na parameter for non-tuple cases
        if is_cudf:
            # cuDF doesn't support na parameter, use fillna
            if self.na is not None:
                return result.fillna(self.na)
            else:
                return result
        else:
            # pandas supports na parameter for case-sensitive str patterns
            if not self.case:
                if self.na is not None:
                    return result.fillna(self.na)
                else:
                    return result
            else:
                return s.str.endswith(self.pat, self.na)

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


def endswith(
    pat: Union[str, tuple],
    case: bool = True,
    na: Optional[bool] = None
) -> Endswith:
    """
    Return whether a given pattern or tuple of patterns is at the
    end of a string

    Args:
        pat: Pattern (str) or tuple of patterns to match at end of
             string. When tuple, returns True if string ends with ANY
             pattern (OR logic)
        case: If True, case-sensitive matching (default: True)
        na: Fill value for missing values (default: None)

    Returns:
        Endswith predicate

    Examples:
        >>> # Single pattern, case-sensitive (default)
        >>> n({"email": endswith(".com")})
        >>> # Single pattern, case-insensitive
        >>> n({"email": endswith(".COM", case=False)})
        >>> # Multiple patterns (OR logic)
        >>> n({"filename": endswith((".txt", ".csv"))})
        >>> # Multiple patterns, case-insensitive
        >>> n({"filename": endswith((".TXT", ".CSV"), case=False)})
    """
    return Endswith(pat, case, na)


class Match(ASTPredicate):
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

    def __call__(self, s: SeriesT) -> SeriesT:
        # workaround cuDF not supporting 'case' and 'na' parameters
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/
        # cudf.core.accessors.string.stringmethods.match/
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        if is_cudf:
            if not self.case:
                s_modified = s.str.lower()
                pat_modified = (
                    self.pat.lower()
                    if isinstance(self.pat, str)
                    else self.pat
                )
                result = s_modified.str.match(pat_modified, flags=self.flags)
            else:
                result = s.str.match(self.pat, flags=self.flags)

            if self.na is not None and isinstance(self.na, bool):
                result = result.fillna(self.na)

            return result
        else:
            return s.str.match(self.pat, self.case, self.flags, self.na)

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


class Fullmatch(ASTPredicate):
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

    def __call__(self, s: SeriesT) -> SeriesT:
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        if is_cudf:
            # cuDF doesn't have fullmatch, use match() with anchors as
            # workaround. fullmatch('abc') is equivalent to match('^abc$')
            anchored_pat = f'^{self.pat}$'

            if not self.case:
                s_modified = s.str.lower()
                pat_modified = (
                    anchored_pat.lower()
                    if isinstance(anchored_pat, str)
                    else anchored_pat
                )
                result = s_modified.str.match(pat_modified, flags=self.flags)
            else:
                result = s.str.match(anchored_pat, flags=self.flags)

            if self.na is not None and isinstance(self.na, bool):
                result = result.fillna(self.na)

            return result
        else:
            # pandas has native fullmatch support
            return s.str.fullmatch(self.pat, self.case, self.flags, self.na)

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


class IsNumeric(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isnumeric()


def isnumeric() -> IsNumeric:
    """
    Return whether a given string is numeric
    """
    return IsNumeric()


class IsAlpha(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isalpha()


def isalpha() -> IsAlpha:
    """
    Return whether a given string is alphabetic
    """
    return IsAlpha()


class IsDigit(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isdigit()


def isdigit() -> IsDigit:
    """
    Return whether a given string is numeric
    """
    return IsDigit()


class IsLower(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.islower()


def islower() -> IsLower:
    """
    Return whether a given string is lowercase
    """
    return IsLower()


class IsUpper(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isupper()


def isupper() -> IsUpper:
    """
    Return whether a given string is uppercase
    """
    return IsUpper()


class IsSpace(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isspace()


def isspace() -> IsSpace:
    """
    Return whether a given string is whitespace
    """
    return IsSpace()


class IsAlnum(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isalnum()


def isalnum() -> IsAlnum:
    """
    Return whether a given string is alphanumeric
    """
    return IsAlnum()


class IsDecimal(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.isdecimal()


def isdecimal() -> IsDecimal:
    """
    Return whether a given string is decimal
    """
    return IsDecimal()


class IsTitle(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.istitle()


def istitle() -> IsTitle:
    """
    Return whether a given string is title case
    """
    return IsTitle()


class IsNull(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isnull()


def isnull() -> IsNull:
    """
    Return whether a given string is null
    """
    return IsNull()


class NotNull(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.notnull()


def notnull() -> NotNull:
    """
    Return whether a given string is not null
    """
    return NotNull()
