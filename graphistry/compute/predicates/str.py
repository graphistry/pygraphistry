from typing import Any, Optional
import pandas as pd

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


class Contains(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na
        self.regex = regex

    def __call__(self, s: SeriesT) -> SeriesT:
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        # workaround cuDF not supporting 'case' and 'na' parameters https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.core.accessors.string.stringmethods.contains/
        if is_cudf:
            if not self.case:
                s_modified = s.str.lower()
                pat_modified = self.pat.lower() if isinstance(self.pat, str) else self.pat
                result = s_modified.str.contains(pat_modified, regex=self.regex, flags=self.flags)
            else:
                result = s.str.contains(self.pat, regex=self.regex, flags=self.flags)

            if self.na is not None and isinstance(self.na, bool):
                result = result.fillna(self.na)

            return result
        else:
            return s.str.contains(self.pat, self.case, self.flags, self.na, self.regex)

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

def contains(pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> Contains:
    """
    Return whether a given pattern or regex is contained within a string
    """
    return Contains(pat, case, flags, na, regex)


class Startswith(ASTPredicate):
    def __init__(self, pat: str, na: Optional[str] = None) -> None:
        self.pat = pat
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        # workaround cuDF not supporting 'na' parameter
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.core.accessors.string.stringmethods.startswith/
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.startswith
        if is_cudf:
            result = s.str.startswith(self.pat)
            return result.fillna(self.na) if self.na is not None else result
        else:
            # pandas supports the na parameter directly
            return s.str.startswith(self.pat, self.na)

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
        
        if not isinstance(self.na, (str, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be string or None",
                field="na",
                value=type(self.na).__name__
            )

def startswith(pat: str, na: Optional[str] = None) -> Startswith:
    """
    Return whether a given pattern is at the start of a string
    """
    return Startswith(pat, na)

class Endswith(ASTPredicate):
    def __init__(self, pat: str, na: Optional[str] = None) -> None:
        self.pat = pat
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        # workaround cuDF not supporting 'na' parameter
        # https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.core.accessors.string.stringmethods.endswith/
        # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html#pandas.Series.str.endswith
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        if is_cudf:
            result = s.str.endswith(self.pat)
            return result.fillna(self.na) if self.na is not None else result
        else:
            return s.str.endswith(self.pat, self.na)

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
        
        if not isinstance(self.na, (str, type(None))):
            raise GFQLTypeError(
                ErrorCode.E201,
                "na must be string or None",
                field="na",
                value=type(self.na).__name__
            )

def endswith(pat: str, na: Optional[str] = None) -> Endswith:
    return Endswith(pat, na)

class Match(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        # workaround cuDF not supporting 'case' and 'na' parameters https://docs.rapids.ai/api/cudf/stable/user_guide/api_docs/api/cudf.core.accessors.string.stringmethods.match/
        is_cudf = hasattr(s, '__module__') and 'cudf' in s.__module__

        if is_cudf:
            if not self.case:
                s_modified = s.str.lower()
                pat_modified = self.pat.lower() if isinstance(self.pat, str) else self.pat
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

def match(pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> Match:
    """
    Return whether a given pattern is at the start of a string
    """
    return Match(pat, case, flags, na)

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
