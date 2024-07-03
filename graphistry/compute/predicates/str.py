from typing import Any, TYPE_CHECKING, Optional
import pandas as pd

from .ASTPredicate import ASTPredicate


if TYPE_CHECKING:
    SeriesT = pd.Series
else:
    SeriesT = Any


class Contains(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na
        self.regex = regex

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.contains(self.pat, self.case, self.flags, self.na, self.regex)
    
    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.case, bool)
        assert isinstance(self.flags, int)
        assert isinstance(self.na, (bool, type(None)))
        assert isinstance(self.regex, bool)

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
        return s.str.startswith(self.pat, self.na)

    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.na, (str, type(None)))

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
        """
        Return whether a given pattern is at the end of a string
        """
        return s.str.endswith(self.pat, self.na)

    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.na, (str, type(None)))

def endswith(pat: str, na: Optional[str] = None) -> Endswith:
    return Endswith(pat, na)

class Match(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.str.match(self.pat, self.case, self.flags, self.na)
    
    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.case, bool)
        assert isinstance(self.flags, int)
        assert isinstance(self.na, (bool, type(None)))

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
