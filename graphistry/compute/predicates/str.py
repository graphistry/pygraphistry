from typing import Optional
import pandas as pd

from .ASTPredicate import ASTPredicate


class Contains(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na
        self.regex = regex

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.contains(self.pat, self.case, self.flags, self.na, self.regex)
    
    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.case, bool)
        assert isinstance(self.flags, int)
        assert isinstance(self.na, (bool, type(None)))
        assert isinstance(self.regex, bool)
    
    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Contains',
            'pat': self.pat,
            'case': self.case,
            'flags': self.flags,
            **({'na': self.na} if self.na is not None else {}),
            'regex': self.regex
        }
    
    @classmethod
    def from_json(cls, d: dict) -> 'Contains':
        assert 'pat' in d
        assert 'case' in d
        assert 'flags' in d
        assert 'regex' in d
        out = Contains(
            pat=d['pat'],
            case=d['case'],
            flags=d['flags'],
            na=d['na'] if 'na' in d else None,
            regex=d['regex']
        )
        out.validate()
        return out

def contains(pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None, regex: bool = True) -> Contains:
    """
    Return whether a given pattern or regex is contained within a string
    """
    return Contains(pat, case, flags, na, regex)


class Startswith(ASTPredicate):
    def __init__(self, pat: str, na: Optional[str] = None) -> None:
        self.pat = pat
        self.na = na

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.startswith(self.pat, self.na)

    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.na, (str, type(None)))
    
    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Startswith',
            'pat': self.pat,
            **({'na': self.na} if self.na is not None else {})
        }
    
    @classmethod
    def from_json(cls, d: dict) -> 'Startswith':
        assert 'pat' in d
        out = Startswith(
            pat=d['pat'],
            na=d['na'] if 'na' in d else None
        )
        out.validate()
        return out

def startswith(pat: str, na: Optional[str] = None) -> Startswith:
    """
    Return whether a given pattern is at the start of a string
    """
    return Startswith(pat, na)

class Endswith(ASTPredicate):
    def __init__(self, pat: str, na: Optional[str] = None) -> None:
        self.pat = pat
        self.na = na

    def __call__(self, s: pd.Series) -> pd.Series:
        """
        Return whether a given pattern is at the end of a string
        """
        return s.str.endswith(self.pat, self.na)

    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.na, (str, type(None)))
    
    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Endswith',
            'pat': self.pat,
            **({'na': self.na} if self.na is not None else {})
        }
    
    @classmethod
    def from_json(cls, d: dict) -> 'Endswith':
        assert 'pat' in d
        out = Endswith(
            pat=d['pat'],
            na=d['na'] if 'na' in d else None
        )
        out.validate()
        return out

def endswith(pat: str, na: Optional[str] = None) -> Endswith:
    return Endswith(pat, na)

class Match(ASTPredicate):
    def __init__(self, pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> None:
        self.pat = pat
        self.case = case
        self.flags = flags
        self.na = na

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.match(self.pat, self.case, self.flags, self.na)
    
    def validate(self) -> None:
        assert isinstance(self.pat, str)
        assert isinstance(self.case, bool)
        assert isinstance(self.flags, int)
        assert isinstance(self.na, (bool, type(None)))
    
    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'Match',
            'pat': self.pat,
            'case': self.case,
            'flags': self.flags,
            **({'na': self.na} if self.na is not None else {})
        }
    
    @classmethod
    def from_json(cls, d: dict) -> 'Match':
        assert 'pat' in d
        assert 'case' in d
        assert 'flags' in d
        out = Match(
            pat=d['pat'],
            case=d['case'],
            flags=d['flags'],
            na=d['na'] if 'na' in d else None
        )
        out.validate()
        return out

def match(pat: str, case: bool = True, flags: int = 0, na: Optional[bool] = None) -> Match:
    """
    Return whether a given pattern is at the start of a string
    """
    return Match(pat, case, flags, na)

class IsNumeric(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isnumeric()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsNumeric'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsNumeric':
        return IsNumeric()
    
def isnumeric() -> IsNumeric:
    """
    Return whether a given string is numeric
    """
    return IsNumeric()

class IsAlpha(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isalpha()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsAlpha'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsAlpha':
        return IsAlpha()

def isalpha() -> IsAlpha:
    """
    Return whether a given string is alphabetic
    """
    return IsAlpha()

class IsDigit(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isdigit()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsDigit'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsDigit':
        return IsDigit()

def isdigit() -> IsDigit:
    """
    Return whether a given string is numeric
    """
    return IsDigit()

class IsLower(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.islower()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsLower'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsLower':
        return IsLower()

def islower() -> IsLower:
    """
    Return whether a given string is lowercase
    """
    return IsLower()

class IsUpper(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isupper()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsUpper'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsUpper':
        return IsUpper()

def isupper() -> IsUpper:
    """
    Return whether a given string is uppercase
    """
    return IsUpper()

class IsSpace(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isspace()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsSpace'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsSpace':
        return IsSpace()
    
def isspace() -> IsSpace:
    """
    Return whether a given string is whitespace
    """
    return IsSpace()

class IsAlnum(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isalnum()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsAlnum'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsAlnum':
        return IsAlnum()
    
def isalnum() -> IsAlnum:
    """
    Return whether a given string is alphanumeric
    """
    return IsAlnum()

class IsDecimal(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.isdecimal()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsDecimal'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsDecimal':
        return IsDecimal()

def isdecimal() -> IsDecimal:
    """
    Return whether a given string is decimal
    """
    return IsDecimal()

class IsTitle(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.str.istitle()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsTitle'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsTitle':
        return IsTitle()

def istitle() -> IsTitle:
    """
    Return whether a given string is title case
    """
    return IsTitle()

class IsNull(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series: 
        return s.isnull()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsNull'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsNull':
        return IsNull()

def isnull() -> IsNull:
    """
    Return whether a given string is null
    """
    return IsNull()

class NotNull(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series: 
        return s.notnull()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'NotNull'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'NotNull':
        return NotNull()

def notnull() -> NotNull:
    """
    Return whether a given string is not null
    """
    return NotNull()
