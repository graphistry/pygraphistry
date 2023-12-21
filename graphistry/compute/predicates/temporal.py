from typing import Optional
import pandas as pd

from .ASTPredicate import ASTPredicate

class IsMonthStart(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_month_start
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsMonthStart'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsMonthStart':
        return IsMonthStart()

def is_month_start() -> IsMonthStart:
    """
    Return whether a given value is a month start
    """
    return IsMonthStart()

class IsMonthEnd(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_month_end
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsMonthEnd'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsMonthEnd':
        return IsMonthEnd()

def is_month_end() -> IsMonthEnd:
    """
    Return whether a given value is a month end
    """
    return IsMonthEnd()

class IsQuarterStart(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_quarter_start
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsQuarterStart'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsQuarterStart':
        return IsQuarterStart()

def is_quarter_start() -> IsQuarterStart:
    """
    Return whether a given value is a quarter start
    """
    return IsQuarterStart()

class IsQuarterEnd(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_quarter_end
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsQuarterEnd'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsQuarterEnd':
        return IsQuarterEnd()

def is_quarter_end() -> IsQuarterEnd:
    """
    Return whether a given value is a quarter end
    """
    return IsQuarterEnd()

class IsYearStart(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_year_start
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsYearStart'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsYearStart':
        return IsYearStart()

def is_year_start() -> IsYearStart:
    """
    Return whether a given value is a year start
    """
    return IsYearStart()

class IsYearEnd(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_year_end
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsYearEnd'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsYearEnd':
        return IsYearEnd()

def is_year_end() -> IsYearEnd:
    """
    Return whether a given value is a year end
    """
    return IsYearEnd()

class IsLeapYear(ASTPredicate):
    def __init__(self) -> None:
        pass

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.dt.is_leap_year
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsLeapYear'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsLeapYear':
        return IsLeapYear()

def is_leap_year() -> IsLeapYear:
    """
    Return whether a given value is a leap year
    """
    return IsLeapYear()
