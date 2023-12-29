from typing import Any, TYPE_CHECKING, Optional
import pandas as pd

from .ASTPredicate import ASTPredicate


if TYPE_CHECKING:
    SeriesT = pd.Series
else:
    SeriesT = Any


class IsMonthStart(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_month_start

def is_month_start() -> IsMonthStart:
    """
    Return whether a given value is a month start
    """
    return IsMonthStart()

class IsMonthEnd(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_month_end

def is_month_end() -> IsMonthEnd:
    """
    Return whether a given value is a month end
    """
    return IsMonthEnd()

class IsQuarterStart(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_quarter_start

def is_quarter_start() -> IsQuarterStart:
    """
    Return whether a given value is a quarter start
    """
    return IsQuarterStart()

class IsQuarterEnd(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_quarter_end

def is_quarter_end() -> IsQuarterEnd:
    """
    Return whether a given value is a quarter end
    """
    return IsQuarterEnd()

class IsYearStart(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_year_start

def is_year_start() -> IsYearStart:
    """
    Return whether a given value is a year start
    """
    return IsYearStart()

class IsYearEnd(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_year_end

def is_year_end() -> IsYearEnd:
    """
    Return whether a given value is a year end
    """
    return IsYearEnd()

class IsLeapYear(ASTPredicate):

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.dt.is_leap_year

def is_leap_year() -> IsLeapYear:
    """
    Return whether a given value is a leap year
    """
    return IsLeapYear()
