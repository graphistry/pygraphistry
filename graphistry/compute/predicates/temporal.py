from typing import Any, cast

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


class _DatetimePropertyPredicate(ASTPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        raise NotImplementedError()

    def __call__(self, s: SeriesT) -> SeriesT:
        return cast(SeriesT, type(self).predicate(s))


class IsMonthStart(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_month_start

def is_month_start() -> IsMonthStart:
    """
    Return whether a given value is a month start
    """
    return IsMonthStart()

class IsMonthEnd(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_month_end

def is_month_end() -> IsMonthEnd:
    """
    Return whether a given value is a month end
    """
    return IsMonthEnd()

class IsQuarterStart(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_quarter_start

def is_quarter_start() -> IsQuarterStart:
    """
    Return whether a given value is a quarter start
    """
    return IsQuarterStart()

class IsQuarterEnd(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_quarter_end

def is_quarter_end() -> IsQuarterEnd:
    """
    Return whether a given value is a quarter end
    """
    return IsQuarterEnd()

class IsYearStart(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_year_start

def is_year_start() -> IsYearStart:
    """
    Return whether a given value is a year start
    """
    return IsYearStart()

class IsYearEnd(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_year_end

def is_year_end() -> IsYearEnd:
    """
    Return whether a given value is a year end
    """
    return IsYearEnd()

class IsLeapYear(_DatetimePropertyPredicate):
    @staticmethod
    def predicate(s: Any) -> Any:
        return s.dt.is_leap_year

def is_leap_year() -> IsLeapYear:
    """
    Return whether a given value is a leap year
    """
    return IsLeapYear()
