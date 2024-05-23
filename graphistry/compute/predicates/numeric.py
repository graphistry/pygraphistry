from typing import Any, TYPE_CHECKING, Union
import pandas as pd

from .ASTPredicate import ASTPredicate


if TYPE_CHECKING:
    SeriesT = pd.Series
else:
    SeriesT = Any


class NumericASTPredicate(ASTPredicate):
    def __init__(self, val: Union[int, float]) -> None:
        self.val = val

    def validate(self) -> None:
        assert isinstance(self.val, (int, float))

###

class GT(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s > self.val

def gt(val: float) -> GT:
    """
    Return whether a given value is greater than a threshold
    """
    return GT(val)

class LT(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s < self.val

def lt(val: float) -> LT:
    """
    Return whether a given value is less than a threshold
    """
    return LT(val)

class GE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s >= self.val

def ge(val: float) -> GE:
    """
    Return whether a given value is greater than or equal to a threshold
    """
    return GE(val)

class LE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s <= self.val

def le(val: float) -> LE:
    """
    Return whether a given value is less than or equal to a threshold
    """
    return LE(val)

class EQ(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s == self.val

def eq(val: float) -> EQ:
    """
    Return whether a given value is equal to a threshold
    """
    return EQ(val)

class NE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return s != self.val

def ne(val: float) -> NE:
    """
    Return whether a given value is not equal to a threshold
    """
    return NE(val)

class Between(ASTPredicate):
    def __init__(self, lower: float, upper: float, inclusive: bool = True) -> None:
        self.lower = lower
        self.upper = upper
        self.inclusive = inclusive

    def __call__(self, s: SeriesT) -> SeriesT:
        if self.inclusive:
            return (s >= self.lower) & (s <= self.upper)
        else:
            return (s > self.lower) & (s < self.upper)
        
    def validate(self) -> None:
        assert isinstance(self.lower, (int, float))
        assert isinstance(self.upper, (int, float))
        assert isinstance(self.inclusive, bool)

def between(lower: float, upper: float, inclusive: bool = True) -> Between:
    """
    Return whether a given value is between a lower and upper threshold
    """
    return Between(lower, upper, inclusive)

class IsNA(ASTPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isna()

def isna() -> IsNA:
    """
    Return whether a given value is NA
    """
    return IsNA()


class NotNA(ASTPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.notna()

def notna() -> NotNA:
    """
    Return whether a given value is not NA
    """
    return NotNA()
