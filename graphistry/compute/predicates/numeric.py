import operator
from typing import Any, ClassVar, Union, cast
import pandas as pd

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT


class NumericASTPredicate(ASTPredicate):
    op: ClassVar[Any]

    def __init__(self, val: Union[int, float]) -> None:
        self.val = val

    def __call__(self, s: SeriesT) -> SeriesT:
        return cast(SeriesT, type(self).op(s, self.val))

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.val, (int, float)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "val must be numeric (int or float)",
                field="val",
                value=type(self.val).__name__
            )

###

class GT(NumericASTPredicate):
    op = staticmethod(operator.gt)

def gt(val: float) -> GT:
    """
    Return whether a given value is greater than a threshold
    """
    return GT(val)

class LT(NumericASTPredicate):
    op = staticmethod(operator.lt)

def lt(val: float) -> LT:
    """
    Return whether a given value is less than a threshold
    """
    return LT(val)

class GE(NumericASTPredicate):
    op = staticmethod(operator.ge)

def ge(val: float) -> GE:
    """
    Return whether a given value is greater than or equal to a threshold
    """
    return GE(val)

class LE(NumericASTPredicate):
    op = staticmethod(operator.le)

def le(val: float) -> LE:
    """
    Return whether a given value is less than or equal to a threshold
    """
    return LE(val)

class EQ(NumericASTPredicate):
    op = staticmethod(operator.eq)

def eq(val: float) -> EQ:
    """
    Return whether a given value is equal to a threshold
    """
    return EQ(val)

class NE(NumericASTPredicate):
    op = staticmethod(operator.ne)

    def __call__(self, s: SeriesT) -> SeriesT:
        # openCypher/SQL three-valued logic: `null <> x` is NULL, so a NULL/NaN cell is NOT a match
        # (an unknown value cannot be proven unequal to x). pandas `NaN != x` returns True, which
        # would wrongly KEEP the null; AND with notna() so it is excluded — matching cuDF + the
        # polars engine. eq/gt/lt/ge/le already drop nulls (their operator returns False on a null).
        return cast(SeriesT, super().__call__(s) & s.notna())

def ne(val: float) -> NE:
    """
    Return whether a given value is not equal to a threshold.

    Follows openCypher/SQL three-valued logic: a NULL/NA cell is NOT a match (``null <> x`` is
    unknown), so null rows are excluded — the same as ``eq`` and the other comparisons.
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
        
    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.lower, (int, float)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "lower must be numeric (int or float)",
                field="lower",
                value=type(self.lower).__name__
            )
        
        if not isinstance(self.upper, (int, float)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "upper must be numeric (int or float)",
                field="upper",
                value=type(self.upper).__name__
            )
        
        if not isinstance(self.inclusive, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "inclusive must be boolean",
                field="inclusive",
                value=type(self.inclusive).__name__
            )

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
