from typing import Union
import pandas as pd

from .ASTPredicate import ASTPredicate


class NumericASTPredicate(ASTPredicate):
    def __init__(self, val: Union[int, float]) -> None:
        self.val = val

    def validate(self) -> None:
        assert isinstance(self.val, (int, float))

###

class GT(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s > self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'GT', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'GT':
        assert 'val' in d
        out = GT(val=d['val'])
        out.validate()
        return out

def gt(val: float) -> GT:
    """
    Return whether a given value is greater than a threshold
    """
    return GT(val)

class LT(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s < self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'LT', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'LT':
        assert 'val' in d
        out = LT(val=d['val'])
        out.validate()
        return out

def lt(val: float) -> LT:
    """
    Return whether a given value is less than a threshold
    """
    return LT(val)

class GE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s >= self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'GE', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'GE':
        assert 'val' in d
        out = GE(val=d['val'])
        out.validate()
        return out

def ge(val: float) -> GE:
    """
    Return whether a given value is greater than or equal to a threshold
    """
    return GE(val)

class LE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s <= self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'LE', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'LE':
        assert 'val' in d
        out = LE(val=d['val'])
        out.validate()
        return out

def le(val: float) -> LE:
    """
    Return whether a given value is less than or equal to a threshold
    """
    return LE(val)

class EQ(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s == self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'EQ', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'EQ':
        assert 'val' in d
        out = EQ(val=d['val'])
        out.validate()
        return out

def eq(val: float) -> EQ:
    """
    Return whether a given value is equal to a threshold
    """
    return EQ(val)

class NE(NumericASTPredicate):
    def __init__(self, val: float) -> None:
        self.val = val

    def __call__(self, s: pd.Series) -> pd.Series:
        return s != self.val

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'NE', 'val': self.val}
    
    @classmethod
    def from_json(cls, d: dict) -> 'NE':
        assert 'val' in d
        out = NE(val=d['val'])
        out.validate()
        return out

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

    def __call__(self, s: pd.Series) -> pd.Series:
        if self.inclusive:
            return (s >= self.lower) & (s <= self.upper)
        else:
            return (s > self.lower) & (s < self.upper)
        
    def validate(self) -> None:
        assert isinstance(self.lower, (int, float))
        assert isinstance(self.upper, (int, float))
        assert isinstance(self.inclusive, bool)

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'Between', 'lower': self.lower, 'upper': self.upper, 'inclusive': self.inclusive}
    
    @classmethod
    def from_json(cls, d: dict) -> 'Between':
        assert 'lower' in d
        assert 'upper' in d
        assert 'inclusive' in d
        out = Between(lower=d['lower'], upper=d['upper'], inclusive=d['inclusive'])
        out.validate()
        return out

def between(lower: float, upper: float, inclusive: bool = True) -> Between:
    """
    Return whether a given value is between a lower and upper threshold
    """
    return Between(lower, upper, inclusive)

class IsNA(ASTPredicate):
    def __call__(self, s: pd.Series) -> pd.Series:
        return s.isna()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'IsNA'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsNA':
        return IsNA()

def isna() -> IsNA:
    """
    Return whether a given value is NA
    """
    return IsNA()


class NotNA(ASTPredicate):
    def __call__(self, s: pd.Series) -> pd.Series:
        return s.notna()
    
    def to_json(self, validate=True) -> dict:
        return {'type': 'NotNA'}
    
    @classmethod
    def from_json(cls, d: dict) -> 'NotNA':
        return NotNA()

def notna() -> NotNA:
    """
    Return whether a given value is not NA
    """
    return NotNA()
