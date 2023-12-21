from typing import Any, List
import pandas as pd

from graphistry.util import assert_json_serializable

from .ASTPredicate import ASTPredicate


class IsIn(ASTPredicate):
    def __init__(self, options: List[Any]) -> None:
        self.options = options
    
    def __call__(self, s: pd.Series) -> pd.Series:
        return s.isin(self.options)
    
    def validate(self) -> None:
        assert isinstance(self.options, list)
        assert_json_serializable(self.options)
    
    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {
            'type': 'IsIn',
            'options': self.options
        }
    
    @classmethod
    def from_json(cls, d: dict) -> 'IsIn':
        assert 'options' in d
        out = IsIn(options=d['options'])
        out.validate()
        return out

def is_in(options: List[Any]) -> IsIn:
    return IsIn(options)
