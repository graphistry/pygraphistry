from typing_extensions import Literal
import pandas as pd

from .ASTPredicate import ASTPredicate

class Duplicated(ASTPredicate):
    def __init__(self, keep: Literal['first', 'last', False] = 'first') -> None:
        self.keep = keep

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.duplicated(keep=self.keep)

    def validate(self) -> None:
        assert self.keep in ['first', 'last', False]

    def to_json(self, validate=True) -> dict:
        if validate:
            self.validate()
        return {'type': 'Duplicated', 'keep': self.keep}
    
    @classmethod
    def from_json(cls, d: dict) -> 'Duplicated':
        assert 'keep' in d
        out = Duplicated(keep=d['keep'])
        out.validate()
        return out

def duplicated(keep: Literal['first', 'last', False] = 'first') -> Duplicated:
    """
    Return whether a given value is duplicated
    """
    return Duplicated(keep=keep)
