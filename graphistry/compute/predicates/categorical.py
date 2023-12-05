from typing_extensions import Literal
import pandas as pd

from .ASTPredicate import ASTPredicate

class Duplicated(ASTPredicate):
    def __init__(self, keep: Literal['first', 'last', False] = 'first') -> None:
        self.keep = keep

    def __call__(self, s: pd.Series) -> pd.Series:
        return s.duplicated(keep=self.keep)

def duplicated(keep: Literal['first', 'last', False] = 'first') -> Duplicated:
    """
    Return whether a given value is duplicated
    """
    return Duplicated(keep=keep)
