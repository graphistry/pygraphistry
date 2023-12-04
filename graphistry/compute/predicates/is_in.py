from typing import Any, List
import pandas as pd

from .ASTPredicate import ASTPredicate


class IsIn(ASTPredicate):
    def __init__(self, options: List[Any]) -> None:
        self.options = options
    
    def __call__(self, s: pd.Series) -> pd.Series:
        return s.isin(self.options)

def is_in(options: List[Any]) -> IsIn:
    return IsIn(options)
