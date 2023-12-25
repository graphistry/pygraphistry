from typing import TYPE_CHECKING, Any, List
import pandas as pd

from graphistry.utils.json import assert_json_serializable
from .ASTPredicate import ASTPredicate


if TYPE_CHECKING:
    SeriesT = pd.Series
else:
    SeriesT = Any


class IsIn(ASTPredicate):
    def __init__(self, options: List[Any]) -> None:
        self.options = options
    
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isin(self.options)
    
    def validate(self) -> None:
        assert isinstance(self.options, list)
        assert_json_serializable(self.options)

def is_in(options: List[Any]) -> IsIn:
    return IsIn(options)
