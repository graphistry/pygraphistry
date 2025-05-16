from typing import Any
from typing_extensions import Literal
import pandas as pd

from .ASTPredicate import ASTPredicate
from graphistry.compute.typing import SeriesT

class Duplicated(ASTPredicate):
    def __init__(self, keep: Literal['first', 'last', False] = 'first') -> None:
        self.keep = keep

    def __call__(self, s: SeriesT) -> SeriesT:
        return s.duplicated(keep=self.keep)

    def validate(self) -> None:
        assert self.keep in ['first', 'last', False]

def duplicated(keep: Literal['first', 'last', False] = 'first') -> Duplicated:
    """
    Return whether a given value is duplicated
    """
    return Duplicated(keep=keep)
