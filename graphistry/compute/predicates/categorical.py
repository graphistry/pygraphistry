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

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if self.keep not in ['first', 'last', False]:
            raise GFQLTypeError(
                ErrorCode.E201,
                "keep must be 'first', 'last', or False",
                field="keep",
                value=self.keep,
                suggestion="Use keep='first', keep='last', or keep=False"
            )

def duplicated(keep: Literal['first', 'last', False] = 'first') -> Duplicated:
    """
    Return whether a given value is duplicated
    """
    return Duplicated(keep=keep)
