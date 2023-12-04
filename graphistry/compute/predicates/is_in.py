from typing import Any, List
from .ASTPredicate import ASTPredicate

class IsIn(ASTPredicate):
    def __init__(self, options: List[Any]) -> None:
        self.options = options

def is_in(options: List[Any]) -> IsIn:
    return IsIn(options)
