from typing import Dict, List, Type

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.categorical import Duplicated
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.predicates.logical import AllOf
from graphistry.compute.predicates.numeric import GT, LT, GE, LE, EQ, NE, Between, IsNA, NotNA
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch, IsNumeric, IsAlpha, IsDecimal, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsTitle, IsNull, NotNull
)
from graphistry.compute.predicates.temporal import (
    IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
    IsYearStart, IsYearEnd, IsLeapYear
)
from graphistry.utils.json import JSONVal


predicates : List[Type[ASTPredicate]] = [
    Duplicated,
    IsIn,
    AllOf,
    GT, LT, GE, LE, EQ, NE, Between, IsNA, NotNA,
    Contains, Startswith, Endswith, Match, Fullmatch, IsNumeric, IsAlpha, IsDecimal, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsDecimal, IsTitle, IsNull, NotNull,
    IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
    IsYearStart, IsYearEnd, IsLeapYear
]

type_to_predicate: Dict[str, Type[ASTPredicate]] = {
    cls.__name__: cls
    for cls in predicates
}

def from_json(d: Dict[str, JSONVal]) -> ASTPredicate:
    if not isinstance(d, dict):
        raise ValueError('d must be a dict')
    if 'type' not in d:
        raise ValueError("d must have a 'type' key")
    if not isinstance(d['type'], str):
        raise ValueError("d['type'] must be a string")
    if d['type'] not in type_to_predicate:
        raise ValueError(f"Unknown predicate type: {d['type']}")
    pred = type_to_predicate[d['type']]
    out = pred.from_json(d)
    if not isinstance(out, ASTPredicate):
        raise ValueError(f'pred.from_json must return an ASTPredicate, got {type(out)}')
    out.validate()
    return out
