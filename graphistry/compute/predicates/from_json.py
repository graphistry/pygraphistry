from typing import Dict, List, Type

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.categorical import Duplicated
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.predicates.logical import AllOf
# Register the comparison.py versions (numeric + temporal + string), NOT numeric.py's numeric-only
# classes: they share __name__ ("EQ", "GT", ...) so the from_json type registry would otherwise
# downgrade a live comparison predicate to numeric-only on a JSON round-trip — raising "val must be
# numeric" for a string/temporal equality or temporal comparison built by Cypher lowering / the
# public predicate API (both of which use comparison.py). See from_json round-trip regression tests.
from graphistry.compute.predicates.comparison import GT, LT, GE, LE, EQ, NE, Between, IsNA, NotNA
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
    IsSpace, IsAlnum, IsTitle, IsNull, NotNull,
    IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
    IsYearStart, IsYearEnd, IsLeapYear
]

type_to_predicate: Dict[str, Type[ASTPredicate]] = {
    cls.__name__: cls
    for cls in predicates
}

def from_json(d: Dict[str, JSONVal]) -> ASTPredicate:
    assert isinstance(d, dict)
    assert 'type' in d
    assert d['type'] in type_to_predicate
    assert isinstance(d['type'], str)
    pred = type_to_predicate[d['type']]
    out = pred.from_json(d)
    assert isinstance(out, ASTPredicate)
    out.validate()
    return out
