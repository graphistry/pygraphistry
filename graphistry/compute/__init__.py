from .ComputeMixin import ComputeMixin
from .ast import (
    n, e, e_forward, e_reverse, e_undirected,
    let, remote, ref, call
)
from .chain import Chain
from .calls import hypergraph
from graphistry.models.gfql.types.call import CallMethodName
from .predicates.is_in import (
    is_in, IsIn
)
from .predicates.categorical import (
    duplicated, Duplicated,
)
from .predicates.temporal import (
    is_month_start, IsMonthStart,
    is_month_end, IsMonthEnd,
    is_quarter_start, IsQuarterStart,
    is_quarter_end, IsQuarterEnd,
    is_year_start, IsYearStart,
    is_year_end, IsYearEnd,
    is_leap_year, IsLeapYear
)
from .ast_temporal import (
    TemporalValue,
    DateTimeValue,
    DateValue,
    TimeValue,
    temporal_value_from_json
)
from .predicates.comparison import (
    gt, GT,
    lt, LT,
    ge, GE,
    le, LE,
    eq, EQ,
    ne, NE,
    between, Between,
    isna, IsNA,
    notna, NotNA
)
from .predicates.str import (
    contains, Contains,
    startswith, Startswith,
    endswith, Endswith,
    match, Match,
    fullmatch, Fullmatch,
    isnumeric, IsNumeric,
    isalpha, IsAlpha,
    isdigit, IsDigit,
    islower, IsLower,
    isupper, IsUpper,
    isspace, IsSpace,
    isalnum, IsAlnum,
    isdecimal, IsDecimal,
    istitle, IsTitle,
    isnull, IsNull,
    notnull, NotNull,
)
from .typing import DataFrameT

__all__ = [
    # Core classes
    'ComputeMixin', 'Chain',
    # AST nodes
    'n', 'e', 'e_forward', 'e_reverse', 'e_undirected',
    'let', 'remote', 'ref', 'call',
    # Call types
    'CallMethodName',
    # Predicates
    'is_in', 'IsIn',
    'duplicated', 'Duplicated',
    'is_month_start', 'IsMonthStart',
    'is_month_end', 'IsMonthEnd',
    'is_quarter_start', 'IsQuarterStart',
    'is_quarter_end', 'IsQuarterEnd',
    'is_year_start', 'IsYearStart',
    'is_year_end', 'IsYearEnd',
    'is_leap_year', 'IsLeapYear',
    # Temporal
    'TemporalValue', 'DateTimeValue', 'DateValue', 'TimeValue',
    'temporal_value_from_json',
    # Comparison predicates
    'gt', 'GT', 'lt', 'LT', 'ge', 'GE', 'le', 'LE',
    'eq', 'EQ', 'ne', 'NE', 'between', 'Between',
    'isna', 'IsNA', 'notna', 'NotNA',
    # String predicates
    'contains', 'Contains', 'startswith', 'Startswith',
    'endswith', 'Endswith', 'match', 'Match',
    'fullmatch', 'Fullmatch',
    'isnumeric', 'IsNumeric', 'isalpha', 'IsAlpha',
    'isdigit', 'IsDigit', 'islower', 'IsLower',
    'isupper', 'IsUpper', 'isspace', 'IsSpace',
    'isalnum', 'IsAlnum', 'isdecimal', 'IsDecimal',
    'istitle', 'IsTitle', 'isnull', 'IsNull',
    'notnull', 'NotNull',
    # Types
    'DataFrameT'
]
