"""
Type definitions for GFQL predicate inputs

Defines what types users can pass to predicates.
Implementation details (like what gets stored internally)
belong in compute/predicates.
"""

from typing import Union, TYPE_CHECKING
import numpy as np

from .temporal import NativeTemporal, TemporalWire
from .numeric import NativeNumeric

if TYPE_CHECKING:
    from graphistry.compute.ast_temporal import TemporalValue


# ============= Basic Types =============
# Simple scalar types

BasicScalar = Union[int, float, str, np.number, None]


# ============= Predicate Input Types =============
# What users can provide to each predicate type

# Comparison predicates (GT, LT, GE, LE, EQ, NE) - strict, no strings
ComparisonInput = Union[
    NativeNumeric,      # Python int, float, np.number
    NativeTemporal,     # Python datetime, date, time, pd.Timestamp
    TemporalWire,       # Wire format: {"type": "datetime", ...}
    "TemporalValue",    # AST temporal values
]

# IsIn predicate - permissive, allows strings and arbitrary values
IsInElementInput = Union[
    BasicScalar,        # Includes strings and None
    NativeTemporal,     # Python datetime types
    TemporalWire,       # Wire format temporal
    "TemporalValue",    # AST temporal values
]

# Between predicate - each bound follows comparison rules
BetweenBoundInput = ComparisonInput
