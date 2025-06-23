"""Type definitions for predicates"""
from typing import Union, Literal
import pandas as pd
import numpy as np
from datetime import datetime, date, time

# Import temporal wire types
from ...models.gfql.types.temporal import DateTimeWire, DateWire, TimeWire

# Normalized types after processing inputs
NormalizedNumeric = Union[int, float, np.number]
NormalizedTemporal = Union[pd.Timestamp, datetime, date, time, DateTimeWire, DateWire, TimeWire]
NormalizedScalar = Union[str, bool, None]

# For is_in after normalization - includes all possible return types from _normalize_value
NormalizedIsInElement = Union[
    NormalizedScalar,
    NormalizedNumeric, 
    NormalizedTemporal
]

# Comparison operators
ComparisonOp = Literal["gt", "lt", "ge", "le", "eq", "ne"]
