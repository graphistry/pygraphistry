"""Type definitions for predicates"""
from typing import Literal, Union
from datetime import date, datetime, time
import numpy as np
import pandas as pd

from graphistry.models.gfql.types.temporal import DateTimeWire, DateWire, TimeWire

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
