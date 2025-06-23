"""Type definitions for predicates"""
from typing import Union, Literal
import pandas as pd
import numpy as np

# Normalized types after processing inputs
NormalizedNumeric = Union[int, float, np.number]
NormalizedTemporal = Union[pd.Timestamp]  # date/datetime -> pd.Timestamp
NormalizedScalar = Union[str, bool, None]

# For is_in after normalization
NormalizedIsInElement = Union[
    NormalizedScalar,
    NormalizedNumeric,
    NormalizedTemporal
]

# Comparison operators
ComparisonOp = Literal["gt", "lt", "ge", "le", "eq", "ne"]
