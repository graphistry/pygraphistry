"""Internal type aliases for predicates - NOT PUBLIC API"""

from typing import Union
from datetime import datetime, date, time
import pandas as pd
import numpy as np

# Type aliases to reduce repetition in function signatures
NumericValueType = Union[int, float, np.number]
TemporalInputType = Union[pd.Timestamp, datetime, date, time]
TaggedDict = dict  # {"type": str, "value": ..., "timezone"?: str}
PredicateInputType = Union[NumericValueType, TemporalInputType, TaggedDict]
