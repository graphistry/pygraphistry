"""Internal type aliases for predicates - NOT PUBLIC API"""

from typing import Union, TypedDict, Literal
from datetime import datetime, date, time
import pandas as pd
import numpy as np

# Type aliases to reduce repetition in function signatures
NumericValueType = Union[int, float, np.number]
TemporalInputType = Union[pd.Timestamp, datetime, date, time]


# Tagged dict types for wire protocol
class DateTimeDict(TypedDict, total=False):
    type: Literal["datetime"]
    value: str
    timezone: str  # Optional, defaults to UTC


class DateDict(TypedDict):
    type: Literal["date"]
    value: str


class TimeDict(TypedDict):
    type: Literal["time"] 
    value: str


# Union of all tagged temporal dicts
TaggedDict = Union[DateTimeDict, DateDict, TimeDict]

# Full input type for predicates
PredicateInputType = Union[NumericValueType, TemporalInputType, TaggedDict]
