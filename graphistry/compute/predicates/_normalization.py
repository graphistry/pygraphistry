"""Internal normalization utilities - NOT PUBLIC API"""

from typing import Union, TypeVar, overload, Dict, Any
from datetime import datetime, date, time
import pandas as pd
import numpy as np

from .temporal_values import TemporalValue, DateTimeValue, DateValue, TimeValue
from ._types import NumericValueType, TemporalInputType, TaggedDict

# Output types for comparison normalization
ComparisonValueType = Union[NumericValueType, TemporalValue]

# Input type for comparison values
ComparisonInputType = Union[NumericValueType, TemporalInputType, TaggedDict, TemporalValue]


@overload
def normalize_comparison_value(val: NumericValueType, class_name: str) -> NumericValueType:
    ...

@overload
def normalize_comparison_value(val: Union[pd.Timestamp, datetime], class_name: str) -> DateTimeValue:
    ...

@overload
def normalize_comparison_value(val: date, class_name: str) -> DateValue:
    ...

@overload  
def normalize_comparison_value(val: time, class_name: str) -> TimeValue:
    ...

@overload
def normalize_comparison_value(val: TaggedDict, class_name: str) -> TemporalValue:
    ...

@overload
def normalize_comparison_value(val: TemporalValue, class_name: str) -> TemporalValue:
    ...


def normalize_comparison_value(val: ComparisonInputType, class_name: str) -> ComparisonValueType:
    """
    Normalize values for comparison predicates (GT, LT, etc.)
    Internal use only - maintains exact behavior of original code
    """
    # Numeric types (existing behavior)
    if isinstance(val, (int, float, np.number)):
        return val
    
    # Native temporal types (new)
    elif isinstance(val, pd.Timestamp):
        return DateTimeValue.from_pandas_timestamp(val)
    elif isinstance(val, datetime):
        return DateTimeValue.from_datetime(val)
    elif isinstance(val, date):
        return DateValue.from_date(val)
    elif isinstance(val, time):
        return TimeValue.from_time(val)
    
    # Tagged dict (for JSON deserialization)
    elif isinstance(val, dict) and "type" in val:
        if val["type"] == "datetime":
            return DateTimeValue(val["value"], val.get("timezone", "UTC"))
        elif val["type"] == "date":
            return DateValue(val["value"])
        elif val["type"] == "time":
            return TimeValue(val["value"])
        else:
            raise ValueError(f"Unknown temporal type: {val['type']}")
    
    # Already a temporal value
    elif isinstance(val, TemporalValue):
        return val
    
    # Reject raw strings as ambiguous
    elif isinstance(val, str):
        raise ValueError(
            f"Raw string '{val}' is ambiguous. Use:\n"
            f"  - {class_name.lower()}(pd.Timestamp('{val}')) for datetime\n"
            f"  - {class_name.lower()}({{'type': 'datetime', 'value': '{val}'}})) for explicit type"
        )
    
    else:
        raise TypeError(f"Unsupported type for {class_name}: {type(val)}")


# Types for IsIn normalization
BasicType = Union[int, float, str, np.number, None]
IsInOutputType = Union[BasicType, pd.Timestamp, date, time, Dict[str, Any]]
IsInInputType = Union[BasicType, TemporalInputType, TaggedDict, TemporalValue, Dict[str, Any]]


@overload
def normalize_isin_value(val: BasicType) -> BasicType:
    ...

@overload
def normalize_isin_value(val: pd.Timestamp) -> pd.Timestamp:
    ...

@overload
def normalize_isin_value(val: Union[datetime, date]) -> pd.Timestamp:
    ...

@overload
def normalize_isin_value(val: time) -> time:
    ...

@overload
def normalize_isin_value(val: TemporalValue) -> Union[pd.Timestamp, date, time]:
    ...

@overload
def normalize_isin_value(val: Dict[str, Any]) -> Union[pd.Timestamp, date, time, Dict[str, Any]]:
    ...


def normalize_isin_value(val: IsInInputType) -> IsInOutputType:
    """
    Normalize values for IsIn predicate
    Internal use only - maintains exact behavior of original code
    """
    # Pass through basic types unchanged
    if isinstance(val, (int, float, str, np.number, type(None))):
        return val
    
    # Native temporal types
    elif isinstance(val, pd.Timestamp):
        return val
    elif isinstance(val, datetime):
        return pd.Timestamp(val)
    elif isinstance(val, date):
        return pd.Timestamp(val)
    elif isinstance(val, time):
        # For time-only values, keep as time object
        return val
    
    # Tagged dict (for JSON deserialization)
    elif isinstance(val, dict) and "type" in val:
        if val["type"] == "datetime":
            dt_val = DateTimeValue(val["value"], val.get("timezone", "UTC"))
            return dt_val.as_pandas_value()
        elif val["type"] == "date":
            date_val = DateValue(val["value"])
            return date_val.as_pandas_value()
        elif val["type"] == "time":
            time_val = TimeValue(val["value"])
            return time_val.as_pandas_value()
        else:
            # Non-temporal dict, pass through
            return val
    
    # Already a temporal value
    elif isinstance(val, TemporalValue):
        return val.as_pandas_value()
    
    # Everything else passes through
    else:
        return val
