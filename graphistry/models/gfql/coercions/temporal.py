"""
Pure temporal type transformations

Functions to convert between Native, Wire, and AST representations.
"""

from typing import Union
from datetime import datetime, date, time
import pandas as pd

from graphistry.compute.ast_temporal import (
    TemporalValue, DateTimeValue, DateValue, TimeValue
)
from graphistry.models.gfql.types.temporal import NativeTemporal, TemporalWire


# ============= To AST Transforms =============


def to_ast(
    val: Union[NativeTemporal, TemporalWire, TemporalValue]
) -> TemporalValue:
    """Convert any temporal representation to AST (TemporalValue)"""
    # Already AST
    if isinstance(val, TemporalValue):
        return val

    # From native
    elif isinstance(val, pd.Timestamp):
        return DateTimeValue.from_pandas_timestamp(val)
    elif isinstance(val, datetime):
        return DateTimeValue.from_datetime(val)
    elif isinstance(val, date):
        return DateValue.from_date(val)
    elif isinstance(val, time):
        return TimeValue.from_time(val)

    # From wire
    elif isinstance(val, dict) and "type" in val:
        if val["type"] == "datetime":
            timezone = val.get("timezone", "UTC")
            assert isinstance(timezone, str)
            return DateTimeValue(val["value"], timezone)
        elif val["type"] == "date":
            return DateValue(val["value"])
        elif val["type"] == "time":
            return TimeValue(val["value"])
        else:
            raise ValueError(f"Unknown temporal wire type: {val['type']}")

    else:
        raise TypeError(f"Cannot convert {type(val)} to AST temporal")


# ============= To Native Transforms =============


def to_native(
    val: Union[NativeTemporal, TemporalWire, TemporalValue]
) -> Union[pd.Timestamp, time]:
    """Convert any temporal representation to native Python/Pandas type"""
    # Already native pandas
    if isinstance(val, pd.Timestamp):
        return val

    # Native Python to pandas
    elif isinstance(val, (datetime, date)):
        return pd.Timestamp(val)
    elif isinstance(val, time):
        return val  # time stays as time

    # From AST
    elif isinstance(val, TemporalValue):
        return val.as_pandas_value()

    # From wire (via AST)
    elif isinstance(val, dict) and "type" in val:
        ast_val = to_ast(val)
        return ast_val.as_pandas_value()

    else:
        raise TypeError(f"Cannot convert {type(val)} to native temporal")


# ============= To Wire Transforms =============


def to_wire(val: Union[NativeTemporal, TemporalValue]) -> TemporalWire:
    """Convert temporal to wire format (for JSON serialization)"""
    # From AST
    if isinstance(val, TemporalValue):
        return val.to_json()

    # From native (via AST)
    else:
        ast_val = to_ast(val)
        return ast_val.to_json()
