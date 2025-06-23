"""
Type definitions for temporal values in GFQL

Temporal data has three representations:
1. Native types - Host language types (Python datetime, pandas.Timestamp, etc.)
2. AST types - Domain model objects used in the query AST (TemporalValue classes)
3. Wire types - JSON serialization format for client/server communication

Conversion functions follow the pattern: to/from_native, to/from_wire, to/from_ast
"""

from typing import Union, TypedDict, Literal
from datetime import datetime, date, time
import pandas as pd


# ============= Native Temporal Types =============
# Host language types that users work with directly

NativeDateTime = Union[pd.Timestamp, datetime]
NativeDate = date  # Note: NOT a Union - important for overload ordering
NativeTime = time
NativeTemporal = Union[NativeDateTime, NativeDate, NativeTime]


# ============= Wire Types (JSON) =============
# Tagged dictionaries for JSON serialization/deserialization

class DateTimeWire(TypedDict, total=False):
    """Wire format for datetime with timezone"""
    type: Literal["datetime"]
    value: str  # ISO 8601 datetime string
    timezone: str  # Optional IANA timezone (default: UTC)


class DateWire(TypedDict):
    """Wire format for date only"""
    type: Literal["date"]
    value: str  # ISO 8601 date (YYYY-MM-DD)


class TimeWire(TypedDict):
    """Wire format for time only"""
    type: Literal["time"]
    value: str  # ISO 8601 time (HH:MM:SS[.ffffff])


# Union of all temporal wire types
TemporalWire = Union[DateTimeWire, DateWire, TimeWire]


# Note: AST types (TemporalValue, DateTimeValue, etc.) are defined
# in graphistry.compute.ast_temporal to avoid circular imports
