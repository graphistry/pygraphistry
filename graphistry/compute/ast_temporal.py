from typing import Any, Dict, Optional, Union, cast
from abc import ABC, abstractmethod
from datetime import date, datetime, time
from datetime import timezone as py_timezone
from dateutil import parser as date_parser  # type: ignore[import]
from datetime import tzinfo as py_tzinfo
import pandas as pd
import sys

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo as _ZoneInfoImport
    ZoneInfo = cast(Any, _ZoneInfoImport)
else:
    ZoneInfo = cast(Any, None)

try:
    from dateutil.tz import gettz as _dateutil_gettz  # type: ignore[import-untyped]
except Exception:
    _dateutil_gettz = None

from graphistry.models.gfql.types.temporal import DateTimeWire, DateWire, TimeWire, TemporalWire
from graphistry.utils.json import JSONVal


def _resolve_timezone(timezone: str) -> Optional[py_tzinfo]:
    if timezone.upper() == "UTC":
        return py_timezone.utc
    if ZoneInfo is not None:
        try:
            return ZoneInfo(timezone)
        except Exception:
            pass
    if _dateutil_gettz is not None:
        return _dateutil_gettz(timezone)
    return None


class TemporalValue(ABC):
    """Base class for temporal values with tagging support"""
    
    @abstractmethod
    def to_json(self) -> TemporalWire:
        """Serialize to JSON-compatible dictionary"""
        pass
    
    @abstractmethod
    def as_pandas_value(self) -> Any:
        """Convert to pandas-compatible value for comparison"""
        pass


class DateTimeValue(TemporalValue):
    """Tagged datetime value with timezone support"""
    
    def __init__(self, value: str, timezone: str = "UTC"):
        self.value = value
        self.timezone = timezone
        self._parsed = self._parse_iso8601(value, timezone)
    
    @classmethod
    def from_pandas_timestamp(cls, ts: pd.Timestamp) -> 'DateTimeValue':
        """Create from pandas Timestamp"""
        tz = str(ts.tz) if ts.tz else "UTC"
        value = ts.isoformat()
        return cls(value, tz)
    
    @classmethod
    def from_datetime(cls, dt: datetime) -> 'DateTimeValue':
        """Create from Python datetime"""
        tz = str(dt.tzinfo) if dt.tzinfo else "UTC"
        value = dt.isoformat()
        return cls(value, tz)
    
    def _parse_iso8601(self, value: str, timezone: str) -> pd.Timestamp:
        """Parse ISO8601 datetime string with timezone"""
        # Parse the datetime
        dt = date_parser.isoparse(value)
        
        # Handle timezone
        ts = pd.Timestamp(dt)
        tzinfo = _resolve_timezone(timezone) or timezone
        if dt.tzinfo is None:
            # Naive datetime - localize to specified timezone
            ts = ts.tz_localize(tzinfo)
        else:
            # Already has timezone - convert to specified timezone
            ts = ts.tz_convert(tzinfo)

        return ts
    
    def to_json(self) -> DateTimeWire:
        """Return dict for tagged temporal value"""
        result: DateTimeWire = {
            "type": "datetime",
            "value": self.value,
            "timezone": self.timezone
        }
        return result
    
    def as_pandas_value(self) -> pd.Timestamp:
        return self._parsed


class DateValue(TemporalValue):
    """Tagged date value"""
    
    def __init__(self, value: str):
        self.value = value
        self._parsed = self._parse_date(value)
    
    @classmethod
    def from_date(cls, d: date) -> 'DateValue':
        """Create from Python date"""
        return cls(d.isoformat())
    
    def _parse_date(self, value: str) -> date:
        """Parse date string in ISO format (YYYY-MM-DD)"""
        return date_parser.isoparse(value).date()
    
    def to_json(self) -> DateWire:
        """Return dict for tagged temporal value"""
        result: DateWire = {
            "type": "date",
            "value": self.value
        }
        return result
    
    def as_pandas_value(self) -> pd.Timestamp:
        # Convert date to pandas Timestamp at midnight
        return pd.Timestamp(self._parsed)


class TimeValue(TemporalValue):
    """Tagged time value"""
    
    def __init__(self, value: str):
        self.value = value
        self._parsed = self._parse_time(value)
    
    @classmethod
    def from_time(cls, t: time) -> 'TimeValue':
        """Create from Python time"""
        return cls(t.isoformat())
    
    def _parse_time(self, value: str) -> time:
        """Parse time string in ISO format (HH:MM:SS)"""
        # Handle time-only strings
        if 'T' not in value and ' ' not in value:
            # Pure time string like "14:30:00"
            return datetime.strptime(value, "%H:%M:%S").time()
        else:
            # Extract time from full datetime
            return date_parser.isoparse(value).time()
    
    def to_json(self) -> TimeWire:
        """Return dict for tagged temporal value"""
        result: TimeWire = {
            "type": "time", 
            "value": self.value
        }
        return result
    
    def as_pandas_value(self) -> time:
        return self._parsed


def temporal_value_from_json(d: Dict[str, Any]) -> TemporalValue:
    """Factory function to create temporal value from JSON dict"""
    if d["type"] == "datetime":
        return DateTimeValue(d["value"], d.get("timezone", "UTC"))
    elif d["type"] == "date":
        return DateValue(d["value"])
    elif d["type"] == "time":
        return TimeValue(d["value"])
    else:
        raise ValueError(f"Unknown temporal value type: {d['type']}")
