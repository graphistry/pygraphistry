from abc import ABC, abstractmethod
from datetime import datetime, date, time
from typing import Dict, Any
import pandas as pd
from dateutil import parser as date_parser
import pytz

from graphistry.utils.json import JSONVal


class TemporalValue(ABC):
    """Base class for temporal values with tagging support"""
    
    @abstractmethod
    def to_json(self) -> Dict[str, JSONVal]:
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
        if dt.tzinfo is None:
            # Naive datetime - localize to specified timezone
            tz = pytz.timezone(timezone)
            dt = tz.localize(dt)
        else:
            # Already has timezone - convert to specified timezone
            tz = pytz.timezone(timezone)
            dt = dt.astimezone(tz)
        
        return pd.Timestamp(dt)
    
    def to_json(self) -> Dict[str, JSONVal]:
        """Return dict for tagged temporal value"""
        return {
            "type": "datetime",
            "value": self.value,
            "timezone": self.timezone
        }
    
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
    
    def to_json(self) -> Dict[str, JSONVal]:
        """Return dict for tagged temporal value"""
        return {
            "type": "date",
            "value": self.value
        }
    
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
    
    def to_json(self) -> Dict[str, JSONVal]:
        """Return dict for tagged temporal value"""
        return {
            "type": "time", 
            "value": self.value
        }
    
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
