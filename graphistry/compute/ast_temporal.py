from typing import ClassVar, Mapping, Optional, Union, cast
from abc import ABC, abstractmethod
import math
from datetime import date, datetime, time
from datetime import timezone as py_timezone
from dateutil import parser as date_parser  # type: ignore[import]
from datetime import tzinfo as py_tzinfo
import numpy as np
import pandas as pd

from graphistry.models.gfql.types.temporal import DateTimeWire, TemporalWire
from graphistry.compute.gfql.temporal.constructors import _named_timezone


def _resolve_timezone(timezone: str) -> Optional[py_tzinfo]:
    if timezone.upper() == "UTC":
        return py_timezone.utc
    return _named_timezone(timezone)


# Epoch units are explicit because guessing s vs ms silently misreads timestamps.
EPOCH_UNITS = ("s", "ms")
# Wire epoch integers are JS-safe; native integers are exact and serialization is a string.
JS_MAX_SAFE_INTEGER = 2**53 - 1
EpochCount = Union[int, float, np.integer, np.floating]


def _epoch_validation_error(message: str, *, field: str, value: object, suggestion: str) -> "Exception":
    from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
    return GFQLTypeError(
        ErrorCode.E201,
        message,
        field=field,
        value=repr(value),
        suggestion=suggestion,
    )


def _validated_epoch_count(value: EpochCount, *, from_wire: bool) -> int:
    """Validate a numeric epoch count and return it as an exact int.

    Rules (shared Python/JSON boundary):
    - booleans are never counts; nonfinite values are never counts;
    - integral floats are accepted and converted exactly (1e3 == 1000); fractional
      values are rejected (a fractional second must be expressed in a smaller unit);
    - native integers are exact and only checked for timestamp representability;
    - floats (native and NumPy, any width) are validated in their own precision, then
      capped at the same +/- (2**53 - 1) exact-integer range as wire numerics;
    - WIRE numeric inputs are additionally capped at |v| <= 2**53 - 1 for
      predictable cross-language (JS) interoperability.
    No rounding, wraparound, or NaT substitution happens anywhere.
    """
    if isinstance(value, (bool, np.bool_)):
        raise _epoch_validation_error(
            "Epoch count must be an integer or integral float, not a boolean",
            field="value", value=value,
            suggestion="Pass an integer epoch count in the declared unit",
        )
    if isinstance(value, (int, np.integer)):
        count = int(value)
    elif isinstance(value, np.floating):
        # Validate the original extended-precision value before narrowing (R3).
        if not np.isfinite(value):
            raise _epoch_validation_error(
                "Epoch count must be finite",
                field="value", value=value,
                suggestion="Pass a finite integer or integral float epoch count",
            )
        if value != np.floor(value):
            raise _epoch_validation_error(
                "Epoch count must be an integral value",
                field="value", value=value,
                suggestion="A fractional second must be expressed as a count in a smaller unit (e.g. 1.5 s as 1500 ms)",
            )
        try:
            f = float(value)
            count = int(f)
        except (OverflowError, ValueError):
            raise _epoch_validation_error(
                "Epoch count is outside the exactly representable float range",
                field="value", value=value,
                suggestion="Pass a native integer epoch count for counts beyond the float-exact range",
            )
        if not math.isfinite(f) or abs(count) > JS_MAX_SAFE_INTEGER:
            raise _epoch_validation_error(
                f"Float epoch count exceeds the exact integer range (+/- 2**53 - 1): {count}",
                field="value", value=value,
                suggestion="Pass a native integer epoch count for counts beyond +/- (2**53 - 1); floats are capped there",
            )
    elif isinstance(value, float):
        f = value
        if not math.isfinite(f):
            raise _epoch_validation_error(
                "Epoch count must be finite",
                field="value", value=value,
                suggestion="Pass a finite integer or integral float epoch count",
            )
        if not f.is_integer():
            raise _epoch_validation_error(
                "Epoch count must be an integral value",
                field="value", value=value,
                suggestion="A fractional second must be expressed as a count in a smaller unit (e.g. 1.5 s as 1500 ms)",
            )
        count = int(f)
        # Native floats use the wire exact-integer limit; native ints stay exact.
        if abs(count) > JS_MAX_SAFE_INTEGER:
            raise _epoch_validation_error(
                f"Float epoch count exceeds the exact integer range (+/- 2**53 - 1): {count}",
                field="value", value=value,
                suggestion="Pass a native integer epoch count for counts beyond +/- (2**53 - 1); floats are capped there",
            )
    else:
        raise _epoch_validation_error(
            f"Epoch count must be an integer or integral float, got {type(value).__name__}",
            field="value", value=value,
            suggestion="Pass an integer epoch count in the declared unit",
        )
    if from_wire and abs(count) > JS_MAX_SAFE_INTEGER:
        raise _epoch_validation_error(
            f"Numeric JSON epoch count exceeds the interoperable integer range (+/- 2**53 - 1): {count}",
            field="value", value=value,
            suggestion="Send epoch counts within +/- (2**53 - 1), or serialize the datetime as an ISO string",
        )
    return count


def _utc_timestamp_from_epoch_count(count: int, unit: str) -> pd.Timestamp:
    """Convert a validated epoch count in 'unit' to an exact UTC pd.Timestamp."""
    if unit not in EPOCH_UNITS:
        raise _epoch_validation_error(
            f"unit must be 's' or 'ms', got {unit!r}",
            field="unit", value=unit,
            suggestion="Use unit='s' (seconds) or unit='ms' (milliseconds); there is no default unit",
        )
    try:
        return pd.Timestamp(count, unit=unit, tz="UTC")
    except (ValueError, OverflowError):
        raise _epoch_validation_error(
            f"Epoch count {count} (unit '{unit}') is outside the representable datetime range",
            field="value", value=count,
            suggestion="Use a count within the engine's datetime range",
        )


class TemporalValue(ABC):
    """Base class for temporal values with tagging support"""
    
    @abstractmethod
    def to_json(self) -> TemporalWire:
        """Serialize to JSON-compatible dictionary"""
        pass
    
    @abstractmethod
    def as_pandas_value(self) -> Union[pd.Timestamp, date, time]:
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

    @classmethod
    def from_epoch(cls, value: EpochCount, *, unit: str) -> 'DateTimeValue':
        """Create from a UTC epoch count with an explicitly declared unit.

        ``unit`` is required and must be ``"s"`` or ``"ms"``; there is intentionally no
        default unit, because a bare number does not express an instant and a guessed
        default silently misreads data. Epoch counts denote UTC instants (no timezone
        parameter); the result is the same DateTimeValue an equivalent ISO UTC string
        would produce, so epoch-origin and ISO-origin values behave identically.
        """
        count = _validated_epoch_count(value, from_wire=False)
        ts = _utc_timestamp_from_epoch_count(count, unit)
        # Validate the full ISO representation, including NaT and year-10000 (R2).
        if pd.isna(ts):
            raise _epoch_validation_error(
                f"Epoch count {count} (unit '{unit}') resolves to the NaT sentinel",
                field="value", value=count,
                suggestion="Use a count whose datetime is not NaT",
            )
        iso = ts.isoformat()
        try:
            reparsed = date_parser.isoparse(iso)
        except (ValueError, OverflowError) as error:
            raise _epoch_validation_error(
                f"Epoch count {count} (unit '{unit}') is outside the representable datetime range: {error}",
                field="value", value=count,
                suggestion="Use a count within the engine's datetime range",
            )
        if pd.Timestamp(reparsed) != ts:
            raise _epoch_validation_error(
                f"Epoch count {count} (unit '{unit}') does not round-trip through its ISO representation",
                field="value", value=count,
                suggestion="Use a count within the engine's datetime range",
            )
        return cls(iso, "UTC")
    
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


class _ScalarTemporalValue(TemporalValue):
    _wire_type: ClassVar[str]

    def __init__(self, value: str):
        self.value = value
        self._parsed = self._parse_value(value)

    @abstractmethod
    def _parse_value(self, value: str) -> Union[pd.Timestamp, date, time]:
        pass

    def to_json(self) -> TemporalWire:
        return cast(TemporalWire, {"type": self._wire_type, "value": self.value})


class DateValue(_ScalarTemporalValue):
    """Tagged date value"""

    _wire_type = "date"
    
    @classmethod
    def from_date(cls, d: date) -> 'DateValue':
        """Create from Python date"""
        return cls(d.isoformat())
    
    def _parse_value(self, value: str) -> date:
        """Parse date string in ISO format (YYYY-MM-DD)"""
        return date_parser.isoparse(value).date()
    
    def as_pandas_value(self) -> pd.Timestamp:
        # Convert date to pandas Timestamp at midnight
        parsed = self._parsed
        if isinstance(parsed, date) and not isinstance(parsed, datetime):
            return pd.Timestamp(parsed)
        raise TypeError(f"Date parser returned non-date value: {type(parsed).__name__}")


class TimeValue(_ScalarTemporalValue):
    """Tagged time value"""

    _wire_type = "time"
    
    @classmethod
    def from_time(cls, t: time) -> 'TimeValue':
        """Create from Python time"""
        return cls(t.isoformat())
    
    def _parse_value(self, value: str) -> time:
        """Parse time string in ISO format (HH:MM:SS)"""
        # Handle time-only strings
        if 'T' not in value and ' ' not in value:
            # Pure time string like "14:30:00"
            return datetime.strptime(value, "%H:%M:%S").time()
        else:
            # Extract time from full datetime
            return date_parser.isoparse(value).time()
    
    def as_pandas_value(self) -> time:
        parsed = self._parsed
        if isinstance(parsed, time):
            return parsed
        raise TypeError(f"Time parser returned non-time value: {type(parsed).__name__}")


def temporal_value_from_json(d: Mapping[str, object]) -> TemporalValue:
    """Factory function to create temporal value from JSON dict.

    Datetime inputs come in exactly two variants:
    - ISO: string ``value`` with optional ``timezone`` (established behavior) and no ``unit``;
    - epoch: integer-valued numeric ``value`` with required ``unit`` ("s"|"ms"), no
      ``timezone`` (epoch counts are UTC instants by definition), additionally capped
      at the JSON-safe integer range.
    """
    if d["type"] == "datetime":
        if "value" not in d:
            raise _epoch_validation_error(
                "datetime value is required",
                field="value", value=d,
                suggestion='Pass {"type": "datetime", "value": <ISO string | epoch count>, ...}',
            )
        value = d["value"]
        unit = d.get("unit")
        if isinstance(value, str):
            if unit is not None:
                raise _epoch_validation_error(
                    "unit is not allowed on an ISO datetime string value",
                    field="unit", value=unit,
                    suggestion="Remove unit, or send an explicit epoch value {\"type\": \"datetime\", \"value\": <int>, \"unit\": \"s\"|\"ms\"}",
                )
            timezone = d.get("timezone", "UTC")
            if not isinstance(timezone, str):
                raise _epoch_validation_error(
                    "datetime timezone must be a string",
                    field="timezone", value=timezone,
                    suggestion="Pass a timezone name such as 'UTC'",
                )
            return DateTimeValue(value, timezone)
        if "timezone" in d:
            raise _epoch_validation_error(
                "timezone is not allowed on a numeric epoch datetime value",
                field="timezone", value=d.get("timezone"),
                suggestion="Epoch counts denote UTC instants by definition; use an ISO string value to declare a timezone",
            )
        if not isinstance(unit, str) or unit not in EPOCH_UNITS:
            raise _epoch_validation_error(
                "numeric datetime value requires unit 's' or 'ms'",
                field="unit", value=unit,
                suggestion='Use {"type": "datetime", "value": <count>, "unit": "s"|"ms"}; there is no default unit',
            )
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise _epoch_validation_error(
                "Numeric datetime value must be an integer or integral float",
                field="value", value=value,
                suggestion='Use {"type": "datetime", "value": <count>, "unit": "s"|"ms"}',
            )
        count = _validated_epoch_count(value, from_wire=True)
        return DateTimeValue.from_epoch(count, unit=unit)
    elif d["type"] == "date":
        date_value = d["value"]
        if not isinstance(date_value, str):
            raise _epoch_validation_error(
                "date value must be a string",
                field="value", value=date_value,
                suggestion="Pass an ISO date string such as '2020-01-01'",
            )
        return DateValue(date_value)
    elif d["type"] == "time":
        time_value = d["value"]
        if not isinstance(time_value, str):
            raise _epoch_validation_error(
                "time value must be a string",
                field="value", value=time_value,
                suggestion="Pass an ISO time string such as '14:30:00'",
            )
        return TimeValue(time_value)
    else:
        raise ValueError(f"Unknown temporal value type: {d['type']}")
