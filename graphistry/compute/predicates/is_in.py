from typing import Any, List, Union
import pandas as pd
import numpy as np
from datetime import datetime, date, time

from graphistry.utils.json import assert_json_serializable
from .ASTPredicate import ASTPredicate
from .temporal_values import TemporalValue, DateTimeValue, DateValue, TimeValue
from graphistry.compute.typing import SeriesT


class IsIn(ASTPredicate):
    def __init__(self, options: List[Any]) -> None:
        self.options = self._normalize_options(options)
    
    def _normalize_options(self, options: List[Any]) -> List[Any]:
        """Normalize options list to handle temporal values"""
        normalized = []
        for val in options:
            normalized.append(self._normalize_value(val))
        return normalized
    
    def _normalize_value(self, val: Any) -> Any:
        """Convert various input types to internal representation"""
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
    
    def __call__(self, s: SeriesT) -> SeriesT:
        # Check if we have any temporal values in options
        has_temporal = any(
            isinstance(opt, (pd.Timestamp, date, time)) 
            for opt in self.options
        )
        
        if has_temporal and hasattr(s, 'dt'):
            # For datetime series with time-only values in options,
            # we need special handling
            time_opts = [opt for opt in self.options if isinstance(opt, time)]
            other_opts = [opt for opt in self.options if not isinstance(opt, time)]
            
            if time_opts:
                # Check time component
                time_matches = s.dt.time.isin(time_opts)
                if other_opts:
                    # Also check other values
                    other_matches = s.isin(other_opts)
                    return time_matches | other_matches
                else:
                    return time_matches
        
        return s.isin(self.options)
    
    def validate(self) -> None:
        assert isinstance(self.options, list)
        # Check that normalized options are still JSON serializable
        # (temporal values are converted to pandas types which are serializable)
        try:
            # Create a test list with JSON-compatible versions
            json_test = []
            for opt in self.options:
                if isinstance(opt, pd.Timestamp):
                    json_test.append(opt.isoformat())
                elif isinstance(opt, (date, time)):
                    json_test.append(str(opt))
                else:
                    json_test.append(opt)
            assert_json_serializable(json_test)
        except Exception as e:
            raise ValueError(f"Options not JSON serializable: {e}")

    def to_json(self, validate=True) -> dict:
        """Override to handle temporal values in options"""
        if validate:
            self.validate()
        
        # Convert temporal values back to tagged dicts for serialization
        json_options = []
        for opt in self.options:
            if isinstance(opt, pd.Timestamp):
                # Convert back to tagged dict
                json_options.append({
                    "type": "datetime",
                    "value": opt.isoformat(),
                    "timezone": str(opt.tz) if opt.tz else "UTC"
                })
            elif isinstance(opt, date) and not isinstance(opt, datetime):
                json_options.append({
                    "type": "date",
                    "value": opt.isoformat()
                })
            elif isinstance(opt, time):
                json_options.append({
                    "type": "time",
                    "value": opt.isoformat()
                })
            else:
                json_options.append(opt)
        
        return {
            'type': self.__class__.__name__,
            'options': json_options
        }


def is_in(options: List[Any]) -> IsIn:
    return IsIn(options)
