from typing import Any, List, Union
from datetime import date, datetime, time
import numpy as np
import pandas as pd

from graphistry.compute.typing import SeriesT
from graphistry.models.gfql.coercions.temporal import to_native
from graphistry.models.gfql.types.guards import is_any_temporal, is_basic_scalar
from graphistry.models.gfql.types.predicates import IsInElementInput
from graphistry.models.gfql.types.temporal import DateTimeWire, DateWire, TemporalWire, TimeWire
from graphistry.utils.json import assert_json_serializable
from .ASTPredicate import ASTPredicate
from .types import NormalizedIsInElement, NormalizedNumeric, NormalizedScalar


class IsIn(ASTPredicate):
    def __init__(self, options: List[IsInElementInput]) -> None:
        self.options = self._normalize_options(options)

    def _normalize_options(
        self, options: List[IsInElementInput]
    ) -> List['NormalizedIsInElement']:
        """Normalize options list to handle temporal values"""
        normalized = []
        has_temporal = False
        has_numeric = False

        for val in options:
            norm_val = self._normalize_value(val)
            normalized.append(norm_val)

            # Track types for validation
            if isinstance(norm_val, (pd.Timestamp, date, time)):
                has_temporal = True
            elif isinstance(norm_val, (int, float, np.number)):
                has_numeric = True

        # Validate no mixing of temporal and numeric types
        if has_temporal and has_numeric:
            raise ValueError(
                "Cannot mix temporal and numeric values in is_in. "
                "Found both temporal and numeric values."
            )

        return normalized

    def _normalize_value(
        self, val: IsInElementInput
    ) -> 'NormalizedIsInElement':
        """Convert various input types to internal representation"""
        # IsIn predicate needs:
        # - Basic scalars (including strings) as-is
        # - Temporals as native/pandas types (for .isin() method)
        # - Everything else passes through
        if is_basic_scalar(val):
            return val
        elif is_any_temporal(val):
            return to_native(val)
        else:
            # At this point, val should only be wire format dicts that aren't temporal
            # Since is_any_temporal catches TemporalValue, we can safely cast
            return val  # type: ignore[return-value]

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
            other_opts = [
                opt for opt in self.options if not isinstance(opt, time)
            ]

            if time_opts:
                time_matches = s.dt.time.isin(time_opts)
                if other_opts:
                    # Also check other values
                    other_matches = s.isin(other_opts)
                    return time_matches | other_matches
                else:
                    return time_matches

        return s.isin(self.options)

    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.options, list):
            raise GFQLTypeError(
                ErrorCode.E201,
                "options must be a list",
                field="options",
                value=type(self.options).__name__
            )
        
        # Check normalized options are JSON serializable
        # (temporal values are converted to pandas types which are serializable)
        try:
            # Create a test list with JSON-compatible versions
            json_test: List[Any] = []
            for opt in self.options:
                if isinstance(opt, pd.Timestamp):
                    json_test.append(opt.isoformat())
                elif isinstance(opt, (date, time)):
                    json_test.append(str(opt))
                elif isinstance(opt, dict):
                    # Handle wire format temporal types
                    json_test.append(opt)
                else:
                    # Handle numeric, string, None types
                    json_test.append(opt)
            assert_json_serializable(json_test)
        except Exception as e:
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Options not JSON serializable: {e}",
                field="options",
                value=str(self.options)
            )

    def to_json(self, validate=True) -> dict:
        """Override to handle temporal values in options"""
        if validate:
            self.validate()

        # Convert temporal values back to tagged dicts
        json_options: List[Any] = []
        for opt in self.options:
            if isinstance(opt, pd.Timestamp):
                # Convert back to tagged dict
                datetime_wire: DateTimeWire = {
                    "type": "datetime",
                    "value": opt.isoformat(),
                    "timezone": str(opt.tz) if opt.tz else "UTC"
                }
                json_options.append(datetime_wire)
            elif isinstance(opt, date) and not isinstance(opt, datetime):
                date_wire: DateWire = {
                    "type": "date",
                    "value": opt.isoformat()
                }
                json_options.append(date_wire)
            elif isinstance(opt, time):
                time_wire: TimeWire = {
                    "type": "time",
                    "value": opt.isoformat()
                }
                json_options.append(time_wire)
            else:
                json_options.append(opt)

        return {
            'type': self.__class__.__name__,
            'options': json_options
        }


def is_in(options: List[IsInElementInput]) -> IsIn:
    return IsIn(options)
