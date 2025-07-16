from typing import Dict, Literal, TYPE_CHECKING, Union, cast, overload
from datetime import date, datetime, time
import numpy as np
import pandas as pd

from graphistry.compute.ast_temporal import DateTimeValue, DateValue, TemporalValue, TimeValue
from graphistry.compute.typing import SeriesT
from graphistry.models.gfql.coercions.temporal import to_ast
from graphistry.models.gfql.types.guards import is_any_temporal, is_native_numeric, is_string
from graphistry.models.gfql.types.predicates import BetweenBoundInput, ComparisonInput
from graphistry.utils.json import JSONVal
from .ASTPredicate import ASTPredicate

if TYPE_CHECKING:
    # Note: pd.Series[T] syntax not supported in Python 3.8
    pass


class ComparisonPredicate(ASTPredicate):
    """Base class for comparison predicates that support both numeric and temporal values"""
    
    def __init__(self, val: ComparisonInput) -> None:
        self.val = self._normalize_value(val)
    
    def _normalize_value(self, val: ComparisonInput) -> Union[int, float, np.number, TemporalValue]:
        """Convert various input types to internal representation"""
        # Comparison predicates need:
        # - Numerics as-is
        # - Temporals as AST objects (for timezone handling)
        # - Strings rejected (ambiguous)
        if is_native_numeric(val):
            return val
        elif is_any_temporal(val):
            # to_ast always returns TemporalValue for valid temporal inputs
            temporal_val = to_ast(val)
            return temporal_val
        elif is_string(val):
            raise ValueError(
                f"Raw string '{val}' is ambiguous. Use:\n"
                f"  - pd.Timestamp('{val}') for datetime\n"
                f"  - {{'type': 'datetime', 'value': '{val}'}} for explicit type"
            )
        else:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(val)}")
    
    def _prepare_temporal_series(self, s: SeriesT, temporal_val: TemporalValue) -> SeriesT:
        """Prepare series for temporal comparison by extracting/converting as needed"""
        if isinstance(temporal_val, DateTimeValue):
            # Normalize series to target timezone for comparison
            if hasattr(s, 'dt') and hasattr(s.dt, 'tz_localize'):
                if s.dt.tz is None:
                    return s.dt.tz_localize('UTC').dt.tz_convert(temporal_val.timezone)
                else:
                    return s.dt.tz_convert(temporal_val.timezone)
            return s
        
        elif isinstance(temporal_val, DateValue):
            # Extract date from datetime series if needed
            if hasattr(s, 'dt'):
                return s.dt.date
            return s
        
        elif isinstance(temporal_val, TimeValue):
            # Extract time from datetime series if needed
            if hasattr(s, 'dt'):
                return s.dt.time
            return s
        
        raise TypeError(f"Unknown temporal value type: {type(temporal_val)}")
    
    def _get_temporal_comparison_value(self, temporal_val: TemporalValue) -> Union[pd.Timestamp, date, time]:
        """Get the appropriate comparison value from a TemporalValue"""
        if isinstance(temporal_val, DateTimeValue):
            return temporal_val.as_pandas_value()
        elif isinstance(temporal_val, (DateValue, TimeValue)):
            return temporal_val._parsed
        raise TypeError(f"Unknown temporal value type: {type(temporal_val)}")
    
    
    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.val, (int, float, TemporalValue)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "val must be numeric or temporal",
                field="val",
                value=type(self.val).__name__,
                suggestion="Use numeric values or temporal objects"
            )
    
    def to_json(self, validate=True) -> dict:
        """Serialize maintaining backward compatibility"""
        if validate:
            self.validate()
        
        result: Dict[str, JSONVal] = {"type": self.__class__.__name__}
        
        if isinstance(self.val, TemporalValue):
            # to_json() returns a dict, not a string
            val_dict = self.val.to_json()
            result["val"] = cast(JSONVal, val_dict)
        else:
            result["val"] = cast(JSONVal, self.val)
        
        return result


# For backward compatibility - keep but deprecated
class NumericASTPredicate(ComparisonPredicate):
    """Deprecated: Use ComparisonPredicate instead"""
    pass

###

class GT(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Greater than comparison"""
        if isinstance(self.val, (int, float)):
            return s > self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s > comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def gt(val: ComparisonInput) -> GT:
    """
    Return whether a given value is greater than a threshold
    """
    return GT(val)

class LT(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Less than comparison"""
        if isinstance(self.val, (int, float)):
            return s < self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s < comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def lt(val: ComparisonInput) -> LT:
    """
    Return whether a given value is less than a threshold
    """
    return LT(val)

class GE(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Greater than or equal comparison"""
        if isinstance(self.val, (int, float)):
            return s >= self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s >= comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def ge(val: ComparisonInput) -> GE:
    """
    Return whether a given value is greater than or equal to a threshold
    """
    return GE(val)

class LE(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Less than or equal comparison"""
        if isinstance(self.val, (int, float)):
            return s <= self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s <= comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def le(val: ComparisonInput) -> LE:
    """
    Return whether a given value is less than or equal to a threshold
    """
    return LE(val)

class EQ(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Equal comparison"""
        if isinstance(self.val, (int, float)):
            return s == self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s == comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def eq(val: ComparisonInput) -> EQ:
    """
    Return whether a given value is equal to a threshold
    """
    return EQ(val)

class NE(ComparisonPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        """Not equal comparison"""
        if isinstance(self.val, (int, float)):
            return s != self.val
        elif isinstance(self.val, TemporalValue):
            prepared_s = self._prepare_temporal_series(s, self.val)
            comparison_val = self._get_temporal_comparison_value(self.val)
            return prepared_s != comparison_val
        else:
            raise TypeError(f"Unexpected value type: {type(self.val)}")

def ne(val: ComparisonInput) -> NE:
    """
    Return whether a given value is not equal to a threshold
    """
    return NE(val)

class Between(ASTPredicate):
    def __init__(self, lower: BetweenBoundInput, 
                 upper: BetweenBoundInput, 
                 inclusive: bool = True) -> None:
        # Store original inputs for creating sub-predicates
        self.lower_input = lower
        self.upper_input = upper
        # Store normalized values for type checking
        self.lower = self._normalize_value(lower)
        self.upper = self._normalize_value(upper)
        self.inclusive = inclusive
    
    def _normalize_value(self, val: BetweenBoundInput) -> Union[int, float, np.number, TemporalValue]:
        """Convert various input types to internal representation"""
        # Same normalization as ComparisonPredicate
        if is_native_numeric(val):
            return val
        elif is_any_temporal(val):
            # to_ast always returns TemporalValue for valid temporal inputs
            temporal_val = to_ast(val)
            return temporal_val
        elif is_string(val):
            raise ValueError(
                f"Raw string '{val}' is ambiguous. Use:\n"
                f"  - pd.Timestamp('{val}') for datetime\n"
                f"  - {{'type': 'datetime', 'value': '{val}'}} for explicit type"
            )
        else:
            raise TypeError(f"Unsupported type for {self.__class__.__name__}: {type(val)}")

    def __call__(self, s: SeriesT) -> SeriesT:
        # Check if both bounds are same type
        lower_is_numeric = isinstance(self.lower, (int, float))
        upper_is_numeric = isinstance(self.upper, (int, float))
        lower_is_temporal = isinstance(self.lower, TemporalValue)
        upper_is_temporal = isinstance(self.upper, TemporalValue)
        
        if lower_is_numeric and upper_is_numeric:
            # Numeric comparison
            if self.inclusive:
                return (s >= self.lower) & (s <= self.upper)
            else:
                return (s > self.lower) & (s < self.upper)
        
        elif lower_is_temporal and upper_is_temporal:
            # Temporal comparison
            # Create comparison predicates using original inputs
            ge_pred = GE(self.lower_input)
            le_pred = LE(self.upper_input)
            gt_pred = GT(self.lower_input)
            lt_pred = LT(self.upper_input)
            
            if self.inclusive:
                return ge_pred(s) & le_pred(s)
            else:
                return gt_pred(s) & lt_pred(s)
        
        else:
            raise TypeError("Between requires both bounds to be same type (numeric or temporal)")
        
    def _validate_fields(self) -> None:
        """Validate predicate fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        # Check types match
        lower_is_numeric = isinstance(self.lower, (int, float))
        upper_is_numeric = isinstance(self.upper, (int, float))
        lower_is_temporal = isinstance(self.lower, TemporalValue)
        upper_is_temporal = isinstance(self.upper, TemporalValue)
        
        if not ((lower_is_numeric and upper_is_numeric) or (lower_is_temporal and upper_is_temporal)):
            raise GFQLTypeError(
                ErrorCode.E201,
                "Between requires both bounds to be same type (numeric or temporal)",
                field="bounds",
                value=f"lower={type(self.lower).__name__}, upper={type(self.upper).__name__}"
            )
        
        if not isinstance(self.inclusive, bool):
            raise GFQLTypeError(
                ErrorCode.E201,
                "inclusive must be boolean",
                field="inclusive",
                value=type(self.inclusive).__name__
            )
    
    def to_json(self, validate=True) -> dict:
        """Serialize maintaining backward compatibility"""
        if validate:
            self.validate()
        
        result: Dict[str, JSONVal] = {"type": self.__class__.__name__, "inclusive": self.inclusive}
        
        # Serialize lower/upper based on type
        if isinstance(self.lower, TemporalValue):
            result["lower"] = cast(JSONVal, self.lower.to_json())
        else:
            result["lower"] = cast(JSONVal, self.lower)
            
        if isinstance(self.upper, TemporalValue):
            result["upper"] = cast(JSONVal, self.upper.to_json())
        else:
            result["upper"] = cast(JSONVal, self.upper)
            
        return result

def between(lower: BetweenBoundInput, 
            upper: BetweenBoundInput, 
            inclusive: bool = True) -> Between:
    """
    Return whether a given value is between a lower and upper threshold
    """
    return Between(lower, upper, inclusive)

class IsNA(ASTPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.isna()

def isna() -> IsNA:
    """
    Return whether a given value is NA
    """
    return IsNA()


class NotNA(ASTPredicate):
    def __call__(self, s: SeriesT) -> SeriesT:
        return s.notna()

def notna() -> NotNA:
    """
    Return whether a given value is not NA
    """
    return NotNA()
