"""
Type detection utilities

Clear functions to check what kind of type a value is.
"""

from typing import Any, TYPE_CHECKING, Union
from datetime import date, datetime, time
import numpy as np
import pandas as pd

# Python 3.10+ has TypeGuard in typing module, older versions need typing_extensions
if TYPE_CHECKING:
    import sys
    if sys.version_info >= (3, 10):
        from typing import TypeGuard
    else:
        from typing_extensions import TypeGuard

from graphistry.compute.ast_temporal import TemporalValue
from .numeric import NativeNumeric
from .temporal import NativeTemporal, TemporalWire


# ============= Temporal Detection =============

def is_native_temporal(val: Any) -> "TypeGuard[NativeTemporal]":
    """Check if value is a native Python/Pandas temporal type"""
    return isinstance(val, (pd.Timestamp, datetime, date, time))


def is_ast_temporal(val: Any) -> "TypeGuard[TemporalValue]":
    """Check if value is an AST TemporalValue"""
    return isinstance(val, TemporalValue)


def is_wire_temporal(val: Any) -> "TypeGuard[TemporalWire]":
    """Check if value is a wire format temporal (tagged dict)"""
    return (
        isinstance(val, dict)
        and "type" in val
        and val["type"] in ["datetime", "date", "time"]
    )


def is_any_temporal(
    val: Any
) -> "TypeGuard[Union[NativeTemporal, TemporalValue, TemporalWire]]":
    """Check if value is any kind of temporal (native, AST, or wire)"""
    return (
        is_native_temporal(val) or is_ast_temporal(val) or is_wire_temporal(val)
    )


# ============= Numeric Detection =============

def is_native_numeric(val: Any) -> "TypeGuard[NativeNumeric]":
    """Check if value is a native numeric type"""
    return isinstance(val, (int, float, np.number))


def is_any_numeric(val: Any) -> bool:
    """Check if value is any kind of numeric (currently same as native)"""
    return is_native_numeric(val)


# ============= Other Detection =============

def is_string(val: Any) -> "TypeGuard[str]":
    """Check if value is a string"""
    return isinstance(val, str)


def is_none(val: Any) -> bool:
    """Check if value is None"""
    return val is None


def is_basic_scalar(val: Any) -> "TypeGuard[Union[int, float, str, np.number, None]]":
    """Check if value is a basic scalar (int, float, str, None)"""
    return isinstance(val, (int, float, str, np.number, type(None)))


def is_dict(val: Any) -> bool:
    """Check if value is a dictionary"""
    return isinstance(val, dict)


def is_tagged_dict(val: Any) -> bool:
    """Check if value is a tagged dictionary (has 'type' field)"""
    return isinstance(val, dict) and "type" in val
