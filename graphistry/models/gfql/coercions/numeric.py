"""
Pure numeric type transformations

Currently numeric types don't need much transformation,
but this provides a consistent interface.
"""

from typing import Union
import numpy as np

from graphistry.models.gfql.types.numeric import NativeNumeric


def to_native(val: NativeNumeric) -> NativeNumeric:
    """Numeric values are already native, just pass through"""
    return val


def to_ast(val: NativeNumeric) -> NativeNumeric:
    """Numeric values don't have special AST representation, pass through"""
    return val


def to_wire(val: NativeNumeric) -> Union[int, float]:
    """Convert numeric to JSON-compatible type"""
    if isinstance(val, np.number):
        # Convert numpy types to Python native
        if isinstance(val, (np.integer, np.int_)):
            return int(val)
        else:
            return float(val)
    return val
