"""
Type definitions for numeric values in GFQL

Numeric data representations:
1. Native types - Host language numeric types (int, float, numpy types)
2. AST types - For numeric predicates, native types are used directly
3. Wire types - JSON number format (no special encoding needed)
"""

from typing import Union
import numpy as np


# ============= Native Numeric Types =============
# Host language numeric types

NativeInt = int
NativeFloat = float
NativeNumeric = Union[int, float, np.number]


# ============= Wire Types (JSON) =============
# For numeric values, JSON numbers map directly to native types
# No special wire format needed - JSON handles int/float natively

WireNumeric = Union[int, float]


# ============= AST Types =============
# Numeric predicates use native types directly in the AST
# No special wrapper needed unlike temporal values

ASTNumeric = NativeNumeric
