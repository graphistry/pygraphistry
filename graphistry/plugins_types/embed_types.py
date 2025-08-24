"""Type definitions for embedding-related functionality."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar, Union

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
else:
    np = Any

# Type variable for generic embedding protocol
TT = TypeVar('TT')

# Symbolic type for embedding protocol/model
ProtoSymbolic = Optional[Union[str, Callable[[TT, TT, TT], TT]]]

# Symbolic type for X feature input
XSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]

# Symbolic type for Y target/label input  
YSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]
