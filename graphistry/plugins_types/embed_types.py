"""Type definitions for embedding-related functionality."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar, Union

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
    from torch import Tensor as TT
else:
    np = Any
    TT = Any

XSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]
ProtoSymbolic = Optional[Union[str, Callable[[TT, TT, TT], TT]]]

YSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]
