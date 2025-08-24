"""Type definitions for embedding-related functionality."""

from typing import TYPE_CHECKING, Any, Callable, List, Optional, TypeVar, Union
from graphistry.utils.lazy_import import lazy_embed_import

import pandas as pd

if TYPE_CHECKING:
    import numpy as np
else:
    np = Any

if TYPE_CHECKING:
    _, torch, _, _, _, _, _, _ = lazy_embed_import()
    TT = torch.Tensor
else:
    TT = Any
    torch = Any


XSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]
ProtoSymbolic = Optional[Union[str, Callable[[TT, TT, TT], TT]]]  # type: ignore

YSymbolic = Optional[Union[pd.DataFrame, 'np.ndarray', List[str]]]
