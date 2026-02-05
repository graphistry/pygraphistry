import pandas as pd
from typing import Any, TYPE_CHECKING, TypeVar, Union

# TODO stubs for Union[cudf.DataFrame, dask.DataFrame, ..] at checking time
if TYPE_CHECKING:
    DataFrameT = pd.DataFrame
    SeriesT = pd.Series
    DomainT = Union[pd.Index, pd.Series]
else:
    DataFrameT = Any
    SeriesT = Any
    DomainT = Any

# Type variable for return type preservation in predicates
T = TypeVar('T')
