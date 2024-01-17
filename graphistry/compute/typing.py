import pandas as pd
from typing import Any, TYPE_CHECKING

# TODO stubs for Union[cudf.DataFrame, dask.DataFrame, ..] at checking time
if TYPE_CHECKING:
    DataFrameT = pd.DataFrame
else:
    DataFrameT = Any
