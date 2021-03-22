from typing import Any

from enum import Enum


class Engine(Enum):
    PANDAS : str = 'pandas'
    CUDF : str = 'cudf'
    DASK : str = 'dask'
    DASK_CUDF : str = 'dask_cudf'


DataframeLike = Any  # pdf, cudf, ddf, dgdf
DataframeLocalLike = Any  # pdf, cudf
GraphistryLke = Any
