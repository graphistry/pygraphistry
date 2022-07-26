import pandas as pd
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

def df_to_pdf(df, engine: Engine):
    if engine == Engine.PANDAS:
        return df
    elif engine == Engine.CUDF:
        return df.to_pandas()
    elif engine == Engine.DASK:
        return df.compute()
    raise ValueError('Only engines pandas/cudf supported')

def df_to_engine(df, engine: Engine):
    if engine == Engine.PANDAS:
        if isinstance(df, pd.DataFrame):
            return df
        else:
            return df.to_pandas()
    elif engine == Engine.CUDF:
        import cudf
        if isinstance(df, cudf.DataFrame):
            return df
        else:
            return cudf.DataFrame.from_pandas(df)
    else:
        raise ValueError('Only engines pandas/cudf supported')

def df_concat(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.concat
    elif engine == Engine.CUDF:
        import cudf
        return cudf.concat

def df_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame
    elif engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame
