from inspect import getmodule
import pandas as pd
from typing import Any, Optional, Union
from enum import Enum
from graphistry.utils.lazy_import import lazy_cudf_import


class Engine(Enum):
    PANDAS : str = 'pandas'
    CUDF : str = 'cudf'
    DASK : str = 'dask'
    DASK_CUDF : str = 'dask_cudf'

class EngineAbstract(Enum):
    PANDAS = Engine.PANDAS.value
    CUDF = Engine.CUDF.value
    DASK = Engine.DASK.value
    DASK_CUDF = Engine.DASK_CUDF.value
    AUTO = 'auto'


DataframeLike = Any  # pdf, cudf, ddf, dgdf
DataframeLocalLike = Any  # pdf, cudf
GraphistryLke = Any

def resolve_engine(
    engine: Union[EngineAbstract, str],
    g_or_df: Optional[Any] = None,
) -> Engine:

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    # if an Engine (concrete), just use that
    if engine != EngineAbstract.AUTO:
        return Engine(engine.value)

    if g_or_df is not None:
        # work around circular dependency
        from graphistry.Plottable import Plottable
        if isinstance(g_or_df, Plottable):
            if g_or_df._nodes is not None and g_or_df._edges is not None:
                if not isinstance(g_or_df._nodes, type(g_or_df._edges)):
                    raise ValueError(f'Edges and nodes must be same type for auto engine selection, got: {type(g_or_df._edges)} and {type(g_or_df._nodes)}')
            g_or_df = g_or_df._edges if g_or_df._edges is not None else g_or_df._nodes
    
    if g_or_df is not None:
        if isinstance(g_or_df, pd.DataFrame):
            return Engine.PANDAS

        if 'cudf.core.dataframe' in str(getmodule(g_or_df)):
            has_cudf_dependancy_, _, _ = lazy_cudf_import()
            if has_cudf_dependancy_:
                import cudf
                if isinstance(g_or_df, cudf.DataFrame):
                    return Engine.CUDF
                raise ValueError(f'Expected cudf dataframe, got: {type(g_or_df)}')
    
    has_cudf_dependancy_, _, _ = lazy_cudf_import()
    if has_cudf_dependancy_:
        return Engine.CUDF
    return Engine.PANDAS

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
    raise ValueError('Only engines pandas/cudf supported')

def df_concat(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.concat
    elif engine == Engine.CUDF:
        import cudf
        return cudf.concat
    raise NotImplementedError("Only pandas/cudf supported")

def df_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame
    elif engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame
    raise NotImplementedError("Only pandas/cudf supported")

def s_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.Series
    elif engine == Engine.CUDF:
        import cudf
        return cudf.Series
    raise NotImplementedError("Only pandas/cudf supported")
