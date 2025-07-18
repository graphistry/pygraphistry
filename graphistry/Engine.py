from inspect import getmodule
import warnings
import numpy as np
import pandas as pd
from typing import Any, Optional, Union
from enum import Enum


class Engine(Enum):
    PANDAS = 'pandas'
    CUDF = 'cudf'
    DASK = 'dask'
    DASK_CUDF = 'dask_cudf'

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

    from graphistry.utils.lazy_import import lazy_cudf_import

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    # if an Engine (concrete), just use that
    if engine != EngineAbstract.AUTO:
        return Engine(engine.value)

    if g_or_df is not None:
        # Use dynamic import to avoid Jinja dependency issues from pandas df.style getter
        is_plottable = False
        try:
            from graphistry.plotter import Plotter
            is_plottable = isinstance(g_or_df, Plotter)
        except ImportError:
            pass

        if not is_plottable:
            # Also check Plottable base class
            try:
                from graphistry.Plottable import Plottable
                is_plottable = isinstance(g_or_df, Plottable)
            except ImportError:
                pass
        
        if is_plottable:
            if g_or_df._nodes is not None and g_or_df._edges is not None:
                if not isinstance(g_or_df._nodes, type(g_or_df._edges)):
                    #raise ValueError(f'Edges and nodes must be same type for auto engine selection, got: {type(g_or_df._edges)} and {type(g_or_df._nodes)}')
                    warnings.warn(f'Edges and nodes must be same type for auto engine selection, got: {type(g_or_df._edges)} and {type(g_or_df._nodes)}')                
            g_or_df = g_or_df._edges if g_or_df._edges is not None else g_or_df._nodes
    
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

def s_to_arr(engine: Engine):
    """
    cudf.Series -> to_cupy()
    pandas.Series -> to_numpy()
    """
    if engine == Engine.PANDAS:
        return lambda x: x.to_numpy()
    elif engine == Engine.CUDF:
        return lambda x: x.to_cupy()
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

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
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def df_concat(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.concat
    elif engine == Engine.CUDF:
        import cudf
        return cudf.concat
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def df_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame
    elif engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.Series
    elif engine == Engine.CUDF:
        import cudf
        return cudf.Series
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_sqrt(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.sqrt
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.sqrt
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_arange(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.arange
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.arange
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_full(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.full
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.full
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_pi(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.pi
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.pi
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_concatenate(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.concatenate
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.concatenate
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_sin(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.sin
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.sin
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_cos(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.cos
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.cos
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_series(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.Series
    elif engine == Engine.CUDF:
        import cudf
        return cudf.Series
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_minimum(engine: Engine):
    if engine == Engine.PANDAS:
        return np.minimum
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.minimum
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_floor(engine: Engine):
    if engine == Engine.PANDAS:
        import numpy as np
        return np.floor
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.floor
    else:
        raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_cumsum(engine: Engine):
    if engine == Engine.PANDAS:
        return lambda x: x.cumsum()
    elif engine == Engine.CUDF:
        return lambda x: x.cumsum()
    else:
        raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_isna(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.isna
    elif engine == Engine.CUDF:
        import cudf
        return lambda x: x.isnull()
    else:
        raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')

def s_maximum(engine: Engine):
    if engine == Engine.PANDAS:
        return np.maximum
    elif engine == Engine.CUDF:
        import cupy as cp
        return cp.maximum
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')
