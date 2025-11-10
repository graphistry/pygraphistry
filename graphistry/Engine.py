from inspect import getmodule
import warnings
import numpy as np
import pandas as pd
from typing import Any, List, Optional, Union
from typing_extensions import Literal
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


# Type alias for engine parameter - accepts both enum values and string literals
# Includes 'auto' for automatic detection
EngineAbstractType = Union[EngineAbstract, Literal['pandas', 'cudf', 'dask', 'dask_cudf', 'auto']]

DataframeLike = Any  # pdf, cudf, ddf, dgdf
DataframeLocalLike = Any  # pdf, cudf
GraphistryLke = Any

def resolve_engine(
    engine: EngineAbstractType,
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


# DataFrame type coercion primitives
# See issue #784: https://github.com/graphistry/pygraphistry/issues/784

def safe_concat(
    dfs: List[DataframeLike],
    engine: Union[Engine, EngineAbstract] = EngineAbstract.AUTO,
    ignore_index: bool = False,
    sort: bool = False
) -> DataframeLike:
    """
    Engine-aware DataFrame concatenation with automatic type conversion.

    Handles mixed pandas/cuDF DataFrames by converting all to the target engine
    before concatenation. Prevents TypeErrors from direct pandas/cuDF mixing.

    Args:
        dfs: List of DataFrames to concatenate (pandas or cuDF)
        engine: Target engine for result ('auto', 'pandas', or 'cudf')
               If 'auto', uses first DataFrame's type
        ignore_index: If True, do not use index values on concatenation axis
        sort: Sort non-concatenation axis if not already aligned

    Returns:
        Concatenated DataFrame in target engine type

    Raises:
        ValueError: If dfs is empty and engine is AUTO

    Examples:
        >>> import pandas as pd
        >>> from graphistry.Engine import EngineAbstract, safe_concat
        >>>
        >>> df1 = pd.DataFrame({'a': [1, 2]})
        >>> df2 = pd.DataFrame({'a': [3, 4]})
        >>> result = safe_concat([df1, df2], engine=EngineAbstract.PANDAS)
        >>> len(result)
        4

        >>> # With cuDF (if available)
        >>> import cudf
        >>> pdf = pd.DataFrame({'a': [1, 2]})
        >>> gdf = cudf.DataFrame({'a': [3, 4]})
        >>> # safe_concat handles mixed types - converts to target engine
        >>> result = safe_concat([pdf, gdf], engine=EngineAbstract.CUDF)
        >>> isinstance(result, cudf.DataFrame)
        True
    """
    # Handle empty list
    if len(dfs) == 0:
        if engine == EngineAbstract.AUTO:
            raise ValueError("Cannot infer engine from empty list - specify engine explicitly")
        # Return empty DataFrame of target type
        if isinstance(engine, EngineAbstract):
            engine_val = Engine(engine.value) if engine != EngineAbstract.AUTO else Engine.PANDAS
        else:
            engine_val = engine
        if engine_val == Engine.PANDAS:
            return pd.DataFrame()  # type: ignore
        elif engine_val == Engine.CUDF:
            import cudf
            return cudf.DataFrame()  # type: ignore
        else:
            raise ValueError(f"Unknown engine: {engine_val}")

    # Resolve target engine
    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    engine_concrete: Engine
    if engine == EngineAbstract.AUTO:
        # Use first DataFrame's engine
        engine_concrete = resolve_engine(EngineAbstract.AUTO, dfs[0])
    else:
        engine_concrete = Engine(engine.value)

    # Convert all DataFrames to target engine
    converted_dfs: List[DataframeLike] = []
    for i, df in enumerate(dfs):
        df_engine = resolve_engine(EngineAbstract.AUTO, df)
        if df_engine != engine_concrete:
            # Type mismatch - convert to target engine
            converted_df = df_to_engine(df, engine_concrete)
            converted_dfs.append(converted_df)
        else:
            converted_dfs.append(df)

    # Use engine-specific concat
    concat_fn = df_concat(engine_concrete)
    result = concat_fn(converted_dfs, ignore_index=ignore_index, sort=sort)

    return result


def safe_merge(
    left: DataframeLike,
    right: DataframeLike,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: Literal['left', 'right', 'outer', 'inner'] = 'inner',
    engine: Union[Engine, EngineAbstract] = EngineAbstract.AUTO
) -> DataframeLike:
    """
    Engine-aware DataFrame merge with automatic type conversion.

    Handles mixed pandas/cuDF DataFrames by converting right DataFrame to match
    left DataFrame's engine before merging. Prevents TypeErrors from direct mixing.

    Args:
        left: Left DataFrame (pandas or cuDF)
        right: Right DataFrame (pandas or cuDF)
        on: Column(s) to join on (must exist in both DataFrames)
        left_on: Column(s) to join on from left DataFrame
        right_on: Column(s) to join on from right DataFrame
        how: Type of merge ('inner', 'outer', 'left', 'right')
        engine: Target engine for result ('auto', 'pandas', or 'cudf')
               If 'auto', uses left DataFrame's type

    Returns:
        Merged DataFrame in target engine type

    Raises:
        ValueError: If both 'on' and 'left_on'/'right_on' are specified

    Examples:
        >>> import pandas as pd
        >>> from graphistry.Engine import EngineAbstract, safe_merge
        >>>
        >>> left = pd.DataFrame({'id': [1, 2], 'val': ['a', 'b']})
        >>> right = pd.DataFrame({'id': [2, 3], 'score': [10, 20]})
        >>> result = safe_merge(left, right, on='id', engine=EngineAbstract.PANDAS)
        >>> len(result)
        1

        >>> # Left join
        >>> result = safe_merge(left, right, on='id', how='left')
        >>> len(result)
        2

        >>> # With cuDF (if available)
        >>> import cudf
        >>> pdf = pd.DataFrame({'id': [1, 2], 'val': ['a', 'b']})
        >>> gdf = cudf.DataFrame({'id': [2, 3], 'score': [10, 20]})
        >>> # safe_merge handles mixed types - converts right to match left
        >>> result = safe_merge(pdf, gdf, on='id')
        >>> isinstance(result, pd.DataFrame)
        True
    """
    # Validate parameters
    if on is not None and (left_on is not None or right_on is not None):
        raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")

    # Resolve target engine
    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    engine_concrete: Engine
    if engine == EngineAbstract.AUTO:
        # Use left DataFrame's engine
        engine_concrete = resolve_engine(EngineAbstract.AUTO, left)
    else:
        engine_concrete = Engine(engine.value)

    # Ensure both DataFrames match target engine
    left_engine = resolve_engine(EngineAbstract.AUTO, left)
    right_engine = resolve_engine(EngineAbstract.AUTO, right)

    if left_engine != engine_concrete:
        # Type mismatch - convert left to target engine
        left = df_to_engine(left, engine_concrete)

    if right_engine != engine_concrete:
        # Type mismatch - convert right to target engine
        right = df_to_engine(right, engine_concrete)

    # Perform merge using DataFrame's native merge method
    # Both pandas and cuDF support the same merge API
    if on is not None:
        result = left.merge(right, on=on, how=how)
    elif left_on is not None and right_on is not None:
        result = left.merge(right, left_on=left_on, right_on=right_on, how=how)
    else:
        raise ValueError("Must specify either 'on' or both 'left_on' and 'right_on'")

    return result
