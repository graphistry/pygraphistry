"""DataFrame type coercion primitives for safe pandas/cuDF operations.

This module provides engine-aware operations that handle mixed pandas/cuDF
DataFrame types automatically, preventing TypeErrors from direct operations.

Centralizes type coercion logic that was previously scattered across GFQL layer.
See issue #784: https://github.com/graphistry/pygraphistry/issues/784
"""

from typing import List, Optional, Union
from typing_extensions import Literal

from graphistry.Engine import Engine, EngineAbstract, df_concat, df_to_engine, resolve_engine
from graphistry.compute.typing import DataFrameT
from graphistry.util import setup_logger

logger = setup_logger(__name__)


def safe_concat(
    dfs: List[DataFrameT],
    engine: Union[Engine, EngineAbstract] = EngineAbstract.AUTO,
    ignore_index: bool = False,
    sort: bool = False
) -> DataFrameT:
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
        >>> from graphistry.Engine import EngineAbstract
        >>> from graphistry.compute.primitives import safe_concat
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
            engine = Engine(engine.value) if engine != EngineAbstract.AUTO else Engine.PANDAS
        if engine == Engine.PANDAS:
            import pandas as pd
            return pd.DataFrame()  # type: ignore
        elif engine == Engine.CUDF:
            import cudf
            return cudf.DataFrame()  # type: ignore
        else:
            raise ValueError(f"Unknown engine: {engine}")

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
    converted_dfs: List[DataFrameT] = []
    for i, df in enumerate(dfs):
        df_engine = resolve_engine(EngineAbstract.AUTO, df)
        if df_engine != engine_concrete:
            logger.debug(
                'Type mismatch in concat at index %d: expected %s but got %s. Converting.',
                i, engine_concrete.value, df_engine.value
            )
            converted_df = df_to_engine(df, engine_concrete)
            converted_dfs.append(converted_df)
        else:
            converted_dfs.append(df)

    # Use engine-specific concat
    concat_fn = df_concat(engine_concrete)
    result = concat_fn(converted_dfs, ignore_index=ignore_index, sort=sort)

    return result


def safe_merge(
    left: DataFrameT,
    right: DataFrameT,
    on: Optional[Union[str, List[str]]] = None,
    left_on: Optional[Union[str, List[str]]] = None,
    right_on: Optional[Union[str, List[str]]] = None,
    how: Literal['left', 'right', 'outer', 'inner'] = 'inner',
    engine: Union[Engine, EngineAbstract] = EngineAbstract.AUTO
) -> DataFrameT:
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
        >>> from graphistry.Engine import EngineAbstract
        >>> from graphistry.compute.primitives import safe_merge
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
        logger.debug(
            'Type mismatch in merge (left): expected %s but got %s. Converting.',
            engine_concrete.value, left_engine.value
        )
        left = df_to_engine(left, engine_concrete)

    if right_engine != engine_concrete:
        logger.debug(
            'Type mismatch in merge (right): expected %s but got %s. Converting.',
            engine_concrete.value, right_engine.value
        )
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
