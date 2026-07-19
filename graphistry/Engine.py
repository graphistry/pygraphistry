from inspect import getmodule
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
from typing import Any, List, Optional, Union
from typing_extensions import Literal
from enum import Enum

from graphistry.models.types import ValidationParam


class Engine(Enum):
    PANDAS = 'pandas'
    CUDF = 'cudf'
    DASK = 'dask'
    DASK_CUDF = 'dask_cudf'
    POLARS = 'polars'
    # GPU execution TARGET of the lazy Polars engine (cudf_polars): frames stay
    # ``pl.DataFrame`` (handled exactly like POLARS in all frame ops); only the
    # lazy ``.collect()`` runs on GPU. Explicit opt-in only — AUTO never selects it.
    POLARS_GPU = 'polars-gpu'

# Engines whose frames use the polars API (unique/with_columns/...) rather than the
# pandas API (drop_duplicates/assign/...). POLARS_GPU is the GPU execution target of
# the same lazy Polars engine — frames stay ``pl.DataFrame``, so it shares the path.
POLARS_ENGINES = (Engine.POLARS, Engine.POLARS_GPU)

class EngineAbstract(Enum):
    PANDAS = Engine.PANDAS.value
    CUDF = Engine.CUDF.value
    DASK = Engine.DASK.value
    DASK_CUDF = Engine.DASK_CUDF.value
    POLARS = Engine.POLARS.value
    POLARS_GPU = Engine.POLARS_GPU.value
    AUTO = 'auto'


# Type alias for engine parameter - accepts both enum values and string literals
# Includes 'auto' for automatic detection
EngineAbstractType = Union[EngineAbstract, Literal['pandas', 'cudf', 'dask', 'dask_cudf', 'polars', 'polars-gpu', 'auto']]

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

        # Arrow and Spark are input formats, not compute engines — coerce to pandas at call sites
        if isinstance(g_or_df, pa.Table):
            return Engine.PANDAS

        try:
            from pyspark.sql import DataFrame as SparkDataFrame
            if isinstance(g_or_df, SparkDataFrame):
                return Engine.PANDAS
        except ImportError:
            pass

        if 'polars' in str(type(g_or_df).__module__):
            try:
                import polars as pl
                if isinstance(g_or_df, (pl.DataFrame, pl.LazyFrame)):
                    return Engine.PANDAS
            except ImportError:
                pass

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

def _cudf_from_pandas_best_effort(df: pd.DataFrame, *, validate: Optional[ValidationParam] = None, warn: bool = True):
    """pandas -> cuDF honoring the repo-wide ``validate``/``warn`` convention.

    Default (``validate=None`` -> ``autofix``): best-effort per-column coercion of
    mixed-type object columns to string (numeric-looking columns kept numeric), warning
    once. ``strict``/``strict-fast``: raise ``ArrowConversionError`` instead of coercing
    (matching the plot()/upload() boundary and the polars converter's strict mode).
    ``validate=False`` == autofix but suppresses the warning.
    """
    import cudf
    from graphistry.validate.common import normalize_validation_params
    validate_mode, warn = normalize_validation_params('autofix' if validate is None else validate, warn)

    try:
        return cudf.from_pandas(df)
    except Exception as e:
        if validate_mode in ('strict', 'strict-fast'):
            from graphistry.exceptions import ArrowConversionError
            raise ArrowConversionError(columns=_mixed_type_object_columns(df), original_error=e) from e
        failed_cols: List[str] = []
        out_gdf = cudf.from_pandas(df[[]])
        for col in df.columns:
            try:
                out_gdf[col] = cudf.from_pandas(df[[col]])[col]
            except Exception:
                series = df[col]
                non_null = series.dropna()
                numeric_values = list(non_null.tolist()) if len(non_null) > 0 else []
                if numeric_values and all(
                    isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
                    for value in numeric_values
                ):
                    try:
                        numeric_series = pd.to_numeric(series, errors="coerce")
                        if all(float(value).is_integer() for value in numeric_values):
                            numeric_series = numeric_series.astype("Int64")
                        numeric_df = pd.DataFrame({col: numeric_series})
                        out_gdf[col] = cudf.from_pandas(numeric_df)[col]
                        continue
                    except Exception:
                        pass
                failed_cols.append(str(col))
                string_df = pd.DataFrame({col: series.astype("string")})
                out_gdf[col] = cudf.from_pandas(string_df)[col]
        if failed_cols and warn:
            warnings.warn(
                "Best-effort pandas->cuDF coercion converted mixed-type columns to string dtype: "
                + ", ".join(f"{col}[{df[col].dtype}]" for col in failed_cols),
                RuntimeWarning,
            )
        return out_gdf


def is_polars_df(df: Any) -> bool:
    """True if ``df`` is a polars DataFrame or LazyFrame.

    Import-light module-name check (polars is an optional dependency, so we avoid importing
    it just to ``isinstance``). ``type(df).__module__`` starts with ``polars.`` for both
    ``pl.DataFrame`` and ``pl.LazyFrame``. Single source of truth — the gfql engine had this
    reimplemented in 5 places."""
    return df is not None and "polars" in type(df).__module__


def active_frames_are_polars(g: Any) -> bool:
    """True if ``g``'s active table (nodes, else edges) is a polars frame."""
    if g._nodes is not None:
        return is_polars_df(g._nodes)
    return is_polars_df(g._edges)


def df_to_engine(df, engine: Engine, *, validate: Optional[ValidationParam] = None, warn: bool = True):
    """Convert ``df`` to ``engine``'s frame type.

    ``validate``/``warn`` (see ``graphistry.models.types.ValidationParam``) control how
    mixed-type object columns that cannot go to Arrow are handled. ``validate=None``
    (default) means "use the engine's own default": cuDF defaults ``autofix`` (best-effort
    coerce mixed cols to string + warn, its shipped behavior), polars defaults ``strict``
    (parity-or-raise). ``strict``/``strict-fast`` raise ``ArrowConversionError`` (cuDF) or
    ``NotImplementedError`` (polars); ``autofix`` coerces + warns on both.
    """
    if engine == Engine.PANDAS:
        if isinstance(df, pd.DataFrame):
            return df
        if isinstance(df, pa.Table):
            return df.to_pandas()
        type_module = str(type(df).__module__)
        if 'pyspark' in type_module:
            from pyspark.sql import DataFrame as SparkDF
            if isinstance(df, SparkDF):
                return df.toPandas()
        # dask_cudf must be checked before dask: 'dask' appears in 'dask_cudf.core' so
        # reversing the order would incorrectly route dask_cudf frames into the dask branch.
        if 'dask_cudf' in type_module:
            import dask_cudf
            if isinstance(df, dask_cudf.DataFrame):
                return df.compute().to_pandas()
        if 'dask' in type_module:
            import dask.dataframe as dd
            if isinstance(df, dd.DataFrame):
                return df.compute()
        if 'cudf' in type_module:
            import cudf
            if isinstance(df, cudf.DataFrame):
                return df.to_pandas()
        if 'polars' in type_module:
            import polars as pl
            if isinstance(df, pl.LazyFrame):
                return df.collect().to_pandas()
            if isinstance(df, pl.DataFrame):
                return df.to_pandas()
        raise ValueError(f'Cannot convert type {type(df)} to pandas')
    elif engine == Engine.CUDF:
        import cudf
        if isinstance(df, cudf.DataFrame):
            return df
        if not isinstance(df, pd.DataFrame):
            df = df_to_engine(df, Engine.PANDAS)
        return _cudf_from_pandas_best_effort(df, validate=validate, warn=warn)
    elif engine == Engine.DASK:
        import dask.dataframe as dd
        if isinstance(df, dd.DataFrame):
            return df
        if not isinstance(df, pd.DataFrame):
            df = df_to_engine(df, Engine.PANDAS)
        return dd.from_pandas(df, npartitions=1)
    elif engine in POLARS_ENGINES:
        import polars as pl
        # polars-engine-specific NaN->null coercion lives with the polars engine; local
        # import (lazy/... never imports Engine at module load -> no cycle)
        from graphistry.compute.gfql.lazy.engine.polars.nan_clean import _pl_nan_to_null
        if isinstance(df, pl.DataFrame):
            return _pl_nan_to_null(df)
        if isinstance(df, pl.LazyFrame):
            # Collect via the target-aware lazy collect so the executor knobs apply:
            # engine=POLARS_GPU collects on the cudf-polars GPU executor (gpu_executor()),
            # engine=POLARS honors cpu_streaming(). A bare df.collect() would materialize on
            # the CPU default executor and ignore both — the POLARS_ENGINES branch does not
            # otherwise distinguish POLARS from POLARS_GPU for a LazyFrame input. Function-local
            # import: lazy/__init__ imports no heavy deps and never imports Engine (no cycle).
            from graphistry.compute.gfql.lazy import collect as _lazy_collect, target_mode, ExecutionTarget
            _tgt = ExecutionTarget.GPU if engine == Engine.POLARS_GPU else ExecutionTarget.CPU
            with target_mode(_tgt):
                return _pl_nan_to_null(_lazy_collect(df))
        if isinstance(df, pa.Table):
            return _pl_nan_to_null(pl.from_arrow(df))
        pl_validate: ValidationParam = 'strict' if validate is None else validate
        if isinstance(df, pd.DataFrame):
            return _pl_from_pandas(df, validate=pl_validate, warn=warn)
        # cuDF (device) -> Arrow -> polars: a single host copy via cuDF's native
        # interchange, not the cuDF -> pandas -> polars double-convert. Besides the
        # extra hop, the pandas detour is lossy (cuDF nullable Int64/boolean ->
        # pandas float+NaN/object), whereas Arrow preserves dtypes and nulls.
        # (Skipping the host round trip entirely for polars-gpu is a deeper
        # follow-up: cudf_polars still ingests a host polars frame today.)
        if 'cudf' in str(type(df).__module__):
            import cudf
            if isinstance(df, cudf.DataFrame):
                return _pl_nan_to_null(pl.from_arrow(df.to_arrow()))
        # dask/spark and anything else: route through pandas
        return _pl_from_pandas(df_to_engine(df, Engine.PANDAS), validate=pl_validate, warn=warn)
    raise ValueError(f'Only engines pandas/cudf/dask/polars supported, got: {engine}')


def _mixed_type_object_columns(df) -> List[str]:
    """Object columns holding >1 Python scalar type among non-null values.

    Cypher properties are dynamically typed, so a pandas object column can hold
    e.g. ``int`` and ``str`` together; polars/Arrow cannot represent that in one
    column. Used to name the offender in the honest ``NotImplementedError``."""
    bad: List[str] = []
    for col in df.columns:
        if str(getattr(df[col], "dtype", "")) != "object":
            continue
        types = set()
        for value in df[col].to_numpy():
            if value is None or (isinstance(value, float) and value != value):  # None / NaN
                continue
            types.add(type(value))
            if len(types) > 1:
                bad.append(str(col))
                break
    return bad


def _pl_from_pandas(df, *, validate: ValidationParam = 'strict', warn: bool = True):
    """``pl.from_pandas`` participating in the repo-wide ``validate``/``warn`` convention.

    Polars/Arrow cannot represent a mixed-type (e.g. int+str) object column, which
    pandas allows for dynamically-typed Cypher properties. This mirrors the plot()/
    upload() ``validate`` knob (``graphistry.models.types.ValidationParam``), but —
    unlike the display/upload boundary, which defaults ``autofix`` — the compute path
    defaults ``strict``: silently string-coercing a mixed column diverges from the
    pandas oracle (pandas keeps it ``object`` + Python-compares), a wrong-answer risk.
    So parity-or-raise is the safe default; ``autofix`` is an explicit opt-in.

    - ``strict``/``strict-fast`` (default): raise ``NotImplementedError`` (use
      ``engine='pandas'``, or ``validate='autofix'`` to coerce), instead of surfacing a
      cryptic ``pyarrow.lib.ArrowInvalid`` from deep inside construction.
    - ``autofix`` (or ``validate=False``): coerce the offending mixed-type object
      column(s) to string and, if ``warn``, emit a ``RuntimeWarning`` — matching
      ``_cudf_from_pandas_best_effort``.
    """
    import polars as pl
    from graphistry.validate.common import normalize_validation_params
    validate_mode, warn = normalize_validation_params(validate, warn)
    try:
        return pl.from_pandas(df)
    except Exception as e:
        bad = _mixed_type_object_columns(df)
        if validate_mode == 'autofix' and bad:
            fixed = df.astype({col: 'string' for col in bad})
            try:
                out = pl.from_pandas(fixed)
            except Exception:
                out = None
            if out is not None:
                if warn:
                    warnings.warn(
                        "engine='polars': coerced mixed-type column(s) to string for "
                        f"Arrow conversion: {bad}. Convert explicitly or use "
                        "engine='pandas' for control.",
                        RuntimeWarning,
                    )
                return out
        hint = f" (mixed-type column(s): {bad})" if bad else ""
        raise NotImplementedError(
            "engine='polars' cannot represent a heterogeneous/mixed-type column"
            f"{hint}; use engine='pandas', or validate='autofix' to coerce to string"
        ) from e

def _pl_concat(frames, *, ignore_index: bool = True, sort: bool = False, **_kw):
    """pandas/cudf-signature-compatible row concat for polars.

    polars has no row index and no ``sort`` kwarg; ``ignore_index``/``sort`` are
    accepted-and-ignored so generic call sites (hop/chain) work unchanged.
    ``vertical_relaxed`` aligns to a common supertype like pandas does for
    mixed-but-compatible column dtypes.
    """
    import polars as pl
    frames = list(frames)
    if len(frames) == 0:
        return pl.DataFrame()
    if len(frames) == 1:
        return frames[0]
    if isinstance(frames[0], pl.Series):
        # Series concat only supports the default vertical strategy.
        return pl.concat(frames)
    return pl.concat(frames, how="vertical_relaxed")


def df_concat(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.concat
    elif engine == Engine.CUDF:
        import cudf
        return cudf.concat
    elif engine in POLARS_ENGINES:
        return _pl_concat
    elif engine == Engine.DASK:
        raise NotImplementedError("DASK is an input format, not a compute engine — use engine='auto' or engine='pandas'")
    raise ValueError(f'Only engines pandas/cudf/polars supported, got: {engine}')


def align_shared_column_dtypes(
    reference: DataframeLike,
    candidate: DataframeLike,
) -> DataframeLike:
    """
    Coerce shared columns on ``candidate`` to the dtypes already used by
    ``reference`` when both frames are on the same engine.

    cuDF row-wise concat is stricter than pandas about shared-column dtype
    mismatches, so endpoint rows synthesized from edges need to inherit the
    node table schema before concatenation.
    """

    if reference is None or candidate is None:
        return candidate

    reference_engine = resolve_engine(EngineAbstract.AUTO, reference)
    candidate_engine = resolve_engine(EngineAbstract.AUTO, candidate)
    if candidate_engine != reference_engine:
        candidate = df_to_engine(candidate, reference_engine)

    shared_cols = [col for col in candidate.columns if col in reference.columns]
    for col in shared_cols:
        ref_dtype = getattr(reference[col], "dtype", None)
        cand_dtype = getattr(candidate[col], "dtype", None)
        if ref_dtype is None or cand_dtype is None or str(ref_dtype) == str(cand_dtype):
            continue
        try:
            candidate[col] = candidate[col].astype(ref_dtype)
        except Exception:
            pass

    return candidate


def safe_row_concat(
    frames: List[DataframeLike],
    *,
    ignore_index: bool = True,
    sort: bool = False,
) -> DataframeLike:
    """
    Concatenate row-wise while preserving the first frame's engine when cuDF
    refuses mixed/missing-column schemas that pandas accepts.
    """

    if len(frames) == 0:
        return pd.DataFrame()
    if len(frames) == 1:
        return frames[0]

    engine_concrete = resolve_engine(EngineAbstract.AUTO, frames[0])
    concat_fn = df_concat(engine_concrete)
    try:
        return concat_fn(frames, ignore_index=ignore_index, sort=sort)
    except Exception:
        if engine_concrete != Engine.CUDF:
            raise
        pandas_frames = [df_to_engine(frame, Engine.PANDAS) for frame in frames]
        pandas_result = pd.concat(pandas_frames, ignore_index=ignore_index, sort=sort)
        return df_to_engine(pandas_result, Engine.CUDF)

def df_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.DataFrame
    elif engine == Engine.CUDF:
        import cudf
        return cudf.DataFrame
    elif engine in POLARS_ENGINES:
        import polars as pl
        return pl.DataFrame
    raise ValueError(f'Only engines pandas/cudf/polars supported, got: {engine}')


def df_unique(df, engine: Engine):
    """Row-dedupe keeping first occurrence, engine-aware.

    pandas/cuDF use ``drop_duplicates``; polars uses ``unique(maintain_order=True)``
    (``maintain_order`` matches ``drop_duplicates(keep='first')`` — same convention as
    ``compute/gfql/row/frame_ops.distinct``). Avoids calling the pandas-only
    ``drop_duplicates`` on a polars frame (the UNION DISTINCT crash)."""
    if engine in POLARS_ENGINES:
        return df.unique(maintain_order=True)
    return df.drop_duplicates(ignore_index=True)

def s_cons(engine: Engine):
    if engine == Engine.PANDAS:
        return pd.Series
    elif engine == Engine.CUDF:
        import cudf
        return cudf.Series
    elif engine in POLARS_ENGINES:
        import polars as pl
        return pl.Series
    raise ValueError(f'Only engines pandas/cudf/polars supported, got: {engine}')

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


def s_to_numeric(engine: Engine):
    """Return engine-appropriate to_numeric function."""
    if engine == Engine.PANDAS:
        return pd.to_numeric
    elif engine == Engine.CUDF:
        import cudf
        return cudf.to_numeric
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}')


def s_na(engine: Engine):
    """Return engine-appropriate NA/null value for DataFrame assignment."""
    if engine == Engine.PANDAS:
        return pd.NA
    elif engine == Engine.CUDF:
        # cuDF doesn't have pd.NA; None works for both but explicit is clearer
        return None
    raise ValueError(f'Only engines pandas/cudf supported, got: {engine}. DASK is an input format — coerce to pandas/cudf first.')


def safe_map_series(series: DataframeLike, mapping: Union[dict, pd.Series, DataframeLike]) -> DataframeLike:
    """Map a Series through a dict-like mapping, safe for cudf.

    cudf Series.map(dict/Series) and Series.to_pandas() both trigger numba JIT
    (via numba_cuda.as_cuda_array) which SIGSEGVs on RAPIDS 25.02.
    For cudf, use a merge-based lookup that stays on GPU; to_arrow() transfers
    only the mapping (small) without the numba path.
    For pandas, use native .map().
    """
    try:
        import cudf
    except Exception:
        cudf = None
    if cudf is not None and isinstance(series, cudf.Series):
        if isinstance(mapping, dict):
            lookup = cudf.DataFrame({"__key__": list(mapping.keys()), "__val__": list(mapping.values())})
        elif isinstance(mapping, pd.Series):
            lookup = cudf.DataFrame({"__key__": mapping.index.tolist(), "__val__": mapping.values.tolist()})
        elif isinstance(mapping, cudf.Series):
            # to_arrow() avoids the numba SIGSEGV path (to_pandas() → numba_cuda.as_cuda_array)
            lookup = cudf.DataFrame({"__key__": mapping.index.to_arrow().to_pylist(), "__val__": mapping.to_arrow().to_pylist()})
        else:
            mapping_pd = mapping.to_pandas() if hasattr(mapping, "to_pandas") else mapping
            result_pd = series.to_pandas().map(mapping_pd)
            return cudf.Series(result_pd, index=series.index)
        lookup = lookup.drop_duplicates(subset=["__key__"], keep="last")
        # Left merge preserves left row order in cudf; no sort needed.
        left = cudf.DataFrame({"__key__": series})
        result = left.merge(lookup, on="__key__", how="left")["__val__"].reset_index(drop=True)
        result.index = series.index
        return result
    return series.map(mapping)


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
