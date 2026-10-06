"""Native property dictionaries and conservative query-key preparation."""
from __future__ import annotations

from bisect import bisect_left
from datetime import datetime
from numbers import Integral
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT, DType, IndexT, SeriesT
from .engine_arrays import as_eager_polars_frame
from .registry import NodePropIndex

if TYPE_CHECKING:
    import polars as pl


def is_string_property(frame: DataFrameT, column: str, engine: Engine) -> bool:
    """Admit homogeneous text, without coercing mixed object or categorical keys."""
    if column not in frame.columns:
        return False
    if engine in POLARS_ENGINES:
        import polars as pl
        eager = as_eager_polars_frame(frame)
        return eager is not None and eager.schema[column] == pl.String
    series = frame[column]
    if engine == Engine.CUDF:
        return series.dtype == np.dtype("object")
    if isinstance(series.dtype, pd.ArrowDtype):
        import pyarrow as pa
        logical_type = series.dtype.pyarrow_dtype
        return pa.types.is_string(logical_type) or pa.types.is_large_string(logical_type)
    return isinstance(series.dtype, pd.StringDtype) or series.dtype.kind == "U" or (
        series.dtype == np.dtype("object")
        and pd.api.types.infer_dtype(series, skipna=True) == "string"
    )


def is_integer_property(frame: DataFrameT, column: str, engine: Engine) -> bool:
    """Classify integer storage before excluding nullable rows."""
    if column not in frame.columns:
        return False
    if engine in POLARS_ENGINES:
        eager = as_eager_polars_frame(frame)
        return eager is not None and eager.schema[column].is_integer()
    return frame[column].dtype.kind in ("i", "u")


def is_categorical_property(frame: DataFrameT, column: str, engine: Engine) -> bool:
    if column not in frame.columns:
        return False
    if engine in POLARS_ENGINES:
        import polars as pl
        eager = as_eager_polars_frame(frame)
        return eager is not None and eager.schema[column] in (pl.Categorical, pl.Enum)
    return isinstance(frame[column].dtype, pd.CategoricalDtype) or (
        engine == Engine.CUDF and frame[column].dtype.name == "category"
    )


def is_timestamp_property(frame: DataFrameT, column: str, engine: Engine) -> bool:
    if column not in frame.columns:
        return False
    if engine in POLARS_ENGINES:
        import polars as pl
        eager = as_eager_polars_frame(frame)
        return eager is not None and isinstance(eager.schema[column], pl.Datetime)
    dtype = frame[column].dtype
    if isinstance(dtype, pd.ArrowDtype):
        import pyarrow as pa
        return pa.types.is_timestamp(dtype.pyarrow_dtype)
    return dtype.kind == "M"


def categorical_property_keys(
    frame: DataFrameT, column: str, engine: Engine,
) -> Tuple[ArrayLike, Optional[SeriesT], Optional[IndexT]]:
    """Reuse physical category codes; never coerce numeric labels to strings."""
    if engine in POLARS_ENGINES:
        import polars as pl
        eager = as_eager_polars_frame(frame)
        assert eager is not None
        decoded = eager.with_columns(pl.col(column).cast(pl.String))
        codes, dictionary = string_property_keys(
            cast(DataFrameT, decoded), column, engine,  # hygiene-ok: explicit-cast -- the eager Polars frame uses the shared engine-polymorphic frame vocabulary
        )
        return codes, dictionary, None
    values = frame[column]
    native_codes = values.cat.codes
    return cast(  # hygiene-ok: explicit-cast -- native NumPy/CuPy category codes implement the bounded array protocol
        ArrayLike, native_codes.values if engine == Engine.CUDF else native_codes.to_numpy(),
    ), None, values.cat.categories


def timestamp_property_keys(frame: DataFrameT, column: str, engine: Engine) -> Tuple[ArrayLike, DType]:
    """Native integer keys; Polars buckets cover canonical comparison casts."""
    if engine in POLARS_ENGINES:
        eager = as_eager_polars_frame(frame)
        assert eager is not None
        import polars as pl
        values = eager.get_column(column)
        timestamp_type = values.dtype
        assert isinstance(timestamp_type, pl.Datetime)
        if timestamp_type.time_unit == "ns":
            # Canonical scalar casts require microsecond candidates; residuals retain exact ns equality.
            values = values.cast(pl.Datetime("us", timestamp_type.time_zone))
        return cast(  # hygiene-ok: explicit-cast -- Polars physical integer storage is a NumPy array compatible with the shared array protocol
            ArrayLike, values.to_physical().to_numpy(),
        ), values.dtype
    values = frame[column]
    if engine == Engine.PANDAS and isinstance(values.dtype, pd.ArrowDtype):
        import pyarrow as pa
        integers = pa.array(values.array).cast(pa.int64()).to_numpy(zero_copy_only=False)
    else:
        integers = values.astype("int64")
        integers = integers.values if engine == Engine.CUDF else integers.to_numpy()
    return cast(  # hygiene-ok: explicit-cast -- build-time native integer extraction implements the shared array protocol
        ArrayLike, integers,
    ), values.dtype


def _timestamp_query_values(
    index: NodePropIndex, members: Sequence[object], predicate: object, xp: ArrayNamespace,
) -> Optional[ArrayLike]:
    dtype = index.timestamp_dtype
    assert dtype is not None
    if not all(isinstance(value, (datetime, np.datetime64, str)) for value in members):
        return None
    if index.engine in POLARS_ENGINES:
        import polars as pl
        native_dtype = cast(  # hygiene-ok: explicit-cast -- the engine and timestamp builder establish a concrete Polars Datetime dtype
            "pl.Datetime", dtype,
        )
        # Canonical temporal membership coercion differs from scalar equality.
        if isinstance(predicate, (IsIn, list, tuple)):
            return None
        value = members[0]
        if isinstance(value, str):
            from graphistry.compute.gfql.lazy.engine.polars.predicates import _parse_temporal_filter_scalar
            value = _parse_temporal_filter_scalar(value, native_dtype)
            if value is None:
                return None
        if isinstance(value, pd.Timestamp) and value.nanosecond:
            return None  # Polars literal precision differs across supported versions.
        try:
            if isinstance(value, datetime) and value.tzinfo is None and native_dtype.time_zone is None:
                encoded = pl.Series([value], dtype=native_dtype).cast(pl.Int64).to_numpy()
            else:
                encoded = pl.select(pl.lit(value).cast(native_dtype).to_physical()).to_series().to_numpy()
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            return None  # No comparable native literal; canonical filter owns errors.
        return xp.asarray(encoded)
    if isinstance(dtype, pd.ArrowDtype):
        unit = dtype.pyarrow_dtype.unit
        timezone = dtype.pyarrow_dtype.tz
    elif isinstance(dtype, pd.DatetimeTZDtype):
        unit, timezone = dtype.unit, dtype.tz
    else:
        unit, timezone = np.datetime_data(dtype)[0], None
    timestamp_ticks = []
    for value in members:  # bounded query literals, never source rows
        try:
            stamp = pd.Timestamp(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if pd.isna(stamp) or (stamp.tz is None) != (timezone is None):
            return None
        timestamp_ticks.append(stamp.asm8.astype(f"datetime64[{unit}]").astype(np.int64))
    return xp.unique(xp.asarray(timestamp_ticks, dtype=xp.int64))


def string_property_keys(
    frame: DataFrameT, column: str, engine: Engine,
) -> Tuple[ArrayLike, SeriesT]:
    """Non-null string rows -> integer codes and their sorted native dictionary."""
    if engine in POLARS_ENGINES:
        eager = as_eager_polars_frame(frame)
        assert eager is not None
        values = eager.get_column(column)
        keys = values.unique().sort()
        return _values_to_codes_polars(keys, values), cast(  # hygiene-ok: explicit-cast -- SeriesT is the engine-polymorphic stored column vocabulary
            SeriesT, keys,
        )
    values = frame[column]
    keys = values.drop_duplicates().sort_values().reset_index(drop=True)
    if engine == Engine.PANDAS:
        keys = pd.Series(keys.to_numpy(dtype=object), dtype=object)
    return keys.searchsorted(values), keys


def _values_to_codes_polars(keys: "pl.Series", values: "pl.Series") -> ArrayLike:
    """Polars native string search returns host integer positions."""
    return cast(  # hygiene-ok: explicit-cast -- Polars emits a NumPy integer array; ArrayLike protocol has stricter operator annotations
        ArrayLike, keys.search_sorted(values).to_numpy(),
    )


def _string_query_codes(index: NodePropIndex, members: Sequence[str], xp: ArrayNamespace) -> ArrayLike:
    keys = index.string_keys
    assert keys is not None
    size = len(keys)
    if size == 0 or not members:
        return xp.zeros(0, dtype=xp.int64)
    if index.engine in POLARS_ENGINES:
        import polars as pl
        native_keys = cast(  # hygiene-ok: explicit-cast -- index.engine establishes the concrete type of the native stored dictionary
            "pl.Series", keys,
        )
        if len(members) == 1:
            value = members[0]
            # O(log dictionary) public native scalar reads; no source rows/export
            # or eager query plan for a single bounded text literal.
            position = bisect_left(native_keys, value)
            if position < size and native_keys.item(position) == value:
                return xp.asarray([position], dtype=index.keys_sorted.dtype)
            return xp.zeros(0, dtype=index.keys_sorted.dtype)
        values = pl.Series(members, dtype=pl.String)
        positions = xp.asarray(_values_to_codes_polars(native_keys, values))
        clipped = xp.minimum(positions, size - 1)
        hits = xp.asarray((native_keys.gather(np.asarray(clipped)) == values).to_numpy())
    else:
        if index.engine == Engine.CUDF:
            import cudf
            values = cudf.Series(members, dtype="str")
        else:
            values = pd.Series(members, dtype=keys.dtype)
        positions = xp.asarray(keys.searchsorted(values))
        clipped = xp.minimum(positions, size - 1)
        equal = keys.iloc[clipped].reset_index(drop=True) == values
        hits = equal.values if index.engine == Engine.CUDF else equal.to_numpy(dtype=bool)
    return xp.unique(positions[(positions < size) & hits])


def property_query_values(index: NodePropIndex, predicate: object, xp: ArrayNamespace) -> Optional[ArrayLike]:
    """Encode supported equality/membership values; decline ambiguous coercions."""
    members = predicate.options if isinstance(predicate, IsIn) else (
        predicate if isinstance(predicate, (list, tuple)) else [predicate]
    )
    if index.timestamp_dtype is not None:
        return _timestamp_query_values(index, members, predicate, xp)
    if index.category_keys is not None:
        if not all(isinstance(value, (str, Integral, float, bool)) for value in members):
            return None
        if not members:
            return xp.zeros(0, dtype=xp.int64)
        if any(isinstance(value, float) and np.isnan(value) for value in members):
            return None  # AST IsIn can match null rows; the index excludes them.
        kinds = {
            "str" if isinstance(value, str) else "bool" if isinstance(value, bool)
            else "int" if isinstance(value, Integral) else "float"
            for value in members
        }
        if len(kinds) != 1:
            return None  # Native Index inference can coerce mixed keys or reject them.
        if index.engine == Engine.PANDAS and type(index.category_keys) is pd.Index and kinds == {"str"} and len(members) == 1:
            # Unique plain category labels have exact scalar lookup semantics.
            # Specialized Index types can parse string keys or return partial slices.
            try:
                position = index.category_keys.get_loc(members[0])
            except KeyError:
                return xp.zeros(0, dtype=index.keys_sorted.dtype)
            return xp.asarray([position], dtype=index.keys_sorted.dtype)
        if index.engine == Engine.CUDF:
            if kinds == {"int"} and any(int(v) < 0 for v in members if isinstance(v, Integral)) and any(
                int(v) > np.iinfo(np.int64).max for v in members if isinstance(v, Integral)
            ):
                return None

            import cudf
            values = cudf.Index(members)
        else:
            values = pd.Index(members, dtype=object)
        codes = xp.asarray(index.category_keys.get_indexer(values))
        return xp.unique(codes[codes > -1])
    if index.string_keys is not None:
        if not all(isinstance(value, str) for value in members):
            return None
        return _string_query_codes(index, [value for value in members if isinstance(value, str)], xp)
    if not all(isinstance(value, Integral) and not isinstance(value, bool) for value in members):
        return None
    bounds = np.iinfo(index.keys_sorted.dtype)
    values = xp.asarray(
        [int(value) for value in members if isinstance(value, Integral) and bounds.min <= int(value) <= bounds.max],
        dtype=index.keys_sorted.dtype,
    )
    return values if int(values.shape[0]) <= 1 else xp.unique(values)
