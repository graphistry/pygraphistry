"""Native string dictionaries and lossless property query-key preparation."""
from __future__ import annotations

from bisect import bisect_left
from numbers import Integral
from sys import getsizeof
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd

from graphistry.Engine import Engine, POLARS_ENGINES
from graphistry.compute.predicates.is_in import IsIn
from graphistry.compute.typing import ArrayLike, ArrayNamespace, DataFrameT, SeriesT
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


def bounded_string_key_positions(
    keys: Optional[SeriesT], engine: Engine,
) -> Tuple[Optional[Mapping[str, int]], int]:
    """Build bounded CPU dictionary metadata once, without exporting source rows."""
    if keys is None or engine != Engine.POLARS or len(keys) > 1024:
        return None, 0
    native_keys = cast("pl.Series", keys)  # hygiene-ok: explicit-cast -- CPU Polars builders supply a native text dictionary
    if native_keys.estimated_size() > 64 * 1024:
        return None, 0
    positions = {key: position for position, key in enumerate(native_keys.to_list())}
    frozen = MappingProxyType(positions)
    nbytes = getsizeof(frozen) + getsizeof(positions) + sum(getsizeof(key) + getsizeof(value) for key, value in positions.items())
    return (frozen, nbytes) if nbytes <= 256 * 1024 else (None, 0)


def _string_query_codes(index: NodePropIndex, members: Sequence[str], xp: ArrayNamespace) -> ArrayLike:
    keys = index.string_keys
    assert keys is not None
    size = len(keys)
    if size == 0 or not members:
        return xp.zeros(0, dtype=xp.int64)
    if len(members) == 1 and type(members[0]) is str and index.string_key_positions is not None:
        position = index.string_key_positions.get(members[0])
        return xp.zeros(0, dtype=index.keys_sorted.dtype) if position is None else xp.asarray([position], dtype=index.keys_sorted.dtype)
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
