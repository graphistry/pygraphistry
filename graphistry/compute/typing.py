import pandas as pd
from typing import Any, Mapping, Optional, Protocol, TYPE_CHECKING, Tuple, TypeVar, Union

# TODO stubs for Union[cudf.DataFrame, dask.DataFrame, ..] at checking time
if TYPE_CHECKING:
    DataFrameT = pd.DataFrame
    SeriesT = pd.Series
    IndexT = pd.Index
    DomainT = Union[pd.Index, pd.Series]
else:
    DataFrameT = Any
    SeriesT = Any
    IndexT = Any
    DomainT = Any

# --- polars vocabulary -------------------------------------------------------------
#
# ``DataFrameT`` is *deliberately* pinned to ``pd.DataFrame`` at checking time (widening it to
# a cross-engine union fans out across every one of the ~325 checked modules). So a helper that
# is UNCONDITIONALLY polars -- not an engine dispatcher, a polars-only kernel -- must not be
# annotated ``DataFrameT``: it should say polars, and these are the names for that.
#
# TYPE_CHECKING-only (no runtime symbol) because polars is an OPTIONAL dependency: importing it
# at module scope would make every importer of this module require polars. This is the shape
# already proven in ``compute/gfql/lazy/engine/polars/dtypes.py``, which now re-exports from
# here so there is one definition. Consumers import these under their own ``if TYPE_CHECKING:``
# and write the annotation as a string (or rely on ``from __future__ import annotations``).
if TYPE_CHECKING:
    import polars as pl

    #: Either polars frame flavour. Use for a parameter that accepts eager *or* lazy.
    PolarsFrame = Union["pl.DataFrame", "pl.LazyFrame"]

    #: Eager-in -> eager-out / lazy-in -> lazy-out. CONSTRAINED (not bound) on purpose: a
    #: ``PolarsFrame`` return would lose the flavour and type-error at every call site that
    #: goes on to use an eager-only method.
    PolarsT = TypeVar("PolarsT", "pl.DataFrame", "pl.LazyFrame")

    #: A polars Series -- the polars counterpart of ``SeriesT``.
    PolarsSeriesT = pl.Series

# Engine-polymorphic column dtype: numpy dtype / pandas ExtensionDtype / polars DataType.
# Honestly Any -- the concrete type is engine-dependent and only ever passed to dtype-inspection
# helpers that accept Any and fail closed.
DType = Any
NodeDtypes = Mapping[str, DType]

# Type variable for return type preservation in predicates
T = TypeVar('T')

class ArrayLike(Protocol):
    """Small numpy/cupy-like 1-D array surface used by compute kernels."""

    shape: Tuple[int, ...]
    dtype: Any
    nbytes: int

    def __getitem__(self, key: Any) -> "ArrayLike":
        ...

    def __setitem__(self, key: Any, value: Any) -> None:
        ...

    def __ne__(self, other: Any) -> "ArrayLike":  # type: ignore[override]
        ...

    def __eq__(self, other: Any) -> "ArrayLike":  # type: ignore[override]
        ...

    def __lt__(self, other: Any) -> "ArrayLike":
        ...

    def __gt__(self, other: Any) -> "ArrayLike":
        ...

    def __invert__(self) -> "ArrayLike":
        ...

    def __and__(self, other: Any) -> "ArrayLike":
        ...

    def __rand__(self, other: Any) -> "ArrayLike":
        ...

    def __add__(self, other: Any) -> "ArrayLike":
        ...

    def __radd__(self, other: Any) -> "ArrayLike":
        ...

    def __sub__(self, other: Any) -> "ArrayLike":
        ...

    def __rsub__(self, other: Any) -> "ArrayLike":
        ...

    def astype(self, dtype: Any) -> "ArrayLike":
        ...

    def sum(self) -> Any:
        ...


class ArrayNamespace(Protocol):
    """Small numpy/cupy namespace surface used by compute kernels."""

    int64: Any

    def zeros(self, shape: Any, dtype: Any = ...) -> ArrayLike:
        ...

    def ones(self, shape: Any, dtype: Any = ...) -> ArrayLike:
        ...

    def empty(self, shape: Any, dtype: Any = ...) -> ArrayLike:  # hygiene-ok: explicit-any -- numpy/cupy shape+dtype args, same shape as zeros/ones above
        ...

    def subtract(self, a: Any, b: Any, out: "Optional[ArrayLike]" = None) -> ArrayLike:  # hygiene-ok: explicit-any -- ufunc accepts array|scalar operands (numpy/cupy)
        ...

    def argsort(self, a: ArrayLike) -> ArrayLike:
        ...

    def bincount(self, a: ArrayLike, weights: "Optional[ArrayLike]" = None, minlength: int = 0) -> ArrayLike:
        ...

    def nonzero(self, a: ArrayLike) -> Tuple[ArrayLike, ...]:
        ...

    def concatenate(self, arrays: Any) -> ArrayLike:
        ...

    def asarray(self, a: Any, dtype: Any = ...) -> ArrayLike:
        ...

    def cumsum(self, a: ArrayLike) -> ArrayLike:
        ...

    def arange(self, *args: Any, **kwargs: Any) -> ArrayLike:
        ...

    def searchsorted(self, a: ArrayLike, v: ArrayLike) -> ArrayLike:
        ...

    def where(self, condition: ArrayLike, x: Any, y: Any) -> ArrayLike:
        ...

    def sort(self, a: ArrayLike) -> ArrayLike:
        ...

    def unique(self, a: ArrayLike) -> ArrayLike:
        ...

    def isnan(self, a: ArrayLike) -> ArrayLike:
        ...

    def promote_types(self, type1: Any, type2: Any) -> Any:
        ...
