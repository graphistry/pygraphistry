import pandas as pd
from typing import Any, Mapping, Protocol, TYPE_CHECKING, Tuple, TypeVar, Union

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

    def argsort(self, a: ArrayLike) -> ArrayLike:
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

    def promote_types(self, type1: Any, type2: Any) -> Any:
        ...
