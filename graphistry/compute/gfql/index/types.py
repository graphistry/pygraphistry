"""Shared internal types for GFQL physical indexes."""
from __future__ import annotations

from typing import Any, List, Literal, Optional, Protocol, Tuple, TypedDict, Union

from graphistry.compute.typing import DataFrameT

IndexKind = Literal["edge_out_adj", "edge_in_adj", "node_id"]
AdjacencyIndexKind = Literal["edge_out_adj", "edge_in_adj"]
IndexBackend = Literal["numpy", "cupy"]
HopDirection = Literal["forward", "reverse", "undirected"]
EdgeIndexDirection = Literal["forward", "reverse", "both"]

FrameLike = DataFrameT


class ArrayLike(Protocol):
    """Small numpy/cupy-like 1-D array surface used by CSR indexes."""

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
    """Tiny numpy/cupy namespace surface used by CSR build/lookup."""

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


IndexPath = Literal["scan", "index"]


class IndexTraceStep(TypedDict, total=False):
    op: str
    direction: HopDirection
    hops: Optional[int]
    policy: str
    engine: str
    frontier_n: int
    path: IndexPath
    decision_reason: str
    n_keys: int
    seed_deg_sum: Optional[int]
    est_result_rows: Optional[int]
    threshold_frac: float


IndexTrace = List[IndexTraceStep]
