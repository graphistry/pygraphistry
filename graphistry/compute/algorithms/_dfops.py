"""Engine-agnostic dataframe primitives for the Graphalytics kernels.

Every function here works identically on pandas and cudf. The engine is inferred
from the frame you pass in -- callers never name an engine.

Portability rules these primitives exist to enforce (each one is a bug we would
otherwise hit at 1B edges):

- `Series.__lshift__` and `&` raise TypeError on pandas. Bit-packing uses `*` and
  `%`, never `<<`/`&`.
- `cupy.minimum` on a cudf Series returns a cupy array and drops the index, so
  elementwise min is `a.where(a <= b, b)`, not `np.minimum`/`s_minimum`.
- pandas `groupby` defaults to `sort=True`, cudf to `sort=False` returning
  arbitrary order. Always pass `sort=False` explicitly and sort after.
- uint64 multiply/xor wraps silently and identically in numpy and libcudf; int64
  overflow warns and Python-int masks upcast. Hashing is uint64 throughout.

Banned in kernels: `.apply`, `groupby().apply`, `idxmin`/`idxmax`, `mode()`,
`pd.factorize`, `np.*` on Series, `.query`, `Categorical`.
"""
from __future__ import annotations

from typing import Any, Iterator, Sequence

import pandas as pd

# 2**32, as a plain Python int. Used for bit-packing via arithmetic.
SHIFT32 = 1 << 32


def is_cudf(obj: Any) -> bool:
    """True when obj belongs to cudf, without importing cudf on CPU-only hosts."""
    return type(obj).__module__.split(".")[0] == "cudf"


def _mod(frame: Any):
    """The dataframe module that produced `frame` (pandas or cudf)."""
    if is_cudf(frame):
        import cudf

        return cudf
    return pd


def df_cons(template: Any, data: dict) -> Any:
    """Build a frame of the same engine as `template`."""
    return _mod(template).DataFrame(data)


def concat_frames(frames: Sequence[Any]) -> Any:
    """Concatenate, dropping the index. Empty-safe."""
    frames = [f for f in frames if f is not None and len(f) > 0]
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    return _mod(frames[0]).concat(frames, ignore_index=True)


def gather(vec: Any, idx: Any) -> Any:
    """vec[idx] as a positional gather -- O(E), no hash table.

    `vec` must be a dense per-vertex Series indexed 0..V-1 (position == vertex
    id, which is what dense_renumber guarantees). This is the primitive that
    replaces `merge(frontier)` in every frontier algorithm; a hash join over 1B
    rows costs roughly 3x what gather + mask costs.
    """
    return vec.take(idx).reset_index(drop=True)


def emin(a: Any, b: Any) -> Any:
    """Elementwise min of two Series, index-preserving on both engines."""
    return a.where(a <= b, b)


def full(template: Any, n: int, value: Any, dtype: str) -> Any:
    """A length-n Series of `value` with a RangeIndex, same engine as template."""
    if is_cudf(template):
        import cudf
        import cupy

        return cudf.Series(cupy.full(n, value, dtype=dtype))
    import numpy as np

    return pd.Series(np.full(n, value, dtype=dtype))


def arange(template: Any, n: int, dtype: str = "int32") -> Any:
    """0..n-1 as a Series, same engine as template."""
    if is_cudf(template):
        import cudf
        import cupy

        return cudf.Series(cupy.arange(n, dtype=dtype))
    import numpy as np

    return pd.Series(np.arange(n, dtype=dtype))


def to_host_int(value: Any) -> int:
    """Scalar reduction result -> Python int, on either engine."""
    return int(value)


def dense_renumber(edges: Any, src: str, dst: str) -> tuple[Any, Any, int]:
    """Map vertex ids to a dense 0..V-1 int32 range, monotonically.

    Returns (edges_dense, ids, V) where `ids` maps dense id -> original id.

    The mapping is MONOTONE (built from sorted uniques), so min(dense id)
    corresponds to min(original id). That is what lets WCC's "label = min vertex
    id in component" and CDLP's "tie-break to smallest label" be computed on
    dense ids and still mean the same thing -- and it lets us assert LDBC's label
    semantics without any reference output.

    Uses a dense LUT gather rather than two hash joins when the raw id space is
    small enough to make that cheaper (it is for every dataset we run: cit-Patents
    maxes near 6.0M, graph500-26 at 2**26).
    """
    mod = _mod(edges)
    # Never concat the E-length columns; unique() first keeps this at <= 2V rows.
    su = edges[src].unique()
    du = edges[dst].unique()
    su = su.to_frame(name="id") if hasattr(su, "to_frame") else mod.DataFrame({"id": su})
    du = du.to_frame(name="id") if hasattr(du, "to_frame") else mod.DataFrame({"id": du})
    ids = concat_frames([su, du])["id"].unique()
    ids = mod.Series(ids).sort_values().reset_index(drop=True)
    n = len(ids)

    max_id = to_host_int(ids.iloc[n - 1])
    if max_id < 4 * n and max_id < (1 << 31):
        # LUT gather: two O(E) gathers instead of two O(E) hash joins.
        lut = full(edges, max_id + 1, -1, "int32")
        lut.iloc[ids] = arange(edges, n, "int32").values if not is_cudf(edges) else arange(
            edges, n, "int32"
        )
        out = df_cons(
            edges,
            {
                src: gather(lut, edges[src]).astype("int32"),
                dst: gather(lut, edges[dst]).astype("int32"),
            },
        )
    else:
        lut = df_cons(edges, {"id": ids, "dense": arange(edges, n, "int32")})
        out = edges[[src, dst]].merge(lut, left_on=src, right_on="id", how="left")
        out = out.drop(columns=[src, "id"]).rename(columns={"dense": src})
        out = out.merge(lut, left_on=dst, right_on="id", how="left")
        out = out.drop(columns=[dst, "id"]).rename(columns={"dense": dst})
        out = out[[src, dst]].astype("int32")

    return out.reset_index(drop=True), ids, n


def chunk_bounds(n_rows: int, chunks: int) -> Iterator[tuple[int, int]]:
    """Contiguous [lo, hi) row ranges. `iloc` slices of these are views."""
    if chunks <= 1:
        yield 0, n_rows
        return
    step = (n_rows + chunks - 1) // chunks
    for lo in range(0, n_rows, step):
        yield lo, min(lo + step, n_rows)


def vertex_ranges(v_count: int, chunks: int) -> Iterator[tuple[int, int]]:
    """Contiguous [a, b) vertex-id ranges."""
    if chunks <= 1:
        yield 0, v_count
        return
    step = (v_count + chunks - 1) // chunks
    for a in range(0, v_count, step):
        yield a, min(a + step, v_count)


def slice_by_key(sorted_edges: Any, key: str, a: int, b: int) -> Any:
    """Rows of a key-sorted frame whose key lies in [a, b), index reset to 0.

    The reset is NOT cosmetic. `gather` returns a 0-based Series, so if a slice
    kept its original index (lo..hi-1) then `df_cons(e, {'v': e[key], 'p':
    gather(...)})` would align on index and silently fill NaN for every row --
    a bug that appears only once chunking is switched on, i.e. only at scale.
    """
    col = sorted_edges[key]
    lo = to_host_int(col.searchsorted(a, side="left"))
    hi = to_host_int(col.searchsorted(b, side="left"))
    return sorted_edges.iloc[lo:hi].reset_index(drop=True)


def rows(frame: Any, lo: int, hi: int) -> Any:
    """Contiguous positional row slice with the index reset. See slice_by_key."""
    return frame.iloc[lo:hi].reset_index(drop=True)


def mask_rows(frame: Any, mask: Any) -> Any:
    """Boolean-filter rows, index reset so later gathers stay aligned."""
    sel = frame[mask.values] if not is_cudf(frame) else frame[mask]
    return sel.reset_index(drop=True)


def align(template: Any, v_count: int, res: Any, key: str, value: str, fill: Any) -> Any:
    """A <=V-row groupby result -> a dense V-length positional vector.

    `fill` is either a scalar or a dense V-length Series supplying values for
    vertices absent from `res` (e.g. keep-your-own-label).
    """
    out = fill.copy() if hasattr(fill, "copy") else full(template, v_count, fill, "float64")
    if res is None or len(res) == 0:
        return out
    out.iloc[res[key]] = res[value].values if not is_cudf(res) else res[value]
    return out


def u64(value: int) -> Any:
    """A uint64 scalar, wrapped mod 2**64.

    Plain Python ints above 2**63 raise `OverflowError: Python int too large to
    convert to C long` when combined with a uint64 Series, so every large
    constant in the hashing path goes through here.
    """
    import numpy as np

    return np.uint64(value % (1 << 64))


def splitmix64(x: Any) -> Any:
    """SplitMix64 finalizer on a uint64 Series. Wraps identically on both engines."""
    x = x.astype("uint64")
    x = (x ^ (x // u64(1 << 30))) * u64(0xBF58476D1CE4E5B9)
    x = (x ^ (x // u64(1 << 27))) * u64(0x94D049BB133111EB)
    return x ^ (x // u64(1 << 31))
