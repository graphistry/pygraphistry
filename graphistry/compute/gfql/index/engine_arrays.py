"""Engine-polymorphic array helpers for GFQL physical indexes.

The index core (build + lookup) is written once against a tiny array protocol
(``searchsorted``, ``argsort``, ``cumsum``, ``arange``, ``repeat``) so the same
CSR + searchsorted gather runs on:

- pandas  -> numpy host arrays, ``df.iloc`` to gather rows
- cudf    -> cupy device arrays, ``df.iloc`` to gather rows
- polars / polars-gpu -> numpy host arrays, polars row-gather

Vectorization-first: no per-element Python work, no ``.to_list()`` ping-pong.
"""
from __future__ import annotations

from typing import Any, Tuple

from graphistry.Engine import Engine


def array_namespace(engine: Engine) -> Tuple[Any, str]:
    """Return (array module, backend tag) for an engine.

    cudf indexes keep their arrays on-device (cupy); everything else uses numpy
    host arrays. The frontier of a seeded query is tiny, so host-side
    searchsorted is cheap even when the frame itself is on GPU (polars-gpu).
    """
    import numpy as np

    if engine == Engine.CUDF:
        try:
            import cupy as cp  # type: ignore

            return cp, "cupy"
        except Exception:  # pragma: no cover - cupy always present with cudf
            return np, "numpy"
    return np, "numpy"


def col_to_array(df: Any, col: str, engine: Engine) -> Any:
    """Extract a column as a backend-native 1-D array (numpy or cupy)."""
    if engine in (Engine.POLARS, Engine.POLARS_GPU):
        return df.get_column(col).to_numpy()
    if engine == Engine.CUDF:
        # cudf Series -> cupy array (stays on device)
        return df[col].values
    return df[col].to_numpy()


def ids_to_array(ids: Any, col: str, engine: Engine) -> Any:
    """Frontier ids (a frame/Series) -> backend array, matching index backend."""
    return col_to_array(ids, col, engine)


def to_backend(arr: Any, backend: str) -> Any:
    """Move a host/device array to the index backend (numpy<->cupy)."""
    import numpy as np

    if backend == "cupy":
        import cupy as cp  # type: ignore

        return cp.asarray(arr)
    if "cupy" in type(arr).__module__:
        return cp_to_numpy(arr)
    return np.asarray(arr)


def cp_to_numpy(arr: Any) -> Any:
    try:
        return arr.get()
    except Exception:
        import numpy as np

        return np.asarray(arr)


def take_rows(df: Any, positions: Any, engine: Engine) -> Any:
    """Positionally gather rows of ``df`` by an integer array ``positions``.

    ``positions`` is a backend array (numpy for pandas/polars, cupy for cudf).
    Returns a frame of the same engine; row order follows ``positions``.
    """
    if engine in (Engine.POLARS, Engine.POLARS_GPU):
        import numpy as np

        idx = np.asarray(positions)
        return df[idx]
    # pandas / cudf: iloc accepts numpy (pandas) or cupy (cudf) int arrays
    return df.iloc[positions]


def select_by_ids(df: Any, col: str, ids: Any, engine: Engine) -> Any:
    """Return rows of ``df`` whose ``col`` is in the id array ``ids`` (set semantics,
    preserves df row order). Engine-polymorphic, vectorized."""
    if engine in (Engine.POLARS, Engine.POLARS_GPU):
        import numpy as np
        import polars as pl

        # Semi-join (not Expr.is_in(Series), which polars 1.42 deprecates as ambiguous —
        # pola-rs/polars#22149) — vectorized AND preserves the left (df) row order, which
        # the node materialization relies on (table-order parity with the scan).
        ids_df = pl.DataFrame({col: np.asarray(ids)}).cast({col: df.schema[col]})
        return df.join(ids_df.unique(), on=col, how="semi")
    if engine == Engine.CUDF:
        import cudf  # type: ignore

        return df[df[col].isin(cudf.Series(ids))]
    import numpy as np

    return df[df[col].isin(np.asarray(ids))]


def set_difference(cand: Any, visited: Any, xp: Any) -> Any:
    """cand minus visited (both backend arrays), vectorized via sorted membership."""
    if int(visited.shape[0]) == 0:
        return cand
    if int(cand.shape[0]) == 0:
        return cand
    vs = xp.sort(visited)
    pos = xp.searchsorted(vs, cand)
    pos_c = xp.where(pos < vs.shape[0], pos, vs.shape[0] - 1)
    ismember = vs[pos_c] == cand
    return cand[~ismember]


def union1d(a: Any, b: Any, xp: Any) -> Any:
    if int(a.shape[0]) == 0:
        return xp.unique(b)
    if int(b.shape[0]) == 0:
        return xp.unique(a)
    return xp.unique(xp.concatenate([a, b]))
