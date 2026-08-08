"""Build CSR adjacency / node-id indexes from a graph's frames.

Build cost is O(E log E) (one sort), paid once per resident graph. The result is
a set of sidecar arrays over edge **row positions** — the user's ``.edges`` frame
is never reordered.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple, Union, cast

from graphistry.Engine import Engine
from graphistry.compute.typing import DataFrameT, SeriesT
from .engine_arrays import array_namespace, col_to_array
from .registry import (
    AdjacencyIndex, ColStatsFact, ColStatsRole, NodeIdIndex, NodePropIndex, PartitionValue,
    frame_fingerprint,
)
from .types import AdjacencyIndexKind, ArrayLike, ArrayNamespace


def _csr_from_keys(keys: ArrayLike, xp: ArrayNamespace) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """(keys array over E rows) -> (unique_keys, group_offsets[U+1], row_positions[E]).

    row_positions = the original row indices grouped (contiguously) by key value.
    Fully vectorized: one argsort + one boundary scan.
    """
    E = int(keys.shape[0])
    if E == 0:
        empty = keys[:0]
        return empty, xp.zeros(1, dtype=xp.int64), xp.zeros(0, dtype=xp.int64)
    order = xp.argsort(keys)                       # row positions sorted by key
    sorted_keys = keys[order]
    row_positions = order.astype(xp.int64)
    change = xp.ones(E, dtype=bool)
    change[1:] = sorted_keys[1:] != sorted_keys[:-1]
    starts = xp.nonzero(change)[0].astype(xp.int64)
    unique_keys = sorted_keys[starts]
    group_offsets = xp.concatenate([starts, xp.asarray([E], dtype=xp.int64)])
    return unique_keys, group_offsets, row_positions


def build_adjacency_index(
    edges: DataFrameT,
    kind: AdjacencyIndexKind,
    key_col: str,
    other_col: str,
    edge_id_col: Optional[str],
    engine: Engine,
    fingerprint_cols: Tuple[str, ...],
) -> AdjacencyIndex:
    xp, backend = array_namespace(engine)
    keys = col_to_array(edges, key_col, engine)
    other_values = col_to_array(edges, other_col, engine)
    unique_keys, group_offsets, row_positions = _csr_from_keys(keys, xp)
    return AdjacencyIndex(
        kind=kind,
        key_col=key_col,
        other_col=other_col,
        edge_id_col=edge_id_col,
        keys_sorted=unique_keys,
        group_offsets=group_offsets,
        row_positions=row_positions,
        other_values=other_values,
        backend=backend,
        engine=engine,
        fingerprint=frame_fingerprint(edges, fingerprint_cols, engine),
        source_ref=cast(DataFrameT, edges),
        n_edges=int(keys.shape[0]),
        n_keys=int(unique_keys.shape[0]),
    )


def build_node_id_index(
    nodes: DataFrameT,
    node_col: str,
    engine: Engine,
) -> Optional[NodeIdIndex]:
    """Sorted node-id -> first-row index, or None when node ids are NOT unique.

    ``_csr_from_keys`` returns ``row_positions`` of length E (all rows, grouped by
    key), but a node-id lookup indexes it with a *unique-key* searchsorted position
    (0..U-1). Those align ONLY when keys are unique — so we (a) collapse to the FIRST
    row position per unique key (``row_positions[group_offsets[:-1]]``, length U,
    aligned with ``unique_keys``) and (b) REFUSE (return None) when ids aren't unique:
    a unique-key CSR can't reproduce the scan's "all rows per id" semantics, so the
    caller falls back to the correct ``select_by_ids`` isin path. (Regression guard: a non-unique
    node-id index dropped reached nodes / emitted unrelated rows.)"""
    xp, backend = array_namespace(engine)
    keys = col_to_array(nodes, node_col, engine)
    unique_keys, group_offsets, row_positions = _csr_from_keys(keys, xp)
    n_keys = int(unique_keys.shape[0])
    if n_keys != int(keys.shape[0]):
        return None  # duplicate node ids -> not a valid unique index; scan fallback
    first_row_per_key = row_positions[group_offsets[:-1]]  # length U, aligned to keys
    return NodeIdIndex(
        key_col=node_col,
        keys_sorted=unique_keys,
        row_positions=first_row_per_key,
        backend=backend,
        engine=engine,
        fingerprint=frame_fingerprint(nodes, (node_col,), engine),
        source_ref=cast(DataFrameT, nodes),
        n_nodes=n_keys,
    )


def build_node_prop_index(
    nodes: DataFrameT,
    column: str,
    engine: Engine,
) -> Optional[NodePropIndex]:
    """Sorted property value -> node row positions (CSR), or None when unindexable.

    Duplicates are fine (CSR keeps every row per value) — this is the secondary
    index, so the caller still applies the remaining predicates to the gathered
    candidates. Declines (None) for anything whose ordering/equality is not
    unambiguous under a vectorized ``searchsorted`` on BOTH backends: non-integer
    dtypes (float NaN ordering, object/string on cupy) and null-bearing columns.
    Widening that gate later is additive — a decline only means "scan", never a
    wrong answer.
    """
    xp, backend = array_namespace(engine)
    try:
        keys = col_to_array(nodes, column, engine)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if str(keys.dtype.kind) not in ("i", "u"):  # numpy/cupy arrays always carry dtype.kind
        return None
    unique_keys, group_offsets, row_positions = _csr_from_keys(keys, xp)
    return NodePropIndex(
        key_col=column,
        keys_sorted=unique_keys,
        group_offsets=group_offsets,
        row_positions=row_positions,
        backend=backend,
        engine=engine,
        fingerprint=frame_fingerprint(nodes, (column,), engine),
        source_ref=cast(DataFrameT, nodes),
        n_nodes=int(keys.shape[0]),
        n_keys=int(unique_keys.shape[0]),
    )


def build_col_stats_fact(
    frame: DataFrameT,
    column: str,
    role: "ColStatsRole",
    engine: Engine,
) -> Optional[ColStatsFact]:
    """Verified min/max/null-count fact for one column of the bound frame.

    Declines (None) are decided by EXPLICIT preconditions -- column absent,
    non-integer dtype (v1), empty frame -- never by swallowing exceptions: an
    error raised by the reductions themselves is a real bug and PROPAGATES.
    A null-bearing integer column still gets a fact (null_count recorded,
    min/max omitted); consumers require null_count == 0 before trusting bounds,
    so such a fact can only ever route to the scan. Widening the dtype gate
    later is additive -- a decline only means "scan", never a wrong answer.
    """
    from graphistry.Engine import POLARS_ENGINES
    n_unique: Optional[int] = None
    if engine in POLARS_ENGINES:
        pl_frame: Any = frame  # engine seam: polars frame rides DataFrameT
        if column not in pl_frame.columns:
            return None
        s = pl_frame.get_column(column)
        if not s.dtype.is_integer():
            return None
        if int(s.len()) == 0:
            return None
        null_count = int(s.null_count())
        mn, mx = ((None, None) if null_count
                  else (int(s.min()), int(s.max())))
        if role == "nodes":
            n_unique = int(s.n_unique())
    else:
        if column not in frame.columns:
            return None
        ser = frame[column]
        if _dtype_kind(ser) not in ("i", "u"):
            return None
        if int(ser.shape[0]) == 0:
            return None
        null_count = int(ser.isna().sum())
        mn, mx = ((None, None) if null_count
                  else (int(ser.min()), int(ser.max())))
        if role == "nodes":
            n_unique = int(ser.nunique())
    return ColStatsFact(
        role=role,
        column=column,
        min_val=mn,
        max_val=mx,
        null_count=null_count,
        is_integer=True,
        engine=engine,
        n_unique=n_unique,
        fingerprint=frame_fingerprint(frame, (column,), engine),
        source_ref=frame,
    )


_MAX_COL_STATS_PARTITIONS = 256


def _dtype_kind(series: SeriesT) -> str:
    """The numpy-style dtype kind letter: 'i'/'u' integer, 'f' float, 'b' bool,
    'M' datetime, 'O' object and every extension dtype.

    Always present: ``pandas.api.extensions.ExtensionDtype`` DEFINES ``kind``, so
    every pandas and cudf dtype has one (measured: ArrowDtype, Interval, Period,
    Sparse, Categorical, DatetimeTZ, and cudf's ListDtype/StructDtype/
    Decimal128Dtype all report a kind). No default is needed.

    pandas' ``is_integer_dtype`` and friends are NOT a substitute here: they
    RAISE ``TypeError`` on cudf ``ListDtype`` (measured), which is precisely the
    exotic dtype this module must DECLINE cleanly rather than crash on.
    """
    return str(series.dtype.kind)  # type: ignore[union-attr]  # engine seam: every backend dtype has .kind


def _column_to_pylist(series: SeriesT, engine: Engine) -> List[PartitionValue]:
    """Host-side values of a pandas/cudf column, dispatched on the ENGINE.

    cudf crosses to the host via arrow rather than ``to_pandas()``, which
    segfaults on string columns in some RAPIDS builds. The engine is already
    known at every call site, so this dispatches on it instead of probing the
    object for a ``to_arrow`` attribute.
    """
    if engine == Engine.CUDF:
        return list(series.to_arrow().to_pylist())  # type: ignore[union-attr]  # engine seam: cudf only
    return list(series.tolist())  # type: ignore[union-attr]  # engine seam: pandas only


def _type_column_is_scalar(frame: DataFrameT, type_column: str, engine: Engine) -> bool:
    """True iff the type column holds SCALARS we can key a partition on.

    List/struct-valued type columns (GFQL's ``labels`` list convention, which
    ``resolve_filter_column`` rewrites ``label__X`` into) are not equality-
    addressable: a query never produces a list-valued partition key, so a fact
    built on one could never be consulted. pandas raises ``unhashable type`` on
    such a groupby while polars happily groups by list -- so this is an EXPLICIT
    precondition on both engines rather than an engine-dependent accident.
    """
    from graphistry.Engine import POLARS_ENGINES
    if engine in POLARS_ENGINES:
        import polars as pl
        dtype: Any = frame.get_column(type_column).dtype  # engine seam
        return not isinstance(dtype, (pl.List, pl.Array, pl.Struct, pl.Object))
    series = frame[type_column]
    if _dtype_kind(series) not in ("O", "S", "U"):
        return True  # numeric/bool/datetime dtypes are scalar by construction
    non_null = series.dropna()
    if int(non_null.shape[0]) == 0:
        return True
    sample = _column_to_pylist(non_null.head(1), engine)
    return not isinstance(sample[0], (list, tuple, set, dict, bytearray))


def build_col_stats_facts_by_type(
    frame: DataFrameT,
    column: str,
    role: "ColStatsRole",
    type_column: str,
    engine: Engine,
) -> List[ColStatsFact]:
    """One fact per value of ``type_column``, from a SINGLE grouped pass.

    Multi-type graphs defeat whole-frame facts: an interval over every node id
    says nothing about the ids of one label, so bound proofs that a homogeneous
    graph passes fail outright. Partition facts restore them per label, and a
    partition fact upper-bounds any further-filtered subset of that partition --
    the same conservative direction as the whole-frame fact.

    Declines (empty list) are decided by EXPLICIT preconditions -- either column
    absent, non-integer value dtype (v1), a null-bearing value column, a
    float/null-bearing type column (NaN group keys are not equality-addressable),
    empty frame, or more than ``_MAX_COL_STATS_PARTITIONS`` distinct types --
    never by swallowing exceptions: an error raised by the aggregation itself is
    a real bug and PROPAGATES. Widening a gate later is additive; a decline only
    means "scan". Null-bearing value columns are declined rather than recorded
    with ``null_count > 0`` (as the whole-frame builder does) because consumers
    require zero nulls before trusting bounds, so such partition facts could only
    ever route to the scan they were built to avoid.
    """
    from graphistry.Engine import POLARS_ENGINES
    fingerprint = frame_fingerprint(frame, tuple(sorted({column, type_column})), engine)

    def fact(type_value: PartitionValue, mn: int, mx: int, n_unique: Optional[int]) -> ColStatsFact:
        return ColStatsFact(
            role=role, column=column, min_val=mn, max_val=mx,
            null_count=0, is_integer=True, engine=engine,
            n_unique=n_unique, type_column=type_column, type_value=type_value,
            fingerprint=fingerprint, source_ref=frame,
        )

    want_unique = role == "nodes"
    facts: List[ColStatsFact] = []
    if engine in POLARS_ENGINES:
        import polars as pl
        pl_frame: Any = frame  # engine seam: polars frame rides DataFrameT
        if column not in pl_frame.columns or type_column not in pl_frame.columns:
            return []
        values = pl_frame.get_column(column)
        if not values.dtype.is_integer() or int(values.null_count()) > 0:
            return []
        if not _type_column_is_scalar(frame, type_column, engine):
            return []
        types = pl_frame.get_column(type_column)
        if types.dtype.is_float() or int(types.null_count()) > 0:
            return []
        if int(pl_frame.height) == 0 or int(types.n_unique()) > _MAX_COL_STATS_PARTITIONS:
            return []
        aggs = [pl.col(column).min().alias("_mn"), pl.col(column).max().alias("_mx")]
        if want_unique:
            aggs.append(pl.col(column).n_unique().alias("_nuniq"))
        for row in pl_frame.group_by(type_column).agg(aggs).iter_rows(named=True):
            facts.append(fact(row[type_column], int(row["_mn"]), int(row["_mx"]),
                              int(row["_nuniq"]) if want_unique else None))
        return facts

    if column not in frame.columns or type_column not in frame.columns:
        return []
    values_ser = frame[column]
    if _dtype_kind(values_ser) not in ("i", "u") or int(values_ser.isna().sum()) > 0:
        return []
    if not _type_column_is_scalar(frame, type_column, engine):
        return []
    types_ser = frame[type_column]
    if _dtype_kind(types_ser) == "f" or int(types_ser.isna().sum()) > 0:
        return []
    if int(frame.shape[0]) == 0 or int(types_ser.nunique()) > _MAX_COL_STATS_PARTITIONS:
        return []
    names = ["min", "max"] + (["nunique"] if want_unique else [])
    grouped = frame.groupby(type_column, sort=False)[column].agg(names).reset_index()
    cols = {name: _column_to_pylist(grouped[name], engine) for name in [type_column] + names}
    for i, type_value in enumerate(cols[type_column]):
        facts.append(fact(type_value, int(cols["min"][i]), int(cols["max"][i]),
                          int(cols["nunique"][i]) if want_unique else None))
    return facts
