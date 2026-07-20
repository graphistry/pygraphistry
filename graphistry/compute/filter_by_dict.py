from typing import Any, Dict, Mapping, Optional, Tuple, Union, cast
import pandas as pd
from graphistry.Engine import EngineAbstract, df_to_engine, resolve_engine
from graphistry.util import setup_logger

from graphistry.Plottable import Plottable
from .predicates.ASTPredicate import ASTPredicate
from .typing import DataFrameT, DType, NodeDtypes


logger = setup_logger(__name__)


def _is_membership_filter_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset, pd.Index, pd.Series))


def _looks_like_edge_dataframe(df: DataFrameT) -> bool:
    cols = {str(col) for col in df.columns}
    return {"s", "d"} <= cols or {"src", "dst"} <= cols or "edge_id" in cols


def _dtype_text(dtype: Any) -> str:
    try:
        return str(dtype).lower()
    except Exception:
        return ""


def _is_numeric_dtype_safe(dtype: Any) -> bool:
    try:
        return bool(pd.api.types.is_numeric_dtype(dtype))
    except Exception:
        kind = getattr(dtype, "kind", None)
        if isinstance(kind, str) and kind in {"b", "i", "u", "f", "c"}:
            return True
        dtype_txt = _dtype_text(dtype)
        return any(token in dtype_txt for token in ("bool", "int", "float", "double", "decimal"))


def _is_string_dtype_safe(dtype: Any) -> bool:
    try:
        return bool(pd.api.types.is_string_dtype(dtype))
    except Exception:
        dtype_txt = _dtype_text(dtype)
        return dtype_txt == "object" or "string" in dtype_txt or dtype_txt.endswith("[python]")


def _normalize_labels_cell(value: Any) -> Tuple[Any, ...]:
    if value is None:
        return ()
    try:
        marker = pd.isna(value)
    except Exception:
        marker = False
    if isinstance(marker, bool) and marker:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple, set, frozenset, pd.Index, pd.Series)):
        return tuple(value)
    return (value,)


def _label_series_contains(series: Any, label: str) -> Any:
    try:
        dtype_txt = _dtype_text(series.dtype)
        if dtype_txt != "object" and _is_string_dtype_safe(series.dtype):
            mask = series == label
            if hasattr(mask, "where") and hasattr(series, "isna"):
                return mask.where(~series.isna(), False)
            return mask
    except Exception:
        pass
    if hasattr(series, "to_pandas") and series.__class__.__module__.startswith("cudf"):
        mask_pd = series.to_pandas().apply(lambda value: label in _normalize_labels_cell(value))
        try:
            import cudf  # type: ignore
            return cudf.Series(mask_pd.tolist(), index=series.index, dtype="bool")
        except Exception:
            return mask_pd
    return series.apply(lambda value: label in _normalize_labels_cell(value))


def _column_names(df: DataFrameT) -> Any:
    """Column names without resolving a polars LazyFrame schema via ``.columns``, which warns
    (PerformanceWarning) and re-resolves on every access — this runs per predicate per call on
    the seeded fast path. ``collect_schema().names()`` is the sanctioned lazy form."""
    collect_schema = getattr(df, "collect_schema", None)
    if collect_schema is not None and type(df).__name__ == "LazyFrame":
        return collect_schema().names()
    return df.columns


def resolve_filter_column(df: DataFrameT, col: str, val: Any) -> Tuple[str, Any]:
    names = _column_names(df)
    if col in names:
        return col, val

    if col.startswith("label__") and val is True:
        label = col[len("label__") :]
        if "labels" in names:
            return "labels", label
        if "type" in names and not _looks_like_edge_dataframe(df):
            return "type", label

    from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError

    raise GFQLSchemaError(
        ErrorCode.E301,
        f'Column "{col}" does not exist in dataframe',
        field=col,
        value=val,
        suggestion=f'Available columns: {", ".join(list(names)[:10])}{"..." if len(names) > 10 else ""}'
    )


def filter_by_dict(df: DataFrameT, filter_dict: Optional[dict] = None, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> DataFrameT:
    """
    return df where rows match all values in filter_dict
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    if filter_dict is None or filter_dict == {}:
        return df

    engine_concrete = resolve_engine(engine, df)
    df = df_to_engine(df, engine_concrete)
    logger.debug('filter_by_dict engine: %s => %s', engine, engine_concrete)

    from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError

    predicates: Dict[str, Tuple[str, ASTPredicate]] = {}
    concrete_filters: Dict[str, Tuple[str, Any]] = {}
    for col, val in filter_dict.items():
        resolved_col, resolved_val = resolve_filter_column(df, col, val)

        # Type checking for non-predicate values
        if not isinstance(resolved_val, ASTPredicate):
            if _is_membership_filter_value(resolved_val):
                concrete_filters[col] = (resolved_col, list(resolved_val))
                continue
            if len(df) == 0:
                concrete_filters[col] = (resolved_col, resolved_val)
                continue
            # Check for obvious type mismatches
            col_dtype = df[resolved_col].dtype
            if _is_numeric_dtype_safe(col_dtype) and isinstance(resolved_val, str):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: column "{resolved_col}" is numeric but filter value is string',
                    field=col,
                    value=resolved_val,
                    column_type=str(col_dtype),
                    suggestion=f'Use a numeric value like {col}=123'
                )
            elif _is_string_dtype_safe(col_dtype) and isinstance(resolved_val, (int, float)) and not isinstance(resolved_val, bool):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: column "{resolved_col}" is string but filter value is numeric',
                    field=col,
                    value=resolved_val,
                    column_type=str(col_dtype),
                    suggestion=f'Use a string value like {col}="value"'
                )
            concrete_filters[col] = (resolved_col, resolved_val)
        else:
            # Validate predicates for appropriate column types
            from .predicates.numeric import NumericASTPredicate, Between
            from .predicates.str import Contains, Startswith, Endswith, Match

            col_dtype = df[resolved_col].dtype

            # Check numeric predicates on non-numeric columns
            if isinstance(resolved_val, (NumericASTPredicate, Between)) and not _is_numeric_dtype_safe(col_dtype):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: numeric predicate used on non-numeric column "{resolved_col}"',
                    field=col,
                    value=f"{resolved_val.__class__.__name__}(...)",
                    column_type=str(col_dtype),
                    suggestion='Use string predicates like contains() or startswith() for string columns'
                )

            # Check string predicates on non-string columns
            if isinstance(resolved_val, (Contains, Startswith, Endswith, Match)) and not _is_string_dtype_safe(col_dtype):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: string predicate used on non-string column "{resolved_col}"',
                    field=col,
                    value=f"{resolved_val.__class__.__name__}(...)",
                    column_type=str(col_dtype),
                    suggestion='Use numeric predicates like gt() or lt() for numeric columns'
                )

            predicates[col] = (resolved_col, resolved_val)

    hits = df[[]].assign(x=True).x
    if concrete_filters:
        for original_col, (resolved_col, resolved_val) in concrete_filters.items():
            if original_col.startswith("label__") and resolved_col == "labels" and isinstance(resolved_val, str):
                hits = hits & _label_series_contains(df[resolved_col], resolved_val)
            elif _is_membership_filter_value(resolved_val):
                # openCypher/SQL 3VL: `null IN [...]` is null -> a NULL cell is NOT a member (and a
                # NULL in the list cannot make a null cell match). `& notna()` excludes null cells —
                # a no-op for pandas (its isin already excludes a NaN cell here) but fixes cuDF, which
                # otherwise matches a null cell against a None/NaN list element.
                hits = hits & df[resolved_col].isin(list(resolved_val)) & df[resolved_col].notna()
            else:
                hits = hits & (df[resolved_col] == resolved_val)
    if predicates:
        for resolved_col, op in predicates.values():
            hits = hits & op(df[resolved_col])
    return df[hits]


def filter_nodes_by_dict(self: Plottable, filter_dict: Optional[dict] = None, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
    """
    filter nodes to those that match all values in filter_dict
    """
    nodes2 = filter_by_dict(self._nodes, filter_dict, engine)
    return self.nodes(nodes2)


def filter_edges_by_dict(self: Plottable, filter_dict: Optional[dict] = None, engine: Union[EngineAbstract, str] = EngineAbstract.AUTO) -> Plottable:
    """
    filter edges to those that match all values in filter_dict
    """
    edges2 = filter_by_dict(self._edges, filter_dict, engine)
    return self.edges(edges2)


class _LazyNodeDtypes(Mapping[str, DType]):
    """Node dtypes, materialized on first lookup.

    Reading them costs a full engine conversion, and only connected-join pushdown ever asks --
    a path most queries never reach. Computing eagerly charged every string query for a frame
    it then discarded (~65ms per million polars rows). Stay a Mapping so callers are unchanged.
    """

    def __init__(self, g: Plottable, engine: Union[EngineAbstract, str]) -> None:
        self._g = g
        self._engine = engine
        self._resolved: Optional[NodeDtypes] = None

    def _materialize(self) -> NodeDtypes:
        if self._resolved is None:
            self._resolved = _read_node_dtypes(self._g, self._engine)
        return self._resolved

    def __getitem__(self, key: str) -> DType:
        return self._materialize()[key]

    def __iter__(self) -> Any:
        return iter(self._materialize())

    def __len__(self) -> int:
        return len(self._materialize())


def _node_dtypes_for_pushdown(
    g: Plottable,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
) -> Optional[NodeDtypes]:
    """Node dtypes for the pushdown gate, or None when there is no graph to read."""
    if g._nodes is None:
        # No graph: the caller falls back to value-type rules, as a bare compile_cypher() does.
        # Distinct from failing to read a graph we have, which must fail closed.
        return None
    return _LazyNodeDtypes(g, engine)


def _object_column_holds_non_strings(frame: DataFrameT, column: str, dtype: DType) -> bool:
    """Whether `column` is an object column whose values `.str` would reject.

    `object` says nothing about contents: pandas stores ordinary strings that way, and also a
    numeric column that acquired a `None`. String predicates are admitted on dtype alone -- by
    this gate and by `filter_by_dict` alike -- but `.str` fails on the values, leaking a raw
    AttributeError. Dropping the column leaves the gate with nothing to look up, so it fails
    closed and the residual answers.

    Keep only the kinds that stay valid under SUBSETTING. `StringMethods._validate` admits
    {string, empty, bytes, mixed, mixed-integer}, but it runs on the frame it is handed -- and
    the frame we inspect is the source, while the pushed filter runs on the join's candidate
    subset. `string` and `empty` survive that (a subset of strings is string or empty), but
    `mixed`/`mixed-integer` do not: drop the strings and they collapse to integer/floating/
    boolean, which `.str` rejects. `bytes` is excluded separately -- it passes `_validate` yet
    `str.contains` forbids it.

    Mirroring a rule is not enough when we evaluate it against a different frame than the
    executor does, so admit only what no subset can invalidate.
    """
    import pandas as _pd

    try:
        is_object = bool(_pd.api.types.is_object_dtype(dtype))
    except Exception:
        return False
    if not is_object:
        return False
    try:
        inferred = _pd.api.types.infer_dtype(frame[column], skipna=True)
    except Exception:
        # It IS an object column and we cannot read its values, so we cannot tell whether
        # `.str` would reject them. Omit it and let the residual answer, matching
        # `_read_node_dtypes`, which returns {} for a schema it cannot read.
        return True
    return inferred not in {"string", "empty"}


def _read_node_dtypes(
    g: Plottable,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
) -> NodeDtypes:
    """Column dtypes the executor will filter on, for the pushdown gate.

    A pushed filter is schema-validated against the real column while the row residual
    evaluates leniently, so the planner needs dtypes to avoid turning a correct empty result
    into a type error. An unreadable schema yields an empty mapping, which fails every column
    lookup and so falls back to the residual.
    """
    nodes = g._nodes
    if nodes is None:
        return {}
    try:
        # Classify the frame the EXECUTOR will filter, not the one the caller handed in.
        # `filter_by_dict` validates post-materialization dtypes, so judging the pre-conversion
        # frame lets the two disagree (polars Decimal reads numeric here, object there).
        #
        # Convert the whole frame, not an empty probe: polars -> pandas is DATA-dependent, so
        # `head(0)` reports a different class than the real conversion for a nullable Boolean
        # (`bool` empty, `object` with a null in it). Identity for pandas; for other engines
        # this costs a real conversion, which is why the caller defers it until a pushdown
        # decision actually needs it.
        probe = df_to_engine(nodes, resolve_engine(cast(Any, engine), nodes), warn=False)
        # zip rather than `.items()`: pandas/cuDF expose a column-indexed Series, polars a list.
        dtypes = {str(col): dtype for col, dtype in zip(list(probe.columns), list(probe.dtypes))}
        return {
            col: dtype
            for col, dtype in dtypes.items()
            if not _object_column_holds_non_strings(probe, col, dtype)
        }
    except Exception:
        # We have a graph but could not read its schema, so we know nothing about any column.
        # An empty mapping fails every column lookup, which fails closed to the residual.
        return {}
