from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd
from graphistry.Engine import EngineAbstract, df_to_engine, resolve_engine
from graphistry.util import setup_logger

from graphistry.Plottable import Plottable
from .predicates.ASTPredicate import ASTPredicate
from .typing import DataFrameT


logger = setup_logger(__name__)


def _is_membership_filter_value(value: Any) -> bool:
    return isinstance(value, (list, tuple, set, frozenset, pd.Index, pd.Series))


def _looks_like_edge_dataframe(df: DataFrameT) -> bool:
    cols = {str(col) for col in df.columns}
    return {"s", "d"} <= cols or {"src", "dst"} <= cols or "edge_id" in cols


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
        dtype_txt = str(series.dtype).lower()
        if dtype_txt != "object" and pd.api.types.is_string_dtype(series.dtype):
            mask = series == label
            if hasattr(mask, "where") and hasattr(series, "isna"):
                return mask.where(~series.isna(), False)
            return mask
    except Exception:
        pass
    return series.apply(lambda value: label in _normalize_labels_cell(value))


def resolve_filter_column(df: DataFrameT, col: str, val: Any) -> Tuple[str, Any]:
    if col in df.columns:
        return col, val

    if col.startswith("label__") and val is True:
        label = col[len("label__") :]
        if "labels" in df.columns:
            return "labels", label
        if "type" in df.columns and _looks_like_edge_dataframe(df):
            return "type", label

    from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError

    raise GFQLSchemaError(
        ErrorCode.E301,
        f'Column "{col}" does not exist in dataframe',
        field=col,
        value=val,
        suggestion=f'Available columns: {", ".join(df.columns[:10])}{"..." if len(df.columns) > 10 else ""}'
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
            if pd.api.types.is_numeric_dtype(col_dtype) and isinstance(resolved_val, str):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: column "{resolved_col}" is numeric but filter value is string',
                    field=col,
                    value=resolved_val,
                    column_type=str(col_dtype),
                    suggestion=f'Use a numeric value like {col}=123'
                )
            elif pd.api.types.is_string_dtype(col_dtype) and isinstance(resolved_val, (int, float)) and not isinstance(resolved_val, bool):
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
            if isinstance(resolved_val, (NumericASTPredicate, Between)) and not pd.api.types.is_numeric_dtype(col_dtype):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: numeric predicate used on non-numeric column "{resolved_col}"',
                    field=col,
                    value=f"{resolved_val.__class__.__name__}(...)",
                    column_type=str(col_dtype),
                    suggestion='Use string predicates like contains() or startswith() for string columns'
                )

            # Check string predicates on non-string columns
            if isinstance(resolved_val, (Contains, Startswith, Endswith, Match)) and not pd.api.types.is_string_dtype(col_dtype):
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
                hits = hits & df[resolved_col].isin(list(resolved_val))
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
