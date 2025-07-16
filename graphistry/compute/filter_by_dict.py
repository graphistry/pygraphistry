from typing import Dict, Optional, Union
import pandas as pd
from graphistry.Engine import EngineAbstract, df_to_engine, resolve_engine, s_cons
from graphistry.util import setup_logger

from graphistry.Plottable import Plottable
from .predicates.ASTPredicate import ASTPredicate
from .typing import DataFrameT


logger = setup_logger(__name__)


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

    predicates: Dict[str, ASTPredicate] = {}
    for col, val in filter_dict.items():
        if col not in df.columns:
            raise GFQLSchemaError(
                ErrorCode.E301,
                f'Column "{col}" does not exist in dataframe',
                field=col,
                value=val,
                suggestion=f'Available columns: {", ".join(df.columns[:10])}{"..." if len(df.columns) > 10 else ""}'
            )

        # Type checking for non-predicate values
        if not isinstance(val, ASTPredicate):
            # Check for obvious type mismatches
            col_dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(col_dtype) and isinstance(val, str):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: column "{col}" is numeric but filter value is string',
                    field=col,
                    value=val,
                    column_type=str(col_dtype),
                    suggestion=f'Use a numeric value like {col}=123'
                )
            elif pd.api.types.is_string_dtype(col_dtype) and isinstance(val, (int, float)):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: column "{col}" is string but filter value is numeric',
                    field=col,
                    value=val,
                    column_type=str(col_dtype),
                    suggestion=f'Use a string value like {col}="value"'
                )
        else:
            # Validate predicates for appropriate column types
            from .predicates.numeric import NumericASTPredicate, Between
            from .predicates.str import Contains, Startswith, Endswith, Match

            col_dtype = df[col].dtype

            # Check numeric predicates on non-numeric columns
            if isinstance(val, (NumericASTPredicate, Between)) and not pd.api.types.is_numeric_dtype(col_dtype):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: numeric predicate used on non-numeric column "{col}"',
                    field=col,
                    value=f"{val.__class__.__name__}(...)",
                    column_type=str(col_dtype),
                    suggestion='Use string predicates like contains() or startswith() for string columns'
                )

            # Check string predicates on non-string columns
            if isinstance(val, (Contains, Startswith, Endswith, Match)) and not pd.api.types.is_string_dtype(col_dtype):
                raise GFQLSchemaError(
                    ErrorCode.E302,
                    f'Type mismatch: string predicate used on non-string column "{col}"',
                    field=col,
                    value=f"{val.__class__.__name__}(...)",
                    column_type=str(col_dtype),
                    suggestion='Use numeric predicates like gt() or lt() for numeric columns'
                )

            predicates[col] = val
    filter_dict_concrete = filter_dict if not predicates else {
        k: v
        for k, v in filter_dict.items()
        if not isinstance(v, ASTPredicate)
    }

    if filter_dict_concrete:
        S = s_cons(engine_concrete)
        hits = (df[list(filter_dict_concrete)] == S(filter_dict_concrete)).all(axis=1)
    else:
        hits = df[[]].assign(x=True).x
    if predicates:
        for col, op in predicates.items():
            hits = hits & op(df[col])
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
