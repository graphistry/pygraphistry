"""Schema validation for GFQL chains without execution."""

from typing import List, Optional, Union, TYPE_CHECKING, cast
import pandas as pd
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge

if TYPE_CHECKING:
    from graphistry.compute.chain import Chain

from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.numeric import NumericASTPredicate, Between
from graphistry.compute.predicates.str import Contains, Startswith, Endswith, Match, Fullmatch


def validate_chain_schema(
    g: Plottable,
    ops: Union[List[ASTObject], 'Chain'],
    collect_all: bool = False
) -> Optional[List[GFQLSchemaError]]:
    """Validate chain operations against graph schema without executing.

    This performs static analysis of the chain operations to detect:
    - References to non-existent columns
    - Type mismatches between filters and column types
    - Invalid predicate usage

    Args:
        g: The graph to validate against
        ops: Chain operations to validate
        collect_all: If True, collect all errors. If False, raise on first error.

    Returns:
        If collect_all=True: List of schema errors (empty if valid)
        If collect_all=False: None if valid

    Raises:
        GFQLSchemaError: If collect_all=False and validation fails
    """
    # Handle Chain objects
    chain_ops: List[ASTObject]
    if hasattr(ops, 'chain'):
        # ops is a Chain object, so access its chain attribute
        # The chain attribute is guaranteed to be List[ASTObject] at runtime
        chain_ops = cast(List[ASTObject], getattr(ops, 'chain'))
    else:
        chain_ops = cast(List[ASTObject], ops)

    errors: List[GFQLSchemaError] = []

    # Get available columns
    node_columns = set(g._nodes.columns) if g._nodes is not None else set()
    edge_columns = set(g._edges.columns) if g._edges is not None else set()

    for i, op in enumerate(chain_ops):
        op_errors = []

        if isinstance(op, ASTNode):
            op_errors = _validate_node_op(op, node_columns, g._nodes, collect_all)
        elif isinstance(op, ASTEdge):
            op_errors = _validate_edge_op(op, node_columns, edge_columns, g._nodes, g._edges, collect_all)
        else:
            # For new AST types (ASTLet, ASTRef, ASTCall, ASTRemoteGraph),
            # they have their own _validate_fields() methods called during construction
            # Schema validation at this level is not applicable since they don't directly
            # filter on dataframe columns like ASTNode/ASTEdge do
            # Just skip validation for these types
            pass

        # Add operation index to all errors
        for e in op_errors:
            e.context['operation_index'] = i

        if op_errors:
            if collect_all:
                errors.extend(op_errors)
            else:
                raise op_errors[0]

    return errors if collect_all else None


def _validate_node_op(op: ASTNode, node_columns: set, nodes_df: Optional[pd.DataFrame], collect_all: bool) -> List[GFQLSchemaError]:
    """Validate node operation against schema."""
    errors = []
    if op.filter_dict and nodes_df is not None:
        errors.extend(_validate_filter_dict(op.filter_dict, node_columns, nodes_df, "node", collect_all))
    return errors


def _validate_edge_op(
    op: ASTEdge,
    node_columns: set,
    edge_columns: set,
    nodes_df: Optional[pd.DataFrame],
    edges_df: Optional[pd.DataFrame],
    collect_all: bool
) -> List[GFQLSchemaError]:
    """Validate edge operation against schema."""
    errors = []

    # Validate edge filters
    if op.edge_match and edges_df is not None:
        errors.extend(_validate_filter_dict(op.edge_match, edge_columns, edges_df, "edge", collect_all))

    # Validate source node filters
    if op.source_node_match and nodes_df is not None:
        errors.extend(_validate_filter_dict(op.source_node_match, node_columns, nodes_df, "source node", collect_all))

    # Validate destination node filters
    if op.destination_node_match and nodes_df is not None:
        errors.extend(_validate_filter_dict(op.destination_node_match, node_columns, nodes_df, "destination node", collect_all))

    return errors


def _validate_filter_dict(
    filter_dict: dict,
    columns: set,
    df: pd.DataFrame,
    context: str,
    collect_all: bool = False
) -> List[GFQLSchemaError]:
    """Validate filter dictionary against dataframe schema."""
    errors = []
    for col, val in filter_dict.items():
        try:
            # Check column exists
            if col not in columns:
                error = GFQLSchemaError(
                    ErrorCode.E301,
                    f'Column "{col}" does not exist in {context} dataframe',
                    field=col,
                    value=val,
                    suggestion=f'Available columns: {", ".join(sorted(columns)[:10])}{"..." if len(columns) > 10 else ""}'
                )
                if collect_all:
                    errors.append(error)
                    continue  # Check next field
                else:
                    raise error

            # Check type compatibility
            col_dtype = df[col].dtype

            if not isinstance(val, ASTPredicate):
                # Check literal value type matches
                if pd.api.types.is_numeric_dtype(col_dtype) and isinstance(val, str):
                    error = GFQLSchemaError(
                        ErrorCode.E302,
                        f'Type mismatch: {context} column "{col}" is numeric but filter value is string',
                        field=col,
                        value=val,
                        column_type=str(col_dtype),
                        suggestion=f'Use a numeric value like {col}=123'
                    )
                    if collect_all:
                        errors.append(error)
                    else:
                        raise error
                elif pd.api.types.is_string_dtype(col_dtype) and isinstance(val, (int, float)):
                    error = GFQLSchemaError(
                        ErrorCode.E302,
                        f'Type mismatch: {context} column "{col}" is string but filter value is numeric',
                        field=col,
                        value=val,
                        column_type=str(col_dtype),
                        suggestion=f'Use a string value like {col}="value"'
                    )
                    if collect_all:
                        errors.append(error)
                    else:
                        raise error
            else:
                # Check predicate type matches column type
                if isinstance(val, (NumericASTPredicate, Between)) and not pd.api.types.is_numeric_dtype(col_dtype):
                    error = GFQLSchemaError(
                        ErrorCode.E302,
                        f'Type mismatch: numeric predicate used on non-numeric {context} column "{col}"',
                        field=col,
                        value=f"{val.__class__.__name__}(...)",
                        column_type=str(col_dtype),
                        suggestion='Use string predicates like contains() or startswith() for string columns'
                    )
                    if collect_all:
                        errors.append(error)
                    else:
                        raise error

                if isinstance(val, (Contains, Startswith, Endswith, Match, Fullmatch)) and not pd.api.types.is_string_dtype(col_dtype):
                    error = GFQLSchemaError(
                        ErrorCode.E302,
                        f'Type mismatch: string predicate used on non-string {context} column "{col}"',
                        field=col,
                        value=f"{val.__class__.__name__}(...)",
                        column_type=str(col_dtype),
                        suggestion='Use numeric predicates like gt() or lt() for numeric columns'
                    )
                    if collect_all:
                        errors.append(error)
                    else:
                        raise error

        except GFQLSchemaError:
            if not collect_all:
                raise

    return errors


# Add to Chain class
def validate_schema(self: 'Chain', g: Plottable, collect_all: bool = False) -> Optional[List[GFQLSchemaError]]:
    """Validate this chain against a graph's schema without executing.

    Args:
        g: Graph to validate against
        collect_all: If True, collect all errors. If False, raise on first.

    Returns:
        If collect_all=True: List of schema errors
        If collect_all=False: None if valid

    Raises:
        GFQLSchemaError: If collect_all=False and validation fails
    """
    return validate_chain_schema(g, self, collect_all)


# Monkey-patching moved to chain.py to avoid circular import
