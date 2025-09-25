"""Schema validation for GFQL chains without execution."""

from typing import List, Optional, Union, TYPE_CHECKING, cast
import pandas as pd
from graphistry.Plottable import Plottable
from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge, ASTLet, ASTRef, ASTRemoteGraph, ASTCall

if TYPE_CHECKING:
    from graphistry.compute.chain import Chain
from graphistry.compute.exceptions import ErrorCode, GFQLSchemaError
from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.numeric import NumericASTPredicate, Between
from graphistry.compute.predicates.str import Contains, Startswith, Endswith, Match


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
    from graphistry.compute.chain import Chain
    if isinstance(ops, Chain):
        chain_ops = ops.chain
    else:
        chain_ops = ops

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
        elif isinstance(op, ASTLet):
            op_errors = _validate_querydag_op(op, g, collect_all)
        elif isinstance(op, ASTRef):
            op_errors = _validate_chainref_op(op, g, collect_all)
        elif isinstance(op, ASTRemoteGraph):
            op_errors = _validate_remotegraph_op(op, collect_all)
        elif isinstance(op, ASTCall):
            op_errors = _validate_call_op(op, node_columns, edge_columns, collect_all)

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


def _validate_querydag_op(op: ASTLet, g: Plottable, collect_all: bool) -> List[GFQLSchemaError]:
    """Validate Let operation against schema."""
    errors = []
    
    # Validate each binding in the DAG
    for binding_name, binding_value in op.bindings.items():
        try:
            # Recursively validate each binding as if it's a single operation
            binding_errors = validate_chain_schema(g, [binding_value], collect_all=True)  # type: ignore
            
            # Add binding context to errors
            if binding_errors:
                for error in binding_errors:
                    error.context['dag_binding'] = binding_name
                
            if binding_errors:
                if collect_all:
                    errors.extend(binding_errors)
                else:
                    raise binding_errors[0]
                    
        except GFQLSchemaError as e:
            e.context['dag_binding'] = binding_name
            if collect_all:
                errors.append(e)
            else:
                raise
    
    return errors


def _validate_chainref_op(op: ASTRef, g: Plottable, collect_all: bool) -> List[GFQLSchemaError]:
    """Validate ChainRef operation against schema."""
    errors = []
    
    # Validate the chain operations in the ChainRef
    if op.chain:
        try:
            chain_errors = validate_chain_schema(g, op.chain, collect_all=True)
            
            # Add ChainRef context to errors
            if chain_errors:
                for error in chain_errors:
                    error.context['chain_ref'] = op.ref
                
            if chain_errors:
                if collect_all:
                    errors.extend(chain_errors)
                else:
                    raise chain_errors[0]
                    
        except GFQLSchemaError as e:
            e.context['chain_ref'] = op.ref
            if collect_all:
                errors.append(e)
            else:
                raise
    
    # Note: We don't validate that op.ref exists here since that's handled
    # by the DAG dependency validation in chain_let.py
    
    return errors


def _validate_remotegraph_op(op: ASTRemoteGraph, collect_all: bool) -> List[GFQLSchemaError]:
    """Validate RemoteGraph operation against schema."""
    errors = []
    
    # Validate dataset_id format
    if not op.dataset_id or not isinstance(op.dataset_id, str):
        error = GFQLSchemaError(
            ErrorCode.E303,
            'RemoteGraph dataset_id must be a non-empty string',
            field='dataset_id',
            value=op.dataset_id,
            suggestion='Provide a valid dataset identifier string'
        )
        if collect_all:
            errors.append(error)
        else:
            raise error
    
    # Validate token format if provided
    if op.token is not None and not isinstance(op.token, str):
        error = GFQLSchemaError(
            ErrorCode.E303,
            'RemoteGraph token must be a string if provided',
            field='token',
            value=type(op.token).__name__,
            suggestion='Provide a valid token string or None'
        )
        if collect_all:
            errors.append(error)
        else:
            raise error
    
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

                if isinstance(val, (Contains, Startswith, Endswith, Match)) and not pd.api.types.is_string_dtype(col_dtype):
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


def _validate_call_op(
    op: ASTCall,
    node_columns: set,
    edge_columns: set,
    collect_all: bool = False
) -> List[GFQLSchemaError]:
    """Validate Call operation schema requirements.
    
    Checks that all columns required by the called method exist in the graph.
    Uses the schema_effects metadata from the safelist to determine requirements.
    
    Args:
        op: ASTCall operation to validate
        node_columns: Set of available node column names
        edge_columns: Set of available edge column names
        collect_all: If True, collect all errors. If False, raise on first error.
        
    Returns:
        List of schema errors found (empty if valid)
        
    Raises:
        GFQLSchemaError: If collect_all=False and validation fails
    """
    errors: List[GFQLSchemaError] = []
    
    # Import safelist to get schema effects
    from graphistry.compute.gfql.call_safelist import SAFELIST_V1
    
    # Check if method is in safelist
    if op.function not in SAFELIST_V1:
        # This should have been caught by parameter validation already
        return errors
    
    method_info = SAFELIST_V1[op.function]
    
    # Check if method has schema effects defined
    if 'schema_effects' not in method_info:
        # Method doesn't define schema effects, so we can't validate
        return errors
    
    schema_effects = method_info['schema_effects']
    
    # Get required columns based on parameters
    if 'requires_node_cols' in schema_effects:
        if callable(schema_effects['requires_node_cols']):
            required_node_cols = schema_effects['requires_node_cols'](op.params)
        else:
            required_node_cols = schema_effects['requires_node_cols']
        
        for col in required_node_cols:
            if col not in node_columns:
                error = GFQLSchemaError(
                    ErrorCode.E301,
                    f'Call operation "{op.function}" requires node column "{col}" which does not exist',
                    field=f'{op.function}.{col}',
                    value=col,
                    suggestion=f'Available node columns: {", ".join(sorted(node_columns)[:10])}{"..." if len(node_columns) > 10 else ""}'
                )
                if collect_all:
                    errors.append(error)
                else:
                    raise error
    
    if 'requires_edge_cols' in schema_effects:
        if callable(schema_effects['requires_edge_cols']):
            required_edge_cols = schema_effects['requires_edge_cols'](op.params)
        else:
            required_edge_cols = schema_effects['requires_edge_cols']
        
        for col in required_edge_cols:
            if col not in edge_columns:
                error = GFQLSchemaError(
                    ErrorCode.E301,
                    f'Call operation "{op.function}" requires edge column "{col}" which does not exist',
                    field=f'{op.function}.{col}',
                    value=col,
                    suggestion=f'Available edge columns: {", ".join(sorted(edge_columns)[:10])}{"..." if len(edge_columns) > 10 else ""}'
                )
                if collect_all:
                    errors.append(error)
                else:
                    raise error
    
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
