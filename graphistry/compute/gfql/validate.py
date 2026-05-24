"""GFQL query validation utilities for syntax and schema checking.

.. deprecated:: 0.34.0
   This module is deprecated. GFQL now has built-in validation.
   See :doc:`/gfql/validation_migration_guide` for migration instructions.
   
   Instead of::
   
       from graphistry.compute.gfql.validate import validate_syntax
       issues = validate_syntax(query)
       
   Use::
   
       from graphistry.compute.chain import Chain
       try:
           chain = Chain(query)  # Automatic validation
       except GFQLValidationError as e:
           print(f"[{e.code}] {e.message}")
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Union, Any, Tuple, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.chain import Chain
    from graphistry.compute.ast import ASTObject, ASTNode, ASTEdge

from graphistry.compute.predicates.ASTPredicate import ASTPredicate
from graphistry.compute.predicates.numeric import NumericASTPredicate
from graphistry.compute.predicates.str import (
    Contains, Startswith, Endswith, Match, Fullmatch,
    IsNumeric, IsAlpha, IsDigit, IsLower, IsUpper,
    IsSpace, IsAlnum, IsDecimal, IsTitle
)
from graphistry.compute.predicates.temporal import (
    IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
    IsYearStart, IsYearEnd, IsLeapYear
)

# NOTE: This module is deprecated but still used internally for backwards compatibility.
# We don't emit warnings on import since that would spam users who aren't directly
# importing this module. Warnings are only shown in the docstring.

@dataclass(eq=False)
class ValidationIssue:
    """Represents a validation issue (error or warning)."""

    level: str  # 'error' or 'warning'
    message: str
    operation_index: Optional[int] = None
    field: Optional[str] = None
    suggestion: Optional[str] = None
    error_type: Optional[str] = None

    def __repr__(self) -> str:
        parts = [f"{self.level.upper()}: {self.message}"]
        if self.operation_index is not None:
            parts.append(f"at operation {self.operation_index}")
        if self.field:
            parts.append(f"field: {self.field}")
        if self.suggestion:
            parts.append(f"Suggestion: {self.suggestion}")
        return " | ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'level': self.level,
            'message': self.message,
            'operation_index': self.operation_index,
            'field': self.field,
            'suggestion': self.suggestion,
            'error_type': self.error_type
        }


class Schema:
    """Represents the schema of node and edge dataframes."""

    def __init__(self,
                 node_columns: Optional[Dict[str, str]] = None,
                 edge_columns: Optional[Dict[str, str]] = None):
        self.node_columns = node_columns or {}
        self.edge_columns = edge_columns or {}

    def __repr__(self) -> str:
        return (f"Schema(nodes={list(self.node_columns.keys())}, "
                f"edges={list(self.edge_columns.keys())})")


# Error message templates
ERROR_MESSAGES = {
    'INVALID_CHAIN_TYPE': (
        'Chain must be a list of operations',
        'Wrap your operations in a list: [n(), e(), n()]'),
    'INVALID_OPERATION': (
        'Operation at index {index} is not a valid GFQL operation',
        'Use n() for nodes, e()/e_forward()/e_reverse() for edges'),
    'INVALID_FILTER_KEY': (
        'Invalid filter key format: {key}',
        'Filter keys must be strings'),
    'INVALID_HOPS': (
        'Hops must be a positive integer or None',
        'Use hops=2 for specific count, or to_fixed_point=True for unbounded'),
    'COLUMN_NOT_FOUND': (
        'Column "{column}" not found in {table} data',
        'Available columns: {available}'),
    'TYPE_MISMATCH': (
        'Column "{column}" is {actual_type} but predicate expects {expected_type}',
        'Use appropriate predicate for {actual_type} columns'),
    'INVALID_PREDICATE': (
        'Predicate {predicate} cannot be used with {column_type} columns',
        'Use {suggested_predicates} for {column_type} columns'),
    'ORPHANED_EDGE': (
        'Edge operation at index {index} not connected to nodes',
        'Add node operations before/after edge: n() -> e() -> n()'),
    'UNBOUNDED_HOPS_WARNING': (
        'Unbounded hops (to_fixed_point=True) may be slow on large graphs',
        'Consider using specific hop count for better performance'),
    'EMPTY_CHAIN': (
        'Chain is empty',
        'Add at least one operation: [n()]'),
    'INVALID_EDGE_DIRECTION': (
        'Invalid edge direction: {direction}',
        'Use "forward", "reverse", or "undirected"'),
}


def _format_error(error_key: str, **kwargs) -> Tuple[str, str]:
    """Format error message with context."""
    message_template, suggestion_template = ERROR_MESSAGES.get(
        error_key,
        (f'Unknown error: {error_key}', 'Check documentation for valid syntax'))
    message = message_template.format(**kwargs)
    suggestion = suggestion_template.format(**kwargs)
    return message, suggestion


def _append_issue(
    issues: List[ValidationIssue],
    level: str,
    error_key: str,
    *,
    operation_index: Optional[int] = None,
    field: Optional[str] = None,
    **kwargs: Any
) -> None:
    message, suggestion = _format_error(error_key, **kwargs)
    issues.append(ValidationIssue(
        level, message, operation_index=operation_index, field=field,
        suggestion=suggestion, error_type=error_key))


def _validate_filter_keys(
    issues: List[ValidationIssue],
    filter_dict: Optional[Dict[Any, Any]],
    op_index: int,
    field_prefix: str = "",
) -> None:
    if not filter_dict:
        return
    for key in filter_dict:
        if not isinstance(key, str):
            _append_issue(
                issues, 'error', 'INVALID_FILTER_KEY',
                operation_index=op_index,
                field=f"{field_prefix}{key}" if field_prefix else str(key),
                key=key)


def _available_columns(schema_columns: Dict[str, str]) -> str:
    available = list(schema_columns.keys())[:5]
    if len(schema_columns) > 5:
        available.append('...')
    return ', '.join(available)


def validate_syntax(chain: Union["Chain", List]) -> List[ValidationIssue]:
    """
    Validate GFQL query syntax without requiring data.

    Args:
        chain: GFQL chain or list of operations

    Returns:
        List of validation issues (errors and warnings)
    """
    issues: List[ValidationIssue] = []

    # Import here to avoid circular import with ast.py
    from graphistry.compute.chain import Chain
    from graphistry.compute.ast import ASTNode, ASTEdge, ASTObject

    # Convert to list if Chain object
    if isinstance(chain, Chain):
        operations = chain.chain
    elif isinstance(chain, list):
        operations = chain
    else:
        _append_issue(issues, 'error', 'INVALID_CHAIN_TYPE')
        return issues

    # Check empty chain
    if not operations:
        _append_issue(issues, 'error', 'EMPTY_CHAIN')
        return issues

    # Validate each operation
    for i, op in enumerate(operations):
        # Check if valid operation type
        if not isinstance(op, ASTObject):
            _append_issue(
                issues, 'error', 'INVALID_OPERATION',
                operation_index=i, index=i)
            continue

        # Validate nodes
        if isinstance(op, ASTNode):
            _validate_filter_keys(issues, op.filter_dict, i)

        # Validate edges
        elif isinstance(op, ASTEdge):
            # Check hops
            if (op.hops is not None
                    and (not isinstance(op.hops, int) or op.hops < 1)):
                _append_issue(
                    issues, 'error', 'INVALID_HOPS',
                    operation_index=i, field='hops')

            # Check unbounded hops warning
            if op.to_fixed_point:
                _append_issue(
                    issues, 'warning', 'UNBOUNDED_HOPS_WARNING',
                    operation_index=i)

            # Check edge filters
            for filter_name, filter_dict in [
                ('edge_match', op.edge_match),
                ('source_node_match', op.source_node_match),
                ('destination_node_match', op.destination_node_match)
            ]:
                _validate_filter_keys(
                    issues, filter_dict, i, field_prefix=f"{filter_name}.")

    # Check semantic issues
    issues.extend(_validate_semantics(operations))

    return issues


def _validate_semantics(operations: List["ASTObject"]) -> List[ValidationIssue]:
    """Validate semantic correctness of operation sequence."""
    issues: List[ValidationIssue] = []

    # Import here to avoid circular import with ast.py
    from graphistry.compute.ast import ASTEdge

    # Check for orphaned edges (edges not between nodes)
    for i, op in enumerate(operations):
        if isinstance(op, ASTEdge):
            # Check if first or last operation
            if i == 0 or i == len(operations) - 1:
                # Edge at boundary - likely orphaned
                _append_issue(
                    issues, 'warning', 'ORPHANED_EDGE',
                    operation_index=i, index=i)
            # Check if between two edges
            elif (i > 0 and isinstance(operations[i - 1], ASTEdge)
                  and i < len(operations) - 1
                  and isinstance(operations[i + 1], ASTEdge)):
                _append_issue(
                    issues, 'warning', 'ORPHANED_EDGE',
                    operation_index=i, index=i)

    return issues


def validate_schema(chain: Union["Chain", List],
                    schema: Schema) -> List[ValidationIssue]:
    """
    Validate query against data schema.

    Args:
        chain: GFQL chain or list of operations
        schema: Schema object with column information

    Returns:
        List of validation issues
    """
    issues = []

    # First do syntax validation
    syntax_issues = validate_syntax(chain)
    # Only keep errors from syntax validation for schema validation
    issues.extend([issue for issue in syntax_issues if issue.level == 'error'])

    if any(issue.level == 'error' for issue in issues):
        return issues  # Don't do schema validation if syntax errors

    # Import here to avoid circular import with ast.py
    from graphistry.compute.chain import Chain
    from graphistry.compute.ast import ASTNode, ASTEdge

    # Convert to list if Chain object
    operations = chain.chain if isinstance(chain, Chain) else chain

    # Validate each operation against schema
    for i, op in enumerate(operations):
        if isinstance(op, ASTNode):
            issues.extend(_validate_node_schema(op, schema, i))
        elif isinstance(op, ASTEdge):
            issues.extend(_validate_edge_schema(op, schema, i))

    return issues


def _validate_node_schema(node: "ASTNode", schema: Schema,
                          op_index: int) -> List[ValidationIssue]:
    """Validate node operation against schema."""
    return _validate_filter_schema(
        node.filter_dict, schema.node_columns, 'node', op_index)


def _validate_edge_schema(edge: "ASTEdge", schema: Schema,
                          op_index: int) -> List[ValidationIssue]:
    """Validate edge operation against schema."""
    issues = []

    issues.extend(_validate_filter_schema(
        edge.edge_match, schema.edge_columns, 'edge', op_index,
        field_prefix="edge_match."))

    # Validate source/dest node filters against node schema
    for filter_name, filter_dict in [
        ('source_node_match', edge.source_node_match),
        ('destination_node_match', edge.destination_node_match)
    ]:
        issues.extend(_validate_filter_schema(
            filter_dict, schema.node_columns, 'node', op_index,
            field_prefix=f"{filter_name}."))

    return issues


def _validate_filter_schema(
    filter_dict: Optional[Dict[str, Any]],
    schema_columns: Dict[str, str],
    table: str,
    op_index: int,
    field_prefix: str = "",
) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not filter_dict or not schema_columns:
        return issues

    for col, predicate in filter_dict.items():
        if col not in schema_columns:
            _append_issue(
                issues, 'error', 'COLUMN_NOT_FOUND',
                operation_index=op_index,
                field=f"{field_prefix}{col}",
                column=col,
                table=table,
                available=_available_columns(schema_columns))
            continue

        if isinstance(predicate, ASTPredicate):
            issues.extend(_validate_predicate_type(
                predicate, col, schema_columns[col], op_index,
                field_prefix=field_prefix))

    return issues


def _validate_predicate_type(predicate: ASTPredicate, column: str,
                             column_type: str, op_index: int,
                             field_prefix: str = "") -> List[ValidationIssue]:
    """Validate predicate is appropriate for column type."""
    issues = []

    # Map pandas/numpy dtypes to categories
    type_category = _get_type_category(column_type)

    # Define string predicate types
    STRING_PREDICATES = (
        Contains, Startswith, Endswith, Match, Fullmatch,
        IsNumeric, IsAlpha, IsDigit, IsLower, IsUpper,
        IsSpace, IsAlnum, IsDecimal, IsTitle
    )

    # Define temporal predicate types
    TEMPORAL_PREDICATES = (
        IsMonthStart, IsMonthEnd, IsQuarterStart, IsQuarterEnd,
        IsYearStart, IsYearEnd, IsLeapYear
    )

    # Check predicate compatibility
    if (isinstance(predicate, NumericASTPredicate)
            and type_category not in ['numeric', 'temporal']):
        message, suggestion = _format_error(
            'TYPE_MISMATCH',
            column=column,
            actual_type=column_type,
            expected_type='numeric')
        issues.append(ValidationIssue(
            'error', message, operation_index=op_index,
            field=f"{field_prefix}{column}",
            suggestion=suggestion,
            error_type='TYPE_MISMATCH'))

    elif (isinstance(predicate, STRING_PREDICATES)
          and type_category != 'string'):
        message, suggestion = _format_error(
            'TYPE_MISMATCH',
            column=column,
            actual_type=column_type,
            expected_type='string')
        issues.append(ValidationIssue(
            'error', message, operation_index=op_index,
            field=f"{field_prefix}{column}",
            suggestion=suggestion,
            error_type='TYPE_MISMATCH'))

    elif (isinstance(predicate, TEMPORAL_PREDICATES)
          and type_category != 'temporal'):
        message, suggestion = _format_error(
            'TYPE_MISMATCH',
            column=column,
            actual_type=column_type,
            expected_type='datetime')
        issues.append(ValidationIssue(
            'error', message, operation_index=op_index,
            field=f"{field_prefix}{column}",
            suggestion=suggestion,
            error_type='TYPE_MISMATCH'))

    return issues


def _get_type_category(dtype_str: str) -> str:
    """Categorize dtype string into broad categories."""
    dtype_lower = str(dtype_str).lower()
    for category, markers in [
        ('numeric', ('int', 'float', 'double', 'numeric', 'decimal')),
        ('string', ('str', 'object', 'char', 'text', 'varchar')),
        ('temporal', ('date', 'time', 'timestamp', 'datetime')),
        ('boolean', ('bool',)),
    ]:
        if any(marker in dtype_lower for marker in markers):
            return category
    return 'unknown'


def validate_query(chain: Union["Chain", List],
                   nodes_df: Optional[pd.DataFrame] = None,
                   edges_df: Optional[pd.DataFrame] = None
                   ) -> List[ValidationIssue]:
    """
    Combined syntax and schema validation.

    Args:
        chain: GFQL chain or list of operations
        nodes_df: Optional node dataframe for schema validation
        edges_df: Optional edge dataframe for schema validation

    Returns:
        List of validation issues
    """
    # Always do syntax validation
    issues = validate_syntax(chain)

    # If data provided, also do schema validation
    if nodes_df is not None or edges_df is not None:
        schema = extract_schema_from_dataframes(nodes_df, edges_df)
        schema_issues = validate_schema(chain, schema)
        # Merge issues, avoiding duplicates
        existing_errors = {(i.error_type, i.operation_index, i.field)
                           for i in issues}
        for issue in schema_issues:
            if ((issue.error_type, issue.operation_index, issue.field)
                    not in existing_errors):
                issues.append(issue)

    return issues


def extract_schema(g: "Plottable") -> Schema:
    """
    Extract schema from a Plottable object.

    Args:
        g: Plottable object with node/edge data

    Returns:
        Schema object
    """

    nodes_df = g._nodes if hasattr(g, '_nodes') else None
    edges_df = g._edges if hasattr(g, '_edges') else None

    return extract_schema_from_dataframes(nodes_df, edges_df)


def extract_schema_from_dataframes(
        nodes_df: Optional[pd.DataFrame] = None,
        edges_df: Optional[pd.DataFrame] = None) -> Schema:
    """
    Extract schema from pandas DataFrames.

    Args:
        nodes_df: Optional node dataframe
        edges_df: Optional edge dataframe

    Returns:
        Schema object with column names and types
    """
    node_columns = {}
    edge_columns = {}

    if nodes_df is not None and hasattr(nodes_df, 'dtypes'):
        node_columns = {str(col): str(dtype)
                        for col, dtype in nodes_df.dtypes.items()}

    if edges_df is not None and hasattr(edges_df, 'dtypes'):
        edge_columns = {str(col): str(dtype)
                        for col, dtype in edges_df.dtypes.items()}

    return Schema(node_columns, edge_columns)


def format_validation_errors(issues: List[ValidationIssue]) -> str:
    """
    Format validation errors for human/LLM consumption.

    Args:
        issues: List of validation issues

    Returns:
        Formatted error string
    """
    if not issues:
        return "No validation issues found."

    lines = ["GFQL Validation Report:"]
    lines.append("-" * 50)

    errors = [i for i in issues if i.level == 'error']
    warning_issues = [i for i in issues if i.level == 'warning']

    for title, group, include_field in [
        ("ERRORS", errors, True),
        ("WARNINGS", warning_issues, False),
    ]:
        if not group:
            continue
        lines.append(f"\n{title} ({len(group)}):")
        for i, issue in enumerate(group, 1):
            lines.append(f"\n{i}. {issue.message}")
            if issue.operation_index is not None:
                lines.append(f"   Location: Operation {issue.operation_index}")
            if include_field and issue.field:
                lines.append(f"   Field: {issue.field}")
            if issue.suggestion:
                lines.append(f"   💡 {issue.suggestion}")

    return "\n".join(lines)


def suggest_fixes(chain: Union["Chain", List],
                  issues: List[ValidationIssue]) -> List[str]:
    """
    Generate fix suggestions for validation issues.

    Args:
        chain: The problematic chain
        issues: Validation issues found

    Returns:
        List of suggested fixes
    """
    suggestions = []

    # Group by error type for consolidated suggestions
    error_types: Dict[str, List[ValidationIssue]] = {}
    for issue in issues:
        if issue.error_type:
            error_types.setdefault(issue.error_type, []).append(issue)

    # Generate type-specific suggestions
    if 'COLUMN_NOT_FOUND' in error_types:
        missing_cols = {issue.field for issue in
                        error_types['COLUMN_NOT_FOUND'] if issue.field}
        suggestions.append(
            f"Missing columns: {', '.join(missing_cols)}. "
            f"Check column names match your data.")

    if 'TYPE_MISMATCH' in error_types:
        suggestions.append(
            "Type mismatches found. Use numeric predicates (gt, lt) "
            "for numbers, string predicates (contains, startswith) for text.")

    if 'ORPHANED_EDGE' in error_types:
        suggestions.append(
            "Edge operations should connect nodes. "
            "Use pattern: n() -> e() -> n()")

    # Add general suggestions from issues
    for issue in issues:
        if issue.suggestion and issue.suggestion not in suggestions:
            suggestions.append(issue.suggestion)

    return suggestions
