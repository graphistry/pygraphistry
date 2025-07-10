"""Enhanced chain function with validation support."""

from typing import Union, List, Optional
from graphistry.Plottable import Plottable
from graphistry.compute.chain import chain as chain_original, Chain
from graphistry.compute.ast import ASTObject
from graphistry.compute.gfql.validate import (
    validate_query, extract_schema, format_validation_errors,
    ValidationIssue, Schema
)
from graphistry.compute.gfql.exceptions import GFQLValidationError
from graphistry.Engine import EngineAbstract
import logging

logger = logging.getLogger(__name__)


def chain_with_validation(
    self: Plottable, 
    ops: Union[List[ASTObject], Chain], 
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    validate: bool = True,
    validate_mode: str = 'warn',  # 'warn', 'error', or 'silent'
    validate_schema: bool = True
) -> Plottable:
    """
    Chain operations with optional validation.
    
    This is a wrapper around the original chain function that adds validation support.
    
    Args:
        self: Plottable instance
        ops: List of operations or Chain object
        engine: Engine to use
        validate: Whether to perform validation
        validate_mode: How to handle validation issues -
            'warn' (Log warnings but continue - default),
            'error' (Raise exception on first error),
            'silent' (Collect issues but don't log/raise)
        validate_schema: Whether to validate against data schema if available
        
    Returns:
        Plottable result
        
    Raises:
        GFQLValidationError: If validate_mode='error' and validation fails
    """
    if not validate:
        return chain_original(self, ops, engine)
    
    # Perform validation
    if validate_schema and (self._nodes is not None or self._edges is not None):
        # Validate with schema
        issues = validate_query(ops, self._nodes, self._edges)
    else:
        # Syntax validation only
        from graphistry.compute.gfql.validate import validate_syntax
        issues = validate_syntax(ops)
    
    # Handle validation results based on mode
    if issues:
        errors = [i for i in issues if i.level == 'error']
        warnings = [i for i in issues if i.level == 'warning']
        
        if validate_mode == 'error' and errors:
            # Raise on first error
            error_msg = format_validation_errors(errors[:1])
            raise GFQLValidationError(error_msg)
        
        elif validate_mode == 'warn':
            # Log all issues
            if errors:
                logger.error("GFQL Validation Errors:\n%s", format_validation_errors(errors))
            if warnings:
                logger.warning("GFQL Validation Warnings:\n%s", format_validation_errors(warnings))
        
        # For 'silent' mode, issues are available but not logged
    
    # Store validation results for access
    if hasattr(self, '_last_validation_issues'):
        self._last_validation_issues = issues
    
    # Execute the chain
    return chain_original(self, ops, engine)


def validate_chain(
    self: Plottable,
    ops: Union[List[ASTObject], Chain],
    return_issues: bool = False
) -> Union[bool, List[ValidationIssue]]:
    """
    Validate a chain without executing it.
    
    Args:
        self: Plottable instance 
        ops: Operations to validate
        return_issues: If True, return list of issues; if False, return bool
        
    Returns:
        If return_issues=False: True if valid, False otherwise
        If return_issues=True: List of ValidationIssue objects
    """
    if self._nodes is not None or self._edges is not None:
        issues = validate_query(ops, self._nodes, self._edges) 
    else:
        from graphistry.compute.gfql.validate import validate_syntax
        issues = validate_syntax(ops)
    
    if return_issues:
        return issues
    else:
        errors = [i for i in issues if i.level == 'error']
        return len(errors) == 0


def get_chain_schema(self: Plottable) -> "Schema":
    """
    Extract schema from Plottable for validation purposes.
    
    Args:
        self: Plottable instance
        
    Returns:
        Schema object with column information
    """
    return extract_schema(self)
