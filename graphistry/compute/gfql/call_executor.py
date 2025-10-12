"""Execute validated method calls on Plottable objects.

This module handles the actual execution of safelisted methods
after parameter validation.
"""

import threading
from typing import Dict, Any
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.gfql.call_safelist import validate_call_params
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

# Thread-local storage for policy context
_thread_local = threading.local()


def execute_call(g: Plottable, function: str, params: Dict[str, Any], engine: Engine, policy=None) -> Plottable:
    """Execute a validated method call on a Plottable.

    Args:
        g: The graph to call the method on
        function: Name of the method to call
        params: Parameters for the method (will be validated)
        engine: Execution engine
        policy: Optional policy dict, or will use thread-local if available

    Returns:
        Result of the method call (usually a new Plottable)

    Raises:
        GFQLTypeError: If validation fails or method doesn't exist
        AttributeError: If method doesn't exist on Plottable
    """
    # Get policy from thread-local if not passed
    if policy is None:
        policy = getattr(_thread_local, 'policy', None)

    # Precall policy phase - before executing call operation
    final_params = params

    import time

    if policy and 'precall' in policy:
        from graphistry.compute.gfql.policy import PolicyContext, PolicyException
        from graphistry.compute.gfql.policy.stats import extract_graph_stats

        stats = extract_graph_stats(g)
        context: PolicyContext = {
            'phase': 'precall',
            'hook': 'precall',
            'query': None,  # Not available in call context
            'current_ast': None,  # Calls don't have AST object
            'call_op': function,
            'call_params': params,
            'plottable': g,  # INPUT graph
            'graph_stats': stats,  # INPUT stats
            '_policy_depth': 0
        }

        try:
            # Policy can only accept (None) or deny (exception)
            policy['precall'](context)

        except PolicyException as e:
            # Enrich exception with context if not already set
            if e.query_type is None:
                e.query_type = 'call'
            if e.data_size is None:
                e.data_size = stats
            raise

    # Initialize variables for finally block
    result = None
    error = None
    success = False
    execution_time = 0.0
    start_time = time.perf_counter()
    validated_params = None

    try:
        # Validate parameters against safelist (inside try block so postcall fires on validation errors)
        validated_params = validate_call_params(function, final_params)

        # Check if method exists on Plottable
        if not hasattr(g, function):
            raise AttributeError(
                f"Plottable has no method '{function}'. "
                f"This should not happen if safelist is properly configured."
            )

        # Get the method
        method = getattr(g, function)

        # Execute the method with validated parameters
        result = method(**validated_params)

        # Calculate execution time
        execution_time = time.perf_counter() - start_time

        # Ensure result is a Plottable (most methods return self or new Plottable)
        # Exception: hypergraph can return DataFrame when return_as != 'graph'
        if not isinstance(result, Plottable) and function != 'hypergraph':
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Method '{function}' returned non-Plottable result",
                field="function",
                value=f"{type(result).__name__}",
                suggestion="Only methods that return Plottable objects are allowed"
            )

        # Mark as successful
        success = True

    except Exception as e:
        # Calculate execution time even on error
        execution_time = time.perf_counter() - start_time

        # Capture error for postcall hook
        error = e
        # Don't re-raise yet - let finally block run first

    finally:
        # Postcall policy phase - ALWAYS fires (even on error)
        policy_error = None
        if policy and 'postcall' in policy:
            from graphistry.compute.gfql.policy import PolicyContext, PolicyException
            from graphistry.compute.gfql.policy.stats import extract_graph_stats

            # Extract stats from result (if success) or input graph (if error)
            graph_for_stats = result if success else g

            # Only extract stats if result is a Plottable (hypergraph can return DataFrame)
            # Avoids Jinja2 error when extract_graph_stats() accesses df.style on DataFrames
            result_stats = extract_graph_stats(graph_for_stats) if isinstance(graph_for_stats, Plottable) else {}
            postcall_context: PolicyContext = {
                'phase': 'postcall',
                'hook': 'postcall',
                'query': None,  # Not available in call context
                'current_ast': None,  # Calls don't have AST object
                'call_op': function,
                'call_params': params,  # Original parameters for reference
                'plottable': graph_for_stats,  # RESULT graph (if success) or INPUT graph (if error)
                'graph_stats': result_stats,
                'execution_time': execution_time,
                'success': success,  # True if successful, False if error
                '_policy_depth': 0
            }

            # Add error information if execution failed
            if error is not None:
                postcall_context['error'] = str(error)  # type: ignore
                postcall_context['error_type'] = type(error).__name__  # type: ignore

            try:
                # Policy can only accept (None) or deny (exception)
                policy['postcall'](postcall_context)

            except PolicyException as e:
                # Enrich exception with context if not already set
                if e.query_type is None:
                    e.query_type = 'call'
                if e.data_size is None:
                    e.data_size = result_stats
                # Capture policy error instead of raising immediately
                policy_error = e

    # After finally block, decide which error to raise
    # Priority: PolicyException > operation error
    if policy_error is not None:
        # Policy denied - chain from operation error if one exists
        if error is not None:
            raise policy_error from error
        else:
            raise policy_error
    elif error is not None:
        # Wrap the error with context
        if isinstance(error, TypeError):
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Parameter error calling '{function}': {str(error)}",
                field="params",
                value=validated_params if validated_params is not None else params,
                suggestion="Check parameter names and types"
            ) from error
        elif isinstance(error, GFQLTypeError):
            # Already a GFQLTypeError (e.g., from validation), just re-raise
            raise error
        else:
            raise GFQLTypeError(
                ErrorCode.E303,
                f"Error executing '{function}': {str(error)}",
                field="function",
                value=function
            ) from error

    return result
