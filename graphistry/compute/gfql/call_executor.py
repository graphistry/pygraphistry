"""Execute validated method calls on Plottable objects.

This module handles the actual execution of safelisted methods
after parameter validation.
"""

from typing import Dict, Any
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.gfql.call_safelist import validate_call_params
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


def execute_call(g: Plottable, function: str, params: Dict[str, Any], engine: Engine) -> Plottable:
    """Execute a validated method call on a Plottable.

    Args:
        g: The graph to call the method on
        function: Name of the method to call
        params: Parameters for the method (will be validated)
        engine: Execution engine

    Returns:
        Result of the method call (usually a new Plottable)

    Raises:
        GFQLTypeError: If validation fails or method doesn't exist
        AttributeError: If method doesn't exist on Plottable
    """
    # Validate parameters against safelist
    validated_params = validate_call_params(function, params)

    # Special handling for hypergraph
    if function == 'hypergraph':
        # Hypergraph needs special handling - use nodes as raw_events
        if g._nodes is None or len(g._nodes) == 0:
            raise GFQLTypeError(
                ErrorCode.E105,
                "Hypergraph requires nodes data",
                field="nodes",
                value="None or empty",
                suggestion="Ensure graph has nodes before calling hypergraph"
            )

        # Call hypergraph with nodes as raw_events
        raw_events = g._nodes

        # Set default engine if not specified
        if 'engine' not in validated_params:
            # Try to detect engine from dataframe type
            if str(type(raw_events).__module__).startswith('cudf'):
                validated_params['engine'] = 'cudf'
            elif str(type(raw_events).__module__).startswith('dask'):
                validated_params['engine'] = 'dask'
            else:
                validated_params['engine'] = 'pandas'

        # Call hypergraph method directly (now properly typed in Plottable Protocol)
        try:
            result = g.hypergraph(raw_events, **validated_params)
            # Hypergraph returns a HypergraphResult dict with 'graph' key containing the Plottable
            return result['graph']
        except Exception as e:
            raise GFQLTypeError(
                ErrorCode.E303,
                f"Error executing hypergraph: {str(e)}",
                field="function",
                value="hypergraph"
            ) from e

    # Check if method exists on Plottable
    if not hasattr(g, function):
        raise AttributeError(
            f"Plottable has no method '{function}'. "
            f"This should not happen if safelist is properly configured."
        )

    # Get the method
    method = getattr(g, function)

    # Special handling for methods that need the engine parameter
    if function in ['materialize_nodes', 'hop']:
        # These methods accept an engine parameter
        if 'engine' not in validated_params:
            # Add current engine if not specified
            validated_params['engine'] = engine

    try:
        # Execute the method with validated parameters
        result = method(**validated_params)
        
        # Ensure result is a Plottable (most methods return self or new Plottable)
        if not isinstance(result, Plottable):
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Method '{function}' returned non-Plottable result",
                field="function",
                value=f"{type(result).__name__}",
                suggestion="Only methods that return Plottable objects are allowed"
            )
        
        return result
        
    except TypeError as e:
        # Handle parameter mismatch errors
        raise GFQLTypeError(
            ErrorCode.E201,
            f"Parameter error calling '{function}': {str(e)}",
            field="params",
            value=validated_params,
            suggestion="Check parameter names and types"
        ) from e
    except Exception as e:
        # Re-raise other exceptions with context
        raise GFQLTypeError(
            ErrorCode.E303,
            f"Error executing '{function}': {str(e)}",
            field="function",
            value=function
        ) from e
