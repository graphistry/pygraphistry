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

    # Call policy phase - before executing call operation
    final_params = params
    final_engine = engine

    if policy and 'call' in policy:
        from graphistry.compute.gfql.policy import PolicyContext, PolicyException, validate_modification
        from graphistry.compute.gfql.policy.stats import extract_graph_stats

        stats = extract_graph_stats(g)
        context: PolicyContext = {
            'phase': 'call',
            'call_op': function,
            'call_params': params,
            'plottable': g,
            'graph_stats': stats,
            '_policy_depth': 0
        }

        try:
            mods = policy['call'](context)
            if mods is not None:
                # Validate modifications
                validated_mods = validate_modification(mods, 'call')

                # Apply engine modification if present
                if 'engine' in validated_mods:
                    eng_str = validated_mods['engine']
                    # Use standard engine resolution
                    from graphistry.Engine import resolve_engine, EngineAbstract
                    final_engine = resolve_engine(EngineAbstract(eng_str), g)

                # Apply parameter modifications if present
                if 'params' in validated_mods and validated_mods['params'] is not None:
                    # Merge parameters - modifications override originals
                    final_params = {**params, **validated_mods['params']}

        except PolicyException as e:
            # Enrich exception with context if not already set
            if e.query_type is None:
                e.query_type = 'call'
            if e.data_size is None:
                e.data_size = stats
            raise

    # Special handling for methods that need the engine parameter
    if function in ['materialize_nodes', 'hop']:
        # These methods accept an engine parameter
        if 'engine' not in final_params:
            # Add current engine if not specified
            # Convert Engine enum to string for validation
            from graphistry.Engine import Engine
            if isinstance(final_engine, Engine):
                final_params['engine'] = final_engine.value
            else:
                final_params['engine'] = str(final_engine)

    # Validate parameters against safelist
    validated_params = validate_call_params(function, final_params)

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
