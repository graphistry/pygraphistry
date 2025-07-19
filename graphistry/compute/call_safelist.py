"""Safelist of allowed methods for GFQL Call operations.

This module defines which Plottable methods can be called through GFQL
and their parameter validation rules.
"""

from typing import Dict, Any, Set, Optional, Union, Type
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError


# Type validators
def is_string(v: Any) -> bool:
    return isinstance(v, str)


def is_int(v: Any) -> bool:
    return isinstance(v, int)


def is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def is_dict(v: Any) -> bool:
    return isinstance(v, dict)


def is_string_or_none(v: Any) -> bool:
    return v is None or isinstance(v, str)


def is_list_of_strings(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(item, str) for item in v)


# Safelist configuration
# Each entry defines:
# - allowed_params: Set of parameter names that can be passed
# - required_params: Set of parameters that must be provided
# - param_validators: Dict of param_name -> validator function
# - description: Human-readable description of what the method does

SAFELIST_V1: Dict[str, Dict[str, Any]] = {
    'get_degrees': {
        'allowed_params': {'col_in', 'col_out', 'col', 'engine'},
        'required_params': set(),
        'param_validators': {
            'col_in': is_string,
            'col_out': is_string,
            'col': is_string,
            'engine': is_string
        },
        'description': 'Calculate node degrees'
    },
    
    'filter_nodes_by_dict': {
        'allowed_params': {'filter_dict'},
        'required_params': {'filter_dict'},
        'param_validators': {
            'filter_dict': is_dict
        },
        'description': 'Filter nodes by attribute values'
    },
    
    'filter_edges_by_dict': {
        'allowed_params': {'filter_dict'},
        'required_params': {'filter_dict'},
        'param_validators': {
            'filter_dict': is_dict
        },
        'description': 'Filter edges by attribute values'
    },
    
    'materialize_nodes': {
        'allowed_params': {'engine', 'reuse'},
        'required_params': set(),
        'param_validators': {
            'engine': is_string,
            'reuse': is_bool
        },
        'description': 'Generate node table from edges'
    },
    
    'hop': {
        'allowed_params': {
            'nodes', 'hops', 'to_fixed_point', 'direction',
            'source_node_match', 'edge_match', 'destination_node_match',
            'source_node_query', 'edge_query', 'destination_node_query',
            'return_as_wave_front', 'target_wave_front', 'engine'
        },
        'required_params': set(),
        'param_validators': {
            'hops': is_int,
            'to_fixed_point': is_bool,
            'direction': lambda v: v in ['forward', 'reverse', 'undirected'],
            'source_node_match': is_dict,
            'edge_match': is_dict,
            'destination_node_match': is_dict,
            'source_node_query': is_string,
            'edge_query': is_string,
            'destination_node_query': is_string,
            'return_as_wave_front': is_bool,
            'engine': is_string
        },
        'description': 'Traverse graph by following edges'
    }
}


def validate_call_params(function: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters for a function call.
    
    Args:
        function: Name of the function to call
        params: Parameters to validate
        
    Returns:
        Validated parameters (may be modified, e.g., defaults added)
        
    Raises:
        GFQLTypeError: If validation fails
    """
    # Check if function is in safelist
    if function not in SAFELIST_V1:
        raise GFQLTypeError(
            ErrorCode.E303,
            f"Function '{function}' is not in the safelist",
            field="function",
            value=function,
            suggestion=f"Available functions: {', '.join(sorted(SAFELIST_V1.keys()))}"
        )
    
    config = SAFELIST_V1[function]
    allowed_params = config['allowed_params']
    required_params = config['required_params']
    param_validators = config['param_validators']
    
    # Check for required parameters
    missing_required = required_params - set(params.keys())
    if missing_required:
        raise GFQLTypeError(
            ErrorCode.E105,
            f"Missing required parameters for '{function}'",
            field="params",
            value=list(missing_required),
            suggestion=f"Required parameters: {', '.join(sorted(missing_required))}"
        )
    
    # Check for unknown parameters
    unknown_params = set(params.keys()) - allowed_params
    if unknown_params:
        raise GFQLTypeError(
            ErrorCode.E303,
            f"Unknown parameters for '{function}'",
            field="params",
            value=list(unknown_params),
            suggestion=f"Allowed parameters: {', '.join(sorted(allowed_params))}"
        )
    
    # Validate parameter types
    for param_name, param_value in params.items():
        if param_name in param_validators:
            validator = param_validators[param_name]
            if not validator(param_value):
                raise GFQLTypeError(
                    ErrorCode.E201,
                    f"Invalid type for parameter '{param_name}' in '{function}'",
                    field=f"params.{param_name}",
                    value=f"{type(param_value).__name__}: {param_value}",
                    suggestion="Check the parameter type requirements"
                )
    
    return params