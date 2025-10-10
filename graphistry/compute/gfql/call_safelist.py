"""Safelist of allowed methods for GFQL Call operations.

This module defines which Plottable methods can be called through GFQL
and their parameter validation rules.

Available Operations:
    Graph Transformations:
        - hypergraph: Transform event data into entity relationships

    Graph Traversals:
        - hop: Multi-hop traversal with configurable direction and depth

    Data Operations:
        - get_degrees: Calculate node degrees (in/out/total)
        - filter_edges_by_dict: Filter edges based on attribute values
        - prune_self_edges: Remove self-referencing edges
        - materialize_nodes: Compute and materialize node DataFrame

    Layout Operations:
        - layout_settings: Configure layout algorithm settings
        - tree_layout: Apply hierarchical tree layout

Usage:
    from graphistry.compute.ast import call
    from graphistry.compute.calls import hypergraph  # Typed alternative

    # Using call() with string name
    g.gfql(call('hypergraph', {'entity_types': ['user', 'product']}))

    # Using typed builder (recommended for hypergraph)
    g.gfql(hypergraph(entity_types=['user', 'product']))
"""

from typing import Dict, Any
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


def is_float(v: Any) -> bool:
    return isinstance(v, float)


def is_int_or_float(v: Any) -> bool:
    return isinstance(v, (int, float))


def is_list_of_strings(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(item, str) for item in v)


def is_dict_str_to_list_str(v: Any) -> bool:
    """Validate dict mapping strings to lists of strings."""
    if not isinstance(v, dict):
        return False
    for k, val in v.items():
        if not isinstance(k, str):
            return False
        if not is_list_of_strings(val):
            return False
    return True


def validate_hypergraph_opts(v: Any) -> bool:
    """Validate hypergraph opts parameter structure.

    Expected structure based on HyperBindings class:
    {
        'TITLE': str,           # default: 'nodeTitle'
        'DELIM': str,          # default: '::'
        'NODEID': str,         # default: 'nodeID'
        'ATTRIBID': str,       # default: 'attribID'
        'EVENTID': str,        # default: 'EventID'
        'EVENTTYPE': str,      # default: 'event'
        'SOURCE': str,         # default: 'src'
        'DESTINATION': str,    # default: 'dst'
        'CATEGORY': str,       # default: 'category'
        'NODETYPE': str,       # default: 'type'
        'EDGETYPE': str,       # default: 'edgeType'
        'NULLVAL': str,        # default: 'null'
        'SKIP': List[str],     # optional list
        'CATEGORIES': Dict[str, List[str]],  # category mappings
        'EDGES': Dict[str, List[str]]        # edge type mappings
    }
    """
    if not isinstance(v, dict):
        return False

    # Known string keys from HyperBindings
    string_keys = {
        'TITLE', 'DELIM', 'NODEID', 'ATTRIBID', 'EVENTID', 'EVENTTYPE',
        'SOURCE', 'DESTINATION', 'CATEGORY', 'NODETYPE', 'EDGETYPE', 'NULLVAL'
    }

    for key, val in v.items():
        if not isinstance(key, str):
            return False

        # Check known string parameters
        if key in string_keys:
            if not isinstance(val, str):
                return False
        # Check SKIP parameter (list of strings)
        elif key == 'SKIP':
            if not is_list_of_strings(val):
                return False
        # Check CATEGORIES and EDGES (dict of string -> list of strings)
        elif key in ('CATEGORIES', 'EDGES'):
            if not is_dict_str_to_list_str(val):
                return False
        # Unknown key - still allow but must be JSON-serializable
        else:
            # For forward compatibility, allow other keys but they should be simple types
            if not isinstance(val, (str, int, float, bool, list, dict, type(None))):
                return False

    return True


# Safelist configuration
# Dictionary mapping allowed Plottable method names to their validation rules.
#
# Each method entry contains:
#     - allowed_params (Set[str]): Parameter names that can be passed to the method
#     - required_params (Set[str]): Parameters that must be provided
#     - param_validators (Dict[str, Callable]): Maps param names to validation functions
#     - description (str): Human-readable description of what the method does
#     - schema_effects (Dict[str, List[str]]): Describes schema changes:
#         - adds_node_cols: Columns added to node DataFrame
#         - adds_edge_cols: Columns added to edge DataFrame
#         - requires_node_cols: Node columns that must exist before calling
#         - requires_edge_cols: Edge columns that must exist before calling
#
# Example entry:
#     'hop': {
#         'allowed_params': {'steps', 'to_fixed_point', 'direction'},
#         'required_params': set(),
#         'param_validators': {
#             'steps': is_int,
#             'to_fixed_point': is_bool,
#             'direction': lambda v: v in ['forward', 'reverse', 'undirected']
#         },
#         'description': 'Traverse graph edges for N steps',
#         'schema_effects': {}
#     }

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
    },

    # In/out degree methods
    'get_indegrees': {
        'allowed_params': {'col'},
        'required_params': set(),
        'param_validators': {
            'col': is_string
        },
        'description': 'Calculate node in-degrees',
        'schema_effects': {
            'adds_node_cols': lambda p: [p.get('col', 'degree_in')],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    'get_outdegrees': {
        'allowed_params': {'col'},
        'required_params': set(),
        'param_validators': {
            'col': is_string
        },
        'description': 'Calculate node out-degrees',
        'schema_effects': {
            'adds_node_cols': lambda p: [p.get('col', 'degree_out')],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    # Graph algorithm operations
    'compute_cugraph': {
        'allowed_params': {'alg', 'out_col', 'params', 'kind', 'directed', 'G'},
        'required_params': {'alg'},
        'param_validators': {
            'alg': is_string,
            'out_col': is_string_or_none,
            'params': is_dict,
            'kind': is_string,
            'directed': is_bool,
            'G': lambda x: x is None  # Allow None only
        },
        'description': 'Run cuGraph algorithms (pagerank, louvain, etc)',
        'schema_effects': {
            'adds_node_cols': lambda p: [p.get('out_col', p['alg'])],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    'compute_igraph': {
        'allowed_params': {'alg', 'out_col', 'directed', 'use_vids', 'params'},
        'required_params': {'alg'},
        'param_validators': {
            'alg': is_string,
            'out_col': is_string_or_none,
            'directed': is_bool,
            'use_vids': is_bool,
            'params': is_dict
        },
        'description': 'Run igraph algorithms'
    },

    # Layout operations
    'layout_cugraph': {
        'allowed_params': {'layout', 'params', 'kind', 'directed', 'G', 'bind_position', 'x_out_col', 'y_out_col', 'play'},
        'required_params': set(),
        'param_validators': {
            'layout': is_string,
            'params': is_dict,
            'kind': is_string,
            'directed': is_bool,
            'G': lambda x: x is None,
            'bind_position': is_bool,
            'x_out_col': is_string,
            'y_out_col': is_string,
            'play': is_int
        },
        'description': 'GPU-accelerated graph layouts'
    },

    'layout_igraph': {
        'allowed_params': {'layout', 'directed', 'use_vids', 'bind_position', 'x_out_col', 'y_out_col', 'params', 'play'},
        'required_params': {'layout'},
        'param_validators': {
            'layout': is_string,
            'directed': is_bool,
            'use_vids': is_bool,
            'bind_position': is_bool,
            'x_out_col': is_string,
            'y_out_col': is_string,
            'params': is_dict,
            'play': is_int
        },
        'description': 'igraph-based layouts'
    },

    'layout_graphviz': {
        'allowed_params': {
            'prog', 'args', 'directed', 'strict', 'graph_attr',
            'node_attr', 'edge_attr', 'x_out_col', 'y_out_col', 'bind_position'
        },
        'required_params': set(),
        'param_validators': {
            'prog': is_string,
            'args': is_string_or_none,
            'directed': is_bool,
            'strict': is_bool,
            'graph_attr': is_dict,
            'node_attr': is_dict,
            'edge_attr': is_dict,
            'x_out_col': is_string,
            'y_out_col': is_string,
            'bind_position': is_bool
        },
        'description': 'Graphviz layouts (dot, neato, etc)'
    },

    'fa2_layout': {
        'allowed_params': {'fa2_params', 'circle_layout_params', 'partition_key', 'remove_self_edges', 'engine', 'featurize'},
        'required_params': set(),
        'param_validators': {
            'fa2_params': is_dict,
            'circle_layout_params': is_dict,
            'partition_key': is_string_or_none,
            'remove_self_edges': is_bool,
            'engine': is_string,
            'featurize': is_dict
        },
        'description': 'ForceAtlas2 layout algorithm'
    },

    # Self-edge pruning
    'prune_self_edges': {
        'allowed_params': set(),
        'required_params': set(),
        'param_validators': {},
        'description': 'Remove self-loops from graph'
    },

    # Graph transformations
    'collapse': {
        'allowed_params': {'node', 'attribute', 'column', 'self_edges', 'unwrap', 'verbose'},
        'required_params': set(),
        'param_validators': {
            'node': is_string_or_none,
            'attribute': is_string_or_none,
            'column': is_string_or_none,
            'self_edges': is_bool,
            'unwrap': is_bool,
            'verbose': is_bool
        },
        'description': 'Collapse nodes by shared attribute values'
    },

    'drop_nodes': {
        'allowed_params': {'nodes'},
        'required_params': {'nodes'},
        'param_validators': {
            'nodes': lambda v: isinstance(v, list) or is_dict(v)
        },
        'description': 'Remove specified nodes and their edges'
    },

    'keep_nodes': {
        'allowed_params': {'nodes'},
        'required_params': {'nodes'},
        'param_validators': {
            'nodes': lambda v: isinstance(v, list) or is_dict(v)
        },
        'description': 'Keep only specified nodes and their edges'
    },

    # Topology analysis
    'get_topological_levels': {
        'allowed_params': {'level_col', 'allow_cycles', 'warn_cycles', 'remove_self_loops'},
        'required_params': set(),
        'param_validators': {
            'level_col': is_string,
            'allow_cycles': is_bool,
            'warn_cycles': is_bool,
            'remove_self_loops': is_bool
        },
        'description': 'Compute topological levels for DAG analysis'
    },

    # Visual encoding methods
    'encode_point_color': {
        'allowed_params': {'column', 'palette', 'as_categorical', 'as_continuous', 'categorical_mapping', 'default_mapping'},
        'required_params': {'column'},
        'param_validators': {
            'column': is_string,
            'palette': lambda v: isinstance(v, list),
            'as_categorical': is_bool,
            'as_continuous': is_bool,
            'categorical_mapping': is_dict,
            'default_mapping': is_string_or_none
        },
        'description': 'Map node column values to colors'
    },

    'encode_edge_color': {
        'allowed_params': {'column', 'palette', 'as_categorical', 'as_continuous', 'categorical_mapping', 'default_mapping'},
        'required_params': {'column'},
        'param_validators': {
            'column': is_string,
            'palette': lambda v: isinstance(v, list),
            'as_categorical': is_bool,
            'as_continuous': is_bool,
            'categorical_mapping': is_dict,
            'default_mapping': is_string_or_none
        },
        'description': 'Map edge column values to colors'
    },

    'encode_point_size': {
        'allowed_params': {'column', 'categorical_mapping', 'default_mapping'},
        'required_params': {'column'},
        'param_validators': {
            'column': is_string,
            'categorical_mapping': is_dict,
            'default_mapping': lambda v: isinstance(v, (int, float))
        },
        'description': 'Map node column values to sizes'
    },

    'encode_point_icon': {
        'allowed_params': {'column', 'categorical_mapping', 'continuous_binning', 'default_mapping', 'as_text'},
        'required_params': {'column'},
        'param_validators': {
            'column': is_string,
            'categorical_mapping': is_dict,
            'continuous_binning': lambda v: isinstance(v, list),
            'default_mapping': is_string_or_none,
            'as_text': is_bool
        },
        'description': 'Map node column values to icons'
    },

    # Metadata methods
    'name': {
        'allowed_params': {'name'},
        'required_params': {'name'},
        'param_validators': {
            'name': is_string
        },
        'description': 'Set visualization name'
    },

    'description': {
        'allowed_params': {'description'},
        'required_params': {'description'},
        'param_validators': {
            'description': is_string
        },
        'description': 'Set visualization description'
    },

    # Layout with community detection
    'group_in_a_box_layout': {
        'allowed_params': {
            'partition_alg', 'partition_params', 'layout_alg', 'layout_params',
            'x', 'y', 'w', 'h', 'encode_colors', 'colors', 'partition_key', 'engine'
        },
        'required_params': set(),
        'param_validators': {
            'partition_alg': is_string_or_none,
            'partition_params': is_dict,
            'layout_alg': lambda v: v is None or is_string(v) or callable(v),
            'layout_params': is_dict,
            'x': lambda v: isinstance(v, (int, float)),
            'y': lambda v: isinstance(v, (int, float)),
            'w': lambda v: v is None or isinstance(v, (int, float)),
            'h': lambda v: v is None or isinstance(v, (int, float)),
            'encode_colors': is_bool,
            'colors': lambda v: v is None or is_list_of_strings(v),
            'partition_key': is_string_or_none,
            'engine': lambda v: v in ['auto', 'cpu', 'gpu', 'pandas', 'cudf']
        },
        'description': 'Group-in-a-box layout with community detection'
    },

    # Hypergraph transformation
    'hypergraph': {
        'allowed_params': {
            'entity_types', 'opts', 'drop_na', 'drop_edge_attrs',
            'verbose', 'direct', 'engine', 'npartitions', 'chunksize',
            'from_edges', 'return_as'
        },
        'required_params': set(),  # All params are optional
        'param_validators': {
            'entity_types': lambda v: v is None or is_list_of_strings(v),
            'opts': validate_hypergraph_opts,  # Use detailed validator
            'drop_na': is_bool,
            'drop_edge_attrs': is_bool,
            'verbose': is_bool,
            'direct': is_bool,
            'engine': lambda v: is_string(v) and v in ['pandas', 'cudf', 'dask', 'auto'],
            'npartitions': lambda v: v is None or is_int(v),
            'chunksize': lambda v: v is None or is_int(v),
            'from_edges': is_bool,
            'return_as': lambda v: is_string(v) and v in ['graph', 'all', 'entities', 'events', 'edges', 'nodes']
        },
        'description': 'Transform event data into a hypergraph'
    },

    # UMAP embedding operations
    'umap': {
        'allowed_params': {
            'X', 'y', 'kind', 'scale', 'n_neighbors', 'min_dist', 'spread',
            'local_connectivity', 'repulsion_strength', 'negative_sample_rate',
            'n_components', 'metric', 'suffix', 'play', 'encode_position',
            'encode_weight', 'dbscan', 'engine', 'feature_engine', 'inplace',
            'memoize', 'umap_kwargs', 'umap_fit_kwargs', 'umap_transform_kwargs'
        },
        'required_params': set(),  # All params are optional
        'param_validators': {
            'X': lambda v: v is None or is_string(v) or is_list_of_strings(v),
            'y': lambda v: v is None or is_string(v) or is_list_of_strings(v),
            'kind': lambda v: v in ['nodes', 'edges'],
            'scale': is_int_or_float,
            'n_neighbors': is_int,
            'min_dist': is_int_or_float,
            'spread': is_int_or_float,
            'local_connectivity': is_int,
            'repulsion_strength': is_int_or_float,
            'negative_sample_rate': is_int,
            'n_components': is_int,
            'metric': is_string,
            'suffix': is_string,
            'play': lambda v: v is None or is_int(v),
            'encode_position': is_bool,
            'encode_weight': is_bool,
            'dbscan': is_bool,
            'engine': lambda v: v in ['auto', 'umap_learn', 'cuml'],
            'feature_engine': is_string,
            'inplace': lambda v: v is False,  # Only False allowed per type hint
            'memoize': is_bool,
            'umap_kwargs': is_dict,
            'umap_fit_kwargs': is_dict,
            'umap_transform_kwargs': is_dict
        },
        'description': 'UMAP dimensionality reduction for graph embeddings'
    }
}


def validate_call_params(function: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Validate parameters for a GFQL Call operation against the safelist.
    
    Performs comprehensive validation:
        1. Checks if function is in the safelist
        2. Verifies all required parameters are present
        3. Ensures no unknown parameters are passed
        4. Validates parameter types using configured validators
        5. Returns the validated parameters unchanged
    
    Args:
        function: Name of the Plottable method to call
        params: Dictionary of parameters to validate
    
    Returns:
        The same parameters dict if validation passes
    
    Raises:
        GFQLTypeError: If function not in safelist (E303)
        GFQLTypeError: If required parameters missing (E105)
        GFQLTypeError: If unknown parameters provided (E303)
        GFQLTypeError: If parameter type validation fails (E201)
    
    **Example::**
    
        # Valid call
        params = validate_call_params('hop', {'steps': 2, 'direction': 'forward'})
        
        # Invalid - unknown function
        validate_call_params('dangerous_method', {})  # Raises E303
        
        # Invalid - missing required param
        validate_call_params('fa2_layout', {})  # Would raise E105 if layout was required
        
        # Invalid - wrong type
        validate_call_params('hop', {'steps': 'two'})  # Raises E201
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
