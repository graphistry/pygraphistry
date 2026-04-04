"""GFQL call safelist and parameter validators."""

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set, Tuple

from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.compute.gfql.call.support import (
    EDGE_COLUMN_SCHEMA_EFFECTS,
    NODE_COLUMN_SCHEMA_EFFECTS,
    NO_SCHEMA_EFFECTS,
    XY_NODE_SCHEMA_EFFECTS,
    XY_OUT_COL_SCHEMA_EFFECTS,
    _group_by_added_node_cols,
    _hypergraph_edge_adds,
    _hypergraph_input_required_cols,
    _hypergraph_node_adds,
    _is_json_compatible_value,
    _method_entry,
    _projection_row_entry,
    _rows_requires_edge_cols,
    _rows_requires_node_cols,
    _schema_effects,
    _select_added_node_cols,
    is_projection_items,
    _umap_edge_adds,
    _umap_edge_required_cols,
    _umap_node_adds,
    _umap_node_required_cols,
    _unwind_added_node_cols,
    is_bool,
    is_dict,
    is_int,
    is_int_or_float,
    is_int_or_none,
    is_list,
    is_list_of_agg_specs,
    is_list_of_strings,
    is_list_or_dict,
    is_non_empty_list_of_strings,
    is_non_empty_string,
    is_non_negative_int_like,
    is_string,
    is_string_or_none,
    is_unwind_expr,
    validate_hypergraph_opts,
)
from graphistry.compute.gfql.row.order_expr import (
    is_order_aggregate_alias_ast,
    order_expr_ast_static_supported,
)

if TYPE_CHECKING:
    from graphistry.compute.gfql.expr_parser import ExprNode

WhereRowsParseFn = Callable[[str], "ExprNode"]
WhereRowsCapabilityFn = Callable[["ExprNode"], List[str]]
WhereRowsCollectIdentifiersFn = Callable[["ExprNode"], Set[str]]
WhereRowsParserBundle = Tuple[
    WhereRowsParseFn,
    WhereRowsCapabilityFn,
    WhereRowsCollectIdentifiersFn,
]
WhereRowsParsedExpr = Tuple["ExprNode", WhereRowsCapabilityFn, WhereRowsCollectIdentifiersFn]


@lru_cache(maxsize=1)
def _where_rows_expr_parser_fn() -> Optional[WhereRowsParserBundle]:
    try:
        from graphistry.compute.gfql.expr_parser import (
            collect_identifiers,
            parse_expr,
            validate_expr_capabilities,
        )
        # Ensure parser backend dependencies are available at runtime (e.g., lark).
        try:
            parse_expr("1 = 1")
        except ImportError:
            return None
        return parse_expr, validate_expr_capabilities, collect_identifiers
    except Exception:
        return None


def _where_rows_expr_parse(expr: str) -> Optional[WhereRowsParsedExpr]:
    parser_bundle = _where_rows_expr_parser_fn()
    if parser_bundle is None:
        return None
    parser, capability_checker, collect_identifiers = parser_bundle
    try:
        return parser(expr), capability_checker, collect_identifiers
    except Exception:
        return None


def _where_rows_expr_parser_parse_ok(expr: str) -> bool:
    parsed = _where_rows_expr_parse(expr)
    if parsed is None:
        return False
    node, capability_checker, _collect_identifiers = parsed
    try:
        capability_errors = capability_checker(node)
        if len(capability_errors) > 0:
            return False
        return True
    except Exception:
        return False


def _where_rows_expr_required_cols(expr: str) -> List[str]:
    parsed = _where_rows_expr_parse(expr)
    if parsed is None:
        return []
    node, _capability_checker, collect_identifiers = parsed
    try:
        names = collect_identifiers(node)
    except Exception:
        return []
    cols: Set[str] = set()
    for name in names:
        if not isinstance(name, str) or name == "":
            continue
        cols.add(name.split(".")[0])
    return sorted(cols)


def _expr_required_cols(params: Dict[str, object], key: str = 'expr') -> List[str]:
    expr = params.get(key)
    return _where_rows_expr_required_cols(expr) if isinstance(expr, str) else []

def is_order_keys(v: object) -> bool:
    def _is_static_order_expr_supported(expr: str) -> bool:
        txt = expr.strip()
        if txt == "":
            return False
        parsed = _where_rows_expr_parse(txt)
        if parsed is None:
            return False
        node, capability_checker, _collect_identifiers = parsed
        if is_order_aggregate_alias_ast(node):
            return True
        try:
            capability_errors = capability_checker(node)
        except Exception:
            return False
        if len(capability_errors) > 0:
            return False
        return order_expr_ast_static_supported(node)

    if not isinstance(v, list):
        return False
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        expr, direction = item
        if not is_non_empty_string(expr):
            return False
        if not _is_static_order_expr_supported(expr):
            return False
        if not isinstance(direction, str) or direction.lower() not in {"asc", "desc"}:
            return False
    return True


def is_where_rows_filter_dict(v: object) -> bool:
    if not isinstance(v, dict):
        return False
    # Lazy import avoids circular import at module import time
    from graphistry.compute.predicates.ASTPredicate import ASTPredicate
    for k, val in v.items():
        if not is_non_empty_string(k):
            return False
        if not (isinstance(val, ASTPredicate) or _is_json_compatible_value(val)):
            return False
    return True


def is_list_of_dicts(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(item, dict) for item in v)


def is_where_rows_expr(v: object) -> bool:
    if not is_non_empty_string(v):
        return False
    txt = str(v).strip()
    parser_bundle = _where_rows_expr_parser_fn()
    if parser_bundle is None:
        return False
    return _where_rows_expr_parser_parse_ok(txt)


def _where_rows_requires_node_cols(params: Dict[str, object]) -> List[str]:
    out: List[str] = []
    filter_dict = params.get("filter_dict")
    if isinstance(filter_dict, dict):
        out.extend([k for k in filter_dict.keys() if isinstance(k, str)])
    out.extend(_expr_required_cols(params))
    return sorted(set(out))


def _unwind_requires_node_cols(params: Dict[str, object]) -> List[str]:
    return _expr_required_cols(params)


def _group_by_requires_node_cols(params: Dict[str, object]) -> List[str]:
    out: List[str] = []
    keys = params.get("keys")
    if isinstance(keys, list):
        out.extend([k for k in keys if isinstance(k, str)])
    aggregations = params.get("aggregations")
    if isinstance(aggregations, list):
        for item in aggregations:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            expr = item[2]
            if isinstance(expr, str) and expr != "*":
                out.extend(_where_rows_expr_required_cols(expr))
    return out


# Parser-backed helpers stay local because tests monkeypatch parser availability
# and capability behavior through this module.

SAFELIST_V1: Dict[str, Dict[str, Any]] = {
    'rows': _method_entry(
        allowed_params={'table', 'source', 'alias_endpoints', 'binding_ops'},
        required_params=set(),
        param_validators={
            'table': lambda v: v in ['nodes', 'edges'],
            'source': is_string_or_none,
            'alias_endpoints': lambda v: isinstance(v, dict),
            'binding_ops': is_list_of_dicts,
        },
        description='Set active row table from nodes/edges, optionally filtered by source alias',
        schema_effects=_schema_effects(
            requires_node_cols=_rows_requires_node_cols,
            requires_edge_cols=_rows_requires_edge_cols,
        ),
    ),

    'select': _projection_row_entry('Project row table columns/expressions into aliased outputs'),

    'return_': _projection_row_entry('RETURN-style row projection alias of select()'),

    'with_': _method_entry(
        allowed_params={'items', 'extend'},
        required_params={'items'},
        param_validators={'items': is_projection_items},
        description='WITH-style row projection; extend=True adds columns without dropping existing ones (#880)',
        schema_effects=_schema_effects(adds_node_cols=_select_added_node_cols),
    ),

    'where_rows': _method_entry(
        allowed_params={'filter_dict', 'expr'},
        required_params=set(),
        param_validators={
            'filter_dict': is_where_rows_filter_dict,
            'expr': is_where_rows_expr,
        },
        description='Filter active row table by column values/predicates',
        schema_effects=_schema_effects(requires_node_cols=_where_rows_requires_node_cols),
    ),

    'order_by': _method_entry(
        allowed_params={'keys'},
        required_params={'keys'},
        param_validators={'keys': is_order_keys},
        description='Sort active row table by expression/direction keys',
        schema_effects=NO_SCHEMA_EFFECTS,
    ),

    'skip': _method_entry(
        allowed_params={'value'},
        required_params={'value'},
        param_validators={'value': is_non_negative_int_like},
        description='Skip first N rows from active row table',
        schema_effects=NO_SCHEMA_EFFECTS,
    ),

    'limit': _method_entry(
        allowed_params={'value'},
        required_params={'value'},
        param_validators={'value': is_non_negative_int_like},
        description='Limit active row table to first N rows',
        schema_effects=NO_SCHEMA_EFFECTS,
    ),

    'unwind': _method_entry(
        allowed_params={'expr', 'as_'},
        required_params={'expr'},
        param_validators={
            'expr': is_unwind_expr,
            'as_': is_non_empty_string,
        },
        description='Explode list-like row expression into multiple rows',
        schema_effects=_schema_effects(
            adds_node_cols=_unwind_added_node_cols,
            requires_node_cols=_unwind_requires_node_cols,
        ),
    ),

    'group_by': _method_entry(
        allowed_params={'keys', 'aggregations'},
        required_params={'keys', 'aggregations'},
        param_validators={
            'keys': is_non_empty_list_of_strings,
            'aggregations': is_list_of_agg_specs,
        },
        description='Group rows by keys and compute vectorized aggregations',
        schema_effects=_schema_effects(
            adds_node_cols=_group_by_added_node_cols,
            requires_node_cols=_group_by_requires_node_cols,
        ),
    ),

    'distinct': _method_entry(
        allowed_params=set(),
        required_params=set(),
        param_validators={},
        description='Drop duplicate rows from active row table',
        schema_effects=NO_SCHEMA_EFFECTS,
    ),

    'get_degrees': {
        'allowed_params': {'col', 'degree_in', 'degree_out', 'engine'},
        'required_params': set(),
        'param_validators': {
            'col': is_string,
            'degree_in': is_string,
            'degree_out': is_string,
            'engine': is_string
        },
        'description': 'Calculate node degrees',
        'schema_effects': {
            'adds_node_cols': lambda p: [
                p.get('col', 'degree'),
                p.get('degree_in', 'degree_in'),
                p.get('degree_out', 'degree_out')
            ],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },
    
    'filter_nodes_by_dict': {
        'allowed_params': {'filter_dict'},
        'required_params': {'filter_dict'},
        'param_validators': {
            'filter_dict': is_dict
        },
        'description': 'Filter nodes by attribute values',
        'schema_effects': {
            'adds_node_cols': [],
            'adds_edge_cols': [],
            'requires_node_cols': lambda p: list((p.get('filter_dict') or {}).keys()),
            'requires_edge_cols': []
        }
    },
    
    'filter_edges_by_dict': {
        'allowed_params': {'filter_dict'},
        'required_params': {'filter_dict'},
        'param_validators': {
            'filter_dict': is_dict
        },
        'description': 'Filter edges by attribute values',
        'schema_effects': {
            'adds_node_cols': [],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': lambda p: list((p.get('filter_dict') or {}).keys())
        }
    },
    
    'materialize_nodes': {
        'allowed_params': {'engine', 'reuse'},
        'required_params': set(),
        'param_validators': {
            'engine': is_string,
            'reuse': is_bool
        },
        'description': 'Generate node table from edges',
        'schema_effects': NO_SCHEMA_EFFECTS
    },
    
    'hop': {
        'allowed_params': {
            'nodes', 'hops', 'to_fixed_point', 'direction',
            'source_node_match', 'edge_match', 'destination_node_match',
            'source_node_query', 'edge_query', 'destination_node_query',
            'return_as_wave_front', 'target_wave_front', 'engine',
            'min_hops', 'max_hops', 'output_min_hops', 'output_max_hops',
            'label_node_hops', 'label_edge_hops', 'label_seeds'
        },
        'required_params': set(),
        'param_validators': {
            'hops': is_int,
            'min_hops': is_int,
            'max_hops': is_int,
            'output_min_hops': is_int_or_none,
            'output_max_hops': is_int_or_none,
            'label_node_hops': is_string_or_none,
            'label_edge_hops': is_string_or_none,
            'label_seeds': is_bool,
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
        'description': 'Traverse edges with optional hop bounds and node/edge hop label columns',
        'schema_effects': {
            'adds_node_cols': lambda p: [p['label_node_hops']] if p.get('label_node_hops') else [],
            'adds_edge_cols': lambda p: [p['label_edge_hops']] if p.get('label_edge_hops') else [],
            'requires_node_cols': lambda p: list((p.get('source_node_match') or {}).keys()) + list((p.get('destination_node_match') or {}).keys()),
            'requires_edge_cols': lambda p: list((p.get('edge_match') or {}).keys())
        }
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
        'description': 'Run igraph algorithms',
        'schema_effects': {
            'adds_node_cols': lambda p: [p.get('out_col', p['alg'])],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
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
        'description': 'GPU-accelerated graph layouts',
        'schema_effects': XY_OUT_COL_SCHEMA_EFFECTS
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
        'description': 'igraph-based layouts',
        'schema_effects': XY_OUT_COL_SCHEMA_EFFECTS
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
        'description': 'Graphviz layouts (dot, neato, etc)',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
    },

    'ring_continuous_layout': {
        'allowed_params': {
            'ring_col', 'min_r', 'max_r', 'normalize_ring_col', 'num_rings',
            'ring_step', 'v_start', 'v_end', 'v_step', 'axis', 'format_axis',
            'format_labels', 'reverse', 'play_ms', 'engine'
        },
        'required_params': set(),
        'param_validators': {
            'ring_col': is_string_or_none,
            'min_r': lambda v: v is None or is_int_or_float(v),
            'max_r': lambda v: v is None or is_int_or_float(v),
            'num_rings': lambda v: v is None or is_int(v),
            'ring_step': lambda v: v is None or is_int_or_float(v),
            'v_start': lambda v: v is None or is_int_or_float(v),
            'v_end': lambda v: v is None or is_int_or_float(v),
            'v_step': lambda v: v is None or is_int_or_float(v),
            'axis': lambda v: v is None or is_list_or_dict(v),
            'normalize_ring_col': is_bool,
            'reverse': is_bool,
            'play_ms': lambda v: v is None or is_int(v),
            'engine': lambda v: v is None or v in ('auto', 'pandas', 'cudf', 'dask', 'dask_cudf')
        },
        'description': 'Radial layout based on numeric columns',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
    },

    'ring_categorical_layout': {
        'allowed_params': {
            'ring_col', 'order', 'drop_empty', 'combine_unhandled',
            'append_unhandled', 'min_r', 'max_r', 'axis', 'format_axis',
            'format_labels', 'reverse', 'play_ms', 'engine'
        },
        'required_params': {'ring_col'},
        'param_validators': {
            'ring_col': is_string,
            'order': lambda v: v is None or is_list(v),
            'drop_empty': is_bool,
            'combine_unhandled': is_bool,
            'append_unhandled': is_bool,
            'min_r': lambda v: v is None or is_int_or_float(v),
            'max_r': lambda v: v is None or is_int_or_float(v),
            'axis': lambda v: v is None or is_list_or_dict(v),
            'reverse': is_bool,
            'play_ms': lambda v: v is None or is_int(v),
            'engine': lambda v: v is None or v in ('auto', 'pandas', 'cudf', 'dask', 'dask_cudf')
        },
        'description': 'Radial layout grouping nodes by categories',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
    },

    'time_ring_layout': {
        'allowed_params': {
            'time_col', 'time_start', 'time_end', 'time_unit', 'num_rings',
            'min_r', 'max_r', 'format_axis', 'format_label',
            'reverse', 'play_ms', 'engine'
        },
        'required_params': {'time_col'},
        'param_validators': {
            'time_col': is_string,
            'time_start': lambda v: v is None or isinstance(v, str),
            'time_end': lambda v: v is None or isinstance(v, str),
            'time_unit': lambda v: v is None or v in ('s', 'm', 'h', 'D', 'W', 'M', 'Y', 'C'),
            'num_rings': lambda v: v is None or is_int(v),
            'min_r': lambda v: v is None or is_int_or_float(v),
            'max_r': lambda v: v is None or is_int_or_float(v),
            'reverse': is_bool,
            'play_ms': lambda v: v is None or is_int(v),
            'engine': lambda v: v is None or v in ('auto', 'pandas', 'cudf', 'dask', 'dask_cudf')
        },
        'description': 'Radial layout for time-series rings',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
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
        'description': 'ForceAtlas2 layout algorithm',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
    },

    # Self-edge pruning
    'prune_self_edges': {
        'allowed_params': set(),
        'required_params': set(),
        'param_validators': {},
        'description': 'Remove self-loops from graph',
        'schema_effects': NO_SCHEMA_EFFECTS
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
        'description': 'Collapse nodes by shared attribute values',
        'schema_effects': {
            'adds_node_cols': ['node_final'],
            'adds_edge_cols': ['src_final', 'dst_final'],
            'requires_node_cols': lambda p: [p['column']] if isinstance(p.get('column'), str) else [],
            'requires_edge_cols': []
        }
    },

    'drop_nodes': {
        'allowed_params': {'nodes'},
        'required_params': {'nodes'},
        'param_validators': {
            'nodes': lambda v: isinstance(v, list) or is_dict(v)
        },
        'description': 'Remove specified nodes and their edges',
        'schema_effects': NO_SCHEMA_EFFECTS
    },

    'keep_nodes': {
        'allowed_params': {'nodes'},
        'required_params': {'nodes'},
        'param_validators': {
            'nodes': lambda v: isinstance(v, list) or is_dict(v)
        },
        'description': 'Keep only specified nodes and their edges',
        'schema_effects': NO_SCHEMA_EFFECTS
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
        'description': 'Compute topological levels for DAG analysis',
        'schema_effects': {
            'adds_node_cols': lambda p: [p.get('level_col', 'level')],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
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
        'description': 'Map node column values to colors',
        'schema_effects': NODE_COLUMN_SCHEMA_EFFECTS
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
        'description': 'Map edge column values to colors',
        'schema_effects': EDGE_COLUMN_SCHEMA_EFFECTS
    },

    'encode_point_size': {
        'allowed_params': {'column', 'categorical_mapping', 'default_mapping'},
        'required_params': {'column'},
        'param_validators': {
            'column': is_string,
            'categorical_mapping': is_dict,
            'default_mapping': lambda v: isinstance(v, (int, float))
        },
        'description': 'Map node column values to sizes',
        'schema_effects': NODE_COLUMN_SCHEMA_EFFECTS
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
        'description': 'Map node column values to icons',
        'schema_effects': NODE_COLUMN_SCHEMA_EFFECTS
    },

    # Metadata methods
    'name': {
        'allowed_params': {'name'},
        'required_params': {'name'},
        'param_validators': {
            'name': is_string
        },
        'description': 'Set visualization name',
        'schema_effects': NO_SCHEMA_EFFECTS
    },

    'description': {
        'allowed_params': {'description'},
        'required_params': {'description'},
        'param_validators': {
            'description': is_string
        },
        'description': 'Set visualization description',
        'schema_effects': NO_SCHEMA_EFFECTS
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
        'description': 'Group-in-a-box layout with community detection',
        'schema_effects': XY_NODE_SCHEMA_EFFECTS
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
        'description': 'Transform event data into a hypergraph',
        'schema_effects': {
            'adds_node_cols': _hypergraph_node_adds,
            'adds_edge_cols': _hypergraph_edge_adds,
            'requires_node_cols': lambda p: [] if p.get('from_edges') else _hypergraph_input_required_cols(p),
            'requires_edge_cols': lambda p: _hypergraph_input_required_cols(p) if p.get('from_edges') else []
        }
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
        'description': 'UMAP dimensionality reduction for graph embeddings',
        'schema_effects': {
            'adds_node_cols': _umap_node_adds,
            'adds_edge_cols': _umap_edge_adds,
            'requires_node_cols': _umap_node_required_cols,
            'requires_edge_cols': _umap_edge_required_cols
        }
    }
}


def validate_call_params(function: str, params: Dict[str, object]) -> Dict[str, object]:
    """Validate a GFQL call against the safelist."""
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

    missing_required = required_params - set(params.keys())
    if missing_required:
        raise GFQLTypeError(
            ErrorCode.E105,
            f"Missing required parameters for '{function}'",
            field="params",
            value=list(missing_required),
            suggestion=f"Required parameters: {', '.join(sorted(missing_required))}"
        )

    unknown_params = set(params.keys()) - allowed_params
    if unknown_params:
        raise GFQLTypeError(
            ErrorCode.E303,
            f"Unknown parameters for '{function}'",
            field="params",
            value=list(unknown_params),
            suggestion=f"Allowed parameters: {', '.join(sorted(allowed_params))}"
        )

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
