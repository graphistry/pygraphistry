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

import re
from typing import Dict, Any, List
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError

_QUOTED_STRING_RE = re.compile(r"(?s)'(?:\\\\.|[^'])*'|\"(?:\\\\.|[^\"])*\"")


def _strip_quoted_string_literals(txt: str) -> str:
    return _QUOTED_STRING_RE.sub(" ", txt)


def _is_identifier_char(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def _where_rows_string_predicate_has_dynamic_rhs(expr: str) -> bool:
    """Detect dynamic column-like RHS for CONTAINS/STARTS WITH/ENDS WITH.

    Runtime only supports scalar RHS for vectorized string predicates; reject
    clearly dynamic identifier RHS at validation time.
    """

    txt = expr
    lower = txt.lower()
    n = len(txt)
    i = 0
    in_single = False
    in_double = False
    escaped = False

    def _match_keyword(start: int) -> int:
        # Return end index (exclusive) when keyword matches, else -1.
        for pattern in (r"contains\b", r"starts\s+with\b", r"ends\s+with\b"):
            m = re.match(pattern, lower[start:])
            if m is not None:
                return start + m.end()
        return -1

    while i < n:
        ch = txt[i]
        if in_single or in_double:
            if escaped:
                escaped = False
                i += 1
                continue
            if ch == "\\":
                escaped = True
                i += 1
                continue
            if in_single and ch == "'":
                in_single = False
            elif in_double and ch == '"':
                in_double = False
            i += 1
            continue

        if ch == "'":
            in_single = True
            i += 1
            continue
        if ch == '"':
            in_double = True
            i += 1
            continue

        kw_end = _match_keyword(i)
        if kw_end < 0:
            i += 1
            continue

        if i > 0 and _is_identifier_char(txt[i - 1]):
            i += 1
            continue
        if kw_end < n and _is_identifier_char(txt[kw_end]):
            i += 1
            continue

        j = kw_end
        while j < n and txt[j].isspace():
            j += 1
        if j >= n:
            return True

        rhs = txt[j:]
        if rhs.startswith("'") or rhs.startswith('"'):
            i = j + 1
            continue
        if re.match(r"(?i)null\b", rhs):
            i = j + 1
            continue
        if re.match(r"[A-Za-z_][A-Za-z0-9_]*", rhs):
            return True
        i = j + 1

    return False


# Type validators
def is_string(v: Any) -> bool:
    return isinstance(v, str)


def is_non_empty_string(v: Any) -> bool:
    return isinstance(v, str) and v.strip() != ""


def is_int(v: Any) -> bool:
    return isinstance(v, int)


def is_bool(v: Any) -> bool:
    return isinstance(v, bool)

def is_int_or_none(v: Any) -> bool:
    return v is None or isinstance(v, int)


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


def is_list(v: Any) -> bool:
    return isinstance(v, list)


def is_list_or_dict(v: Any) -> bool:
    return isinstance(v, (list, dict))


def is_list_of_pairs(v: Any) -> bool:
    return isinstance(v, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in v)


def _is_json_compatible_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, list):
        return all(_is_json_compatible_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_compatible_value(val) for k, val in v.items())
    return False


def is_projection_items(v: Any) -> bool:
    if not isinstance(v, list):
        return False
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        alias, expr = item
        if not is_non_empty_string(alias):
            return False
        if not (isinstance(expr, str) or _is_json_compatible_value(expr)):
            return False
    return True


def is_order_keys(v: Any) -> bool:
    def _is_static_order_expr_supported(expr: str) -> bool:
        safe_funcs = {
            "abs",
            "tostring",
            "toboolean",
            "coalesce",
            "size",
            "count",
            "sum",
            "min",
            "max",
            "avg",
            "mean",
            "collect",
        }
        txt = expr.strip()
        if txt == "":
            return False
        if re.search(r"[\[\]{}]", txt):
            return False
        if re.search(r"(?i)\b(?:ANY|ALL|NONE|SINGLE)\s*\(", txt):
            return False
        func_calls = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt)
        if any(fn.lower() not in safe_funcs for fn in func_calls):
            return False
        if re.fullmatch(r"[A-Za-z0-9_.'\"+\-*/%<>=!(),\s]+", txt) is None:
            return False
        return True

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


def is_where_rows_filter_dict(v: Any) -> bool:
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


def is_where_rows_expr(v: Any) -> bool:
    if not is_non_empty_string(v):
        return False
    txt = str(v).strip()
    txt_lex = _strip_quoted_string_literals(txt)
    safe_funcs = {
        "abs",
        "coalesce",
        "ends_with",
        "head",
        "nodes",
        "relationships",
        "reverse",
        "sign",
        "size",
        "starts_with",
        "tail",
        "toboolean",
        "tostring",
    }
    if re.search(r"(?i)\b(?:ANY|ALL|NONE|SINGLE)\s*\(", txt_lex):
        return False
    func_calls = [
        fn
        for fn in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", txt_lex)
        if fn.lower() not in {"and", "or", "not"}
    ]
    if any(fn.lower() not in safe_funcs for fn in func_calls):
        return False
    if _where_rows_string_predicate_has_dynamic_rhs(txt):
        return False
    if re.fullmatch(r"[A-Za-z0-9_.'\"+\-*/%<>=!(),\[\]{}:\s]+", txt) is None:
        return False
    return True


def is_non_empty_list_of_strings(v: Any) -> bool:
    return is_list_of_strings(v) and len(v) > 0


def is_list_of_agg_specs(v: Any) -> bool:
    allowed = {"count", "count_distinct", "sum", "min", "max", "avg", "mean", "collect"}
    if not isinstance(v, list):
        return False
    for item in v:
        if not isinstance(item, (list, tuple)) or len(item) not in {2, 3}:
            return False
        alias = item[0]
        func = item[1]
        if not is_non_empty_string(alias):
            return False
        if not isinstance(func, str):
            return False
        func_l = func.lower()
        if func_l not in allowed:
            return False
        if func_l == "count":
            if len(item) == 2:
                continue
            expr = item[2]
            if not (expr is None or expr == "*" or is_non_empty_string(expr)):
                return False
            continue
        # non-count aggs require a column expression
        if len(item) != 3:
            return False
        expr = item[2]
        if not is_non_empty_string(expr):
            return False
        if expr == "*":
                return False
    return True


def is_unwind_expr(v: Any) -> bool:
    if is_non_empty_string(v):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_json_compatible_value(item) for item in v)
    return False


def is_non_negative_int_like(v: Any) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= 0
    if isinstance(v, float):
        return v.is_integer() and v >= 0
    if isinstance(v, str):
        txt = v.strip()
        return txt.isdigit()
    return False


def _rows_requires_node_cols(params: Dict[str, Any]) -> list:
    if params.get('table', 'nodes') != 'nodes':
        return []
    source = params.get('source')
    return [source] if isinstance(source, str) else []


def _rows_requires_edge_cols(params: Dict[str, Any]) -> list:
    if params.get('table', 'nodes') != 'edges':
        return []
    source = params.get('source')
    return [source] if isinstance(source, str) else []


def _select_added_node_cols(params: Dict[str, Any]) -> list:
    out: list = []
    for item in params.get('items', []):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        alias = item[0]
        if isinstance(alias, str):
            out.append(alias)
        else:
            out.append(str(alias))
    return out


def _where_rows_requires_node_cols(params: Dict[str, Any]) -> list:
    out: list = []
    filter_dict = params.get('filter_dict')
    if not isinstance(filter_dict, dict):
        out = []
    else:
        out.extend([k for k in filter_dict.keys() if isinstance(k, str)])

    expr = params.get('expr')
    if isinstance(expr, str):
        expr_clean = _strip_quoted_string_literals(expr)
        ids = set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\b", expr_clean))
        reserved = {
            "and",
            "or",
            "not",
            "is",
            "null",
            "in",
            "contains",
            "starts",
            "with",
            "ends",
            "true",
            "false",
            "abs",
            "coalesce",
            "head",
            "nodes",
            "relationships",
            "reverse",
            "sign",
            "size",
            "tail",
            "toboolean",
            "tostring",
        }
        out.extend([name for name in ids if name.lower() not in reserved])
    return sorted(set(out))


def _unwind_requires_node_cols(params: Dict[str, Any]) -> list:
    expr = params.get('expr')
    if isinstance(expr, str):
        txt = expr.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", txt):
            return [txt]
    return []


def _unwind_added_node_cols(params: Dict[str, Any]) -> list:
    as_name = params.get('as_', 'value')
    return [as_name] if isinstance(as_name, str) and as_name != '' else []


def _group_by_requires_node_cols(params: Dict[str, Any]) -> list:
    out: list = []
    keys = params.get('keys')
    if isinstance(keys, list):
        out.extend([k for k in keys if isinstance(k, str)])
    aggregations = params.get('aggregations')
    if isinstance(aggregations, list):
        for item in aggregations:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                continue
            expr = item[2]
            if (
                isinstance(expr, str)
                and expr != "*"
                and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr.strip()) is not None
            ):
                out.append(expr)
    return out


def _group_by_added_node_cols(params: Dict[str, Any]) -> list:
    out: list = []
    keys = params.get('keys')
    if isinstance(keys, list):
        out.extend([k for k in keys if isinstance(k, str)])
    aggregations = params.get('aggregations')
    if isinstance(aggregations, list):
        for item in aggregations:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            alias = item[0]
            if isinstance(alias, str):
                out.append(alias)
    return out


def _symbolic_cols(v: Any) -> list:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(item, str) for item in v):
        return list(v)
    return []


def _resolve_hyper_opts(params: Dict[str, Any]) -> Dict[str, Any]:
    opts = params.get('opts')
    return opts if isinstance(opts, dict) else {}


def _hypergraph_input_required_cols(params: Dict[str, Any]) -> list:
    cols: list = []
    entity_types = params.get('entity_types')
    if isinstance(entity_types, list):
        cols.extend([c for c in entity_types if isinstance(c, str)])
    opts = _resolve_hyper_opts(params)
    event_id = opts.get('EVENTID')
    if isinstance(event_id, str):
        cols.append(event_id)
    return cols


def _hypergraph_node_adds(params: Dict[str, Any]) -> list:
    opts = _resolve_hyper_opts(params)
    node_id = opts.get('NODEID', 'nodeID')
    node_type = opts.get('NODETYPE', 'type')
    title = opts.get('TITLE', 'nodeTitle')
    out = []
    if isinstance(node_id, str):
        out.append(node_id)
    if isinstance(node_type, str):
        out.append(node_type)
    if isinstance(title, str):
        out.append(title)
    return out


def _hypergraph_edge_adds(params: Dict[str, Any]) -> list:
    opts = _resolve_hyper_opts(params)
    edge_type = opts.get('EDGETYPE', 'edgeType')
    if params.get('direct'):
        src = opts.get('SOURCE', 'src')
        dst = opts.get('DESTINATION', 'dst')
    else:
        src = opts.get('ATTRIBID', 'attribID')
        dst = opts.get('EVENTID', 'EventID')
    out = []
    if isinstance(src, str):
        out.append(src)
    if isinstance(dst, str):
        out.append(dst)
    if isinstance(edge_type, str):
        out.append(edge_type)
    return out


def _umap_kind(params: Dict[str, Any]) -> str:
    return params.get('kind', 'nodes')


def _umap_suffix(params: Dict[str, Any]) -> str:
    suffix = params.get('suffix', '')
    return suffix if isinstance(suffix, str) else ''


def _umap_node_required_cols(params: Dict[str, Any]) -> list:
    if _umap_kind(params) != 'nodes':
        return []
    return _symbolic_cols(params.get('X')) + _symbolic_cols(params.get('y'))


def _umap_edge_required_cols(params: Dict[str, Any]) -> list:
    if _umap_kind(params) != 'edges':
        return []
    return _symbolic_cols(params.get('X')) + _symbolic_cols(params.get('y'))


def _umap_node_adds(params: Dict[str, Any]) -> list:
    if _umap_kind(params) != 'nodes':
        return []
    suffix = _umap_suffix(params)
    return [f'x{suffix}', f'y{suffix}']


def _umap_edge_adds(params: Dict[str, Any]) -> list:
    kind = _umap_kind(params)
    suffix = _umap_suffix(params)
    if kind == 'edges':
        return [f'x{suffix}', f'y{suffix}']
    if kind == 'nodes' and params.get('encode_weight', True):
        return ['_src_implicit', '_dst_implicit', f'_weight{suffix}']
    return []


def _xy_out_cols(params: Dict[str, Any]) -> list:
    return [params.get('x_out_col', 'x'), params.get('y_out_col', 'y')]


def _required_column(params: Dict[str, Any]) -> list:
    col = params.get('column')
    return [col] if isinstance(col, str) else []




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
# Shared no-op schema effects for calls that neither require nor add columns.
NO_SCHEMA_EFFECTS: Dict[str, List[str]] = {
    'adds_node_cols': [],
    'adds_edge_cols': [],
    'requires_node_cols': [],
    'requires_edge_cols': [],
}

XY_OUT_COL_SCHEMA_EFFECTS: Dict[str, Any] = {
    'adds_node_cols': _xy_out_cols,
    'adds_edge_cols': [],
    'requires_node_cols': [],
    'requires_edge_cols': [],
}

XY_NODE_SCHEMA_EFFECTS: Dict[str, List[str]] = {
    'adds_node_cols': ['x', 'y'],
    'adds_edge_cols': [],
    'requires_node_cols': [],
    'requires_edge_cols': [],
}

NODE_COLUMN_SCHEMA_EFFECTS: Dict[str, Any] = {
    'adds_node_cols': [],
    'adds_edge_cols': [],
    'requires_node_cols': _required_column,
    'requires_edge_cols': [],
}

EDGE_COLUMN_SCHEMA_EFFECTS: Dict[str, Any] = {
    'adds_node_cols': [],
    'adds_edge_cols': [],
    'requires_node_cols': [],
    'requires_edge_cols': _required_column,
}

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
    'rows': {
        'allowed_params': {'table', 'source'},
        'required_params': set(),
        'param_validators': {
            'table': lambda v: v in ['nodes', 'edges'],
            'source': is_string_or_none
        },
        'description': 'Set active row table from nodes/edges, optionally filtered by source alias',
        'schema_effects': {
            'adds_node_cols': [],
            'adds_edge_cols': [],
            'requires_node_cols': _rows_requires_node_cols,
            'requires_edge_cols': _rows_requires_edge_cols
        }
    },

    'select': {
        'allowed_params': {'items'},
        'required_params': {'items'},
        'param_validators': {
            'items': is_projection_items
        },
        'description': 'Project row table columns/expressions into aliased outputs',
        'schema_effects': {
            'adds_node_cols': _select_added_node_cols,
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    'with_': {
        'allowed_params': {'items'},
        'required_params': {'items'},
        'param_validators': {
            'items': is_projection_items
        },
        'description': 'WITH-style row projection with scope-reset semantics',
        'schema_effects': {
            'adds_node_cols': _select_added_node_cols,
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    'where_rows': {
        'allowed_params': {'filter_dict', 'expr'},
        'required_params': set(),
        'param_validators': {
            'filter_dict': is_where_rows_filter_dict,
            'expr': is_where_rows_expr,
        },
        'description': 'Filter active row table by column values/predicates',
        'schema_effects': {
            'adds_node_cols': [],
            'adds_edge_cols': [],
            'requires_node_cols': _where_rows_requires_node_cols,
            'requires_edge_cols': []
        }
    },

    'order_by': {
        'allowed_params': {'keys'},
        'required_params': {'keys'},
        'param_validators': {
            'keys': is_order_keys
        },
        'description': 'Sort active row table by expression/direction keys',
        'schema_effects': {
            'adds_node_cols': [],
            'adds_edge_cols': [],
            'requires_node_cols': [],
            'requires_edge_cols': []
        }
    },

    'skip': {
        'allowed_params': {'value'},
        'required_params': {'value'},
        'param_validators': {
            'value': is_non_negative_int_like
        },
        'description': 'Skip first N rows from active row table',
        'schema_effects': NO_SCHEMA_EFFECTS
    },

    'limit': {
        'allowed_params': {'value'},
        'required_params': {'value'},
        'param_validators': {
            'value': is_non_negative_int_like
        },
        'description': 'Limit active row table to first N rows',
        'schema_effects': NO_SCHEMA_EFFECTS
    },

    'unwind': {
        'allowed_params': {'expr', 'as_'},
        'required_params': {'expr'},
        'param_validators': {
            'expr': is_unwind_expr,
            'as_': is_non_empty_string
        },
        'description': 'Explode list-like row expression into multiple rows',
        'schema_effects': {
            'adds_node_cols': _unwind_added_node_cols,
            'adds_edge_cols': [],
            'requires_node_cols': _unwind_requires_node_cols,
            'requires_edge_cols': []
        }
    },

    'group_by': {
        'allowed_params': {'keys', 'aggregations'},
        'required_params': {'keys', 'aggregations'},
        'param_validators': {
            'keys': is_non_empty_list_of_strings,
            'aggregations': is_list_of_agg_specs
        },
        'description': 'Group rows by keys and compute vectorized aggregations',
        'schema_effects': {
            'adds_node_cols': _group_by_added_node_cols,
            'adds_edge_cols': [],
            'requires_node_cols': _group_by_requires_node_cols,
            'requires_edge_cols': []
        }
    },

    'distinct': {
        'allowed_params': set(),
        'required_params': set(),
        'param_validators': {},
        'description': 'Drop duplicate rows from active row table',
        'schema_effects': NO_SCHEMA_EFFECTS
    },

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
