"""Shared non-parser helpers for GFQL call safelist definitions."""

from typing import Any, Callable, Dict, List, Set

from graphistry.compute.gfql.language_defs import GFQL_AGGREGATION_FUNCTIONS


def is_string(v: object) -> bool:
    return isinstance(v, str)


def is_non_empty_string(v: object) -> bool:
    return isinstance(v, str) and v.strip() != ""


def is_int(v: object) -> bool:
    return isinstance(v, int)


def is_bool(v: object) -> bool:
    return isinstance(v, bool)


def is_int_or_none(v: object) -> bool:
    return v is None or isinstance(v, int)


def is_dict(v: object) -> bool:
    return isinstance(v, dict)


def is_string_or_none(v: object) -> bool:
    return v is None or isinstance(v, str)


def is_int_or_float(v: object) -> bool:
    return isinstance(v, (int, float))


def is_list_of_strings(v: object) -> bool:
    return isinstance(v, list) and all(isinstance(item, str) for item in v)


def is_list(v: object) -> bool:
    return isinstance(v, list)


def is_list_or_dict(v: object) -> bool:
    return isinstance(v, (list, dict))


def _is_json_compatible_value(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, (bool, int, float, str)):
        return True
    if isinstance(v, list):
        return all(_is_json_compatible_value(item) for item in v)
    if isinstance(v, dict):
        return all(isinstance(k, str) and _is_json_compatible_value(val) for k, val in v.items())
    return False


def is_projection_items(v: object) -> bool:
    if not isinstance(v, list):
        return False
    for item in v:
        if isinstance(item, str):
            if not is_non_empty_string(item):
                return False
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return False
        alias, expr = item
        if not is_non_empty_string(alias):
            return False
        if not (isinstance(expr, str) or _is_json_compatible_value(expr)):
            return False
    return True


def is_non_empty_list_of_strings(v: object) -> bool:
    if not isinstance(v, list):
        return False
    return len(v) > 0 and all(isinstance(item, str) for item in v)


def is_list_of_agg_specs(v: object) -> bool:
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
        if func_l not in GFQL_AGGREGATION_FUNCTIONS:
            return False
        if func_l == "count":
            if len(item) == 2:
                continue
            expr = item[2]
            if not (expr is None or expr == "*" or is_non_empty_string(expr)):
                return False
            continue
        if len(item) != 3:
            return False
        expr = item[2]
        if not is_non_empty_string(expr):
            return False
        if expr == "*":
            return False
    return True


def is_unwind_expr(v: object) -> bool:
    if is_non_empty_string(v):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_json_compatible_value(item) for item in v)
    return False


def is_non_negative_int_like(v: object) -> bool:
    if isinstance(v, bool):
        return False
    if isinstance(v, int):
        return v >= 0
    if isinstance(v, float):
        return v.is_integer() and v >= 0
    if isinstance(v, str):
        return v.strip().isdigit()
    return False


def _rows_requires_cols(params: Dict[str, object], table: str) -> List[str]:
    if params.get("table", "nodes") != table:
        return []
    source = params.get("source")
    return [source] if isinstance(source, str) else []


def _rows_requires_node_cols(params: Dict[str, object]) -> List[str]:
    return _rows_requires_cols(params, "nodes")


def _rows_requires_edge_cols(params: Dict[str, object]) -> List[str]:
    return _rows_requires_cols(params, "edges")


def _select_added_node_cols(params: Dict[str, object]) -> List[str]:
    out: List[str] = []
    items = params.get("items")
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            out.append(item)
            continue
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        alias = item[0]
        out.append(alias if isinstance(alias, str) else str(alias))
    return out


def _unwind_added_node_cols(params: Dict[str, object]) -> List[str]:
    as_name = params.get("as_", "value")
    return [as_name] if isinstance(as_name, str) and as_name != "" else []


def _group_by_added_node_cols(params: Dict[str, object]) -> List[str]:
    out: List[str] = []
    keys = params.get("keys")
    if isinstance(keys, list):
        out.extend([k for k in keys if isinstance(k, str)])
    aggregations = params.get("aggregations")
    if isinstance(aggregations, list):
        for item in aggregations:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            alias = item[0]
            if isinstance(alias, str):
                out.append(alias)
    return out


def _symbolic_cols(v: object) -> List[str]:
    if isinstance(v, str):
        return [v]
    if isinstance(v, list) and all(isinstance(item, str) for item in v):
        return list(v)
    return []


def _resolve_hyper_opts(params: Dict[str, object]) -> Dict[str, object]:
    opts = params.get("opts")
    return opts if isinstance(opts, dict) else {}


def _hypergraph_input_required_cols(params: Dict[str, object]) -> List[str]:
    cols: List[str] = []
    entity_types = params.get("entity_types")
    if isinstance(entity_types, list):
        cols.extend([c for c in entity_types if isinstance(c, str)])
    opts = _resolve_hyper_opts(params)
    event_id = opts.get("EVENTID")
    if isinstance(event_id, str):
        cols.append(event_id)
    return cols


def _hypergraph_node_adds(params: Dict[str, object]) -> List[str]:
    opts = _resolve_hyper_opts(params)
    node_id = opts.get("NODEID", "nodeID")
    node_type = opts.get("NODETYPE", "type")
    title = opts.get("TITLE", "nodeTitle")
    out: List[str] = []
    if isinstance(node_id, str):
        out.append(node_id)
    if isinstance(node_type, str):
        out.append(node_type)
    if isinstance(title, str):
        out.append(title)
    return out


def _hypergraph_edge_adds(params: Dict[str, object]) -> List[str]:
    opts = _resolve_hyper_opts(params)
    edge_type = opts.get("EDGETYPE", "edgeType")
    if params.get("direct"):
        src = opts.get("SOURCE", "src")
        dst = opts.get("DESTINATION", "dst")
    else:
        src = opts.get("ATTRIBID", "attribID")
        dst = opts.get("EVENTID", "EventID")
    out: List[str] = []
    if isinstance(src, str):
        out.append(src)
    if isinstance(dst, str):
        out.append(dst)
    if isinstance(edge_type, str):
        out.append(edge_type)
    return out


def _umap_kind(params: Dict[str, object]) -> str:
    kind = params.get("kind", "nodes")
    return kind if isinstance(kind, str) else "nodes"


def _umap_suffix(params: Dict[str, object]) -> str:
    suffix = params.get("suffix", "")
    return suffix if isinstance(suffix, str) else ""


def _umap_required_cols(params: Dict[str, object], kind: str) -> List[str]:
    if _umap_kind(params) != kind:
        return []
    return _symbolic_cols(params.get("X")) + _symbolic_cols(params.get("y"))


def _umap_node_required_cols(params: Dict[str, object]) -> List[str]:
    return _umap_required_cols(params, "nodes")


def _umap_edge_required_cols(params: Dict[str, object]) -> List[str]:
    return _umap_required_cols(params, "edges")


def _umap_node_adds(params: Dict[str, object]) -> List[str]:
    if _umap_kind(params) != "nodes":
        return []
    suffix = _umap_suffix(params)
    return [f"x{suffix}", f"y{suffix}"]


def _umap_edge_adds(params: Dict[str, object]) -> List[str]:
    kind = _umap_kind(params)
    suffix = _umap_suffix(params)
    if kind == "edges":
        return [f"x{suffix}", f"y{suffix}"]
    if kind == "nodes" and params.get("encode_weight", True):
        return ["_src_implicit", "_dst_implicit", f"_weight{suffix}"]
    return []


def _xy_out_cols(params: Dict[str, object]) -> List[str]:
    out: List[str] = []
    x_col = params.get("x_out_col", "x")
    y_col = params.get("y_out_col", "y")
    if isinstance(x_col, str):
        out.append(x_col)
    if isinstance(y_col, str):
        out.append(y_col)
    return out


def _required_column(params: Dict[str, object]) -> List[str]:
    col = params.get("column")
    return [col] if isinstance(col, str) else []


def is_dict_str_to_list_str(v: object) -> bool:
    if not isinstance(v, dict):
        return False
    for key, val in v.items():
        if not isinstance(key, str):
            return False
        if not is_list_of_strings(val):
            return False
    return True


def validate_hypergraph_opts(v: object) -> bool:
    if not isinstance(v, dict):
        return False

    string_keys = {
        "TITLE",
        "DELIM",
        "NODEID",
        "ATTRIBID",
        "EVENTID",
        "EVENTTYPE",
        "SOURCE",
        "DESTINATION",
        "CATEGORY",
        "NODETYPE",
        "EDGETYPE",
        "NULLVAL",
    }

    for key, val in v.items():
        if not isinstance(key, str):
            return False
        if key in string_keys:
            if not isinstance(val, str):
                return False
        elif key == "SKIP":
            if not is_list_of_strings(val):
                return False
        elif key in ("CATEGORIES", "EDGES"):
            if not is_dict_str_to_list_str(val):
                return False
        elif not isinstance(val, (str, int, float, bool, list, dict, type(None))):
            return False
    return True


NO_SCHEMA_EFFECTS: Dict[str, List[str]] = {
    "adds_node_cols": [],
    "adds_edge_cols": [],
    "requires_node_cols": [],
    "requires_edge_cols": [],
}

XY_OUT_COL_SCHEMA_EFFECTS: Dict[str, Any] = {
    "adds_node_cols": _xy_out_cols,
    "adds_edge_cols": [],
    "requires_node_cols": [],
    "requires_edge_cols": [],
}

XY_NODE_SCHEMA_EFFECTS: Dict[str, List[str]] = {
    "adds_node_cols": ["x", "y"],
    "adds_edge_cols": [],
    "requires_node_cols": [],
    "requires_edge_cols": [],
}

NODE_COLUMN_SCHEMA_EFFECTS: Dict[str, Any] = {
    "adds_node_cols": [],
    "adds_edge_cols": [],
    "requires_node_cols": _required_column,
    "requires_edge_cols": [],
}

EDGE_COLUMN_SCHEMA_EFFECTS: Dict[str, Any] = {
    "adds_node_cols": [],
    "adds_edge_cols": [],
    "requires_node_cols": [],
    "requires_edge_cols": _required_column,
}


def _schema_effects(
    *,
    adds_node_cols: Any = None,
    adds_edge_cols: Any = None,
    requires_node_cols: Any = None,
    requires_edge_cols: Any = None,
) -> Dict[str, Any]:
    return {
        "adds_node_cols": [] if adds_node_cols is None else adds_node_cols,
        "adds_edge_cols": [] if adds_edge_cols is None else adds_edge_cols,
        "requires_node_cols": [] if requires_node_cols is None else requires_node_cols,
        "requires_edge_cols": [] if requires_edge_cols is None else requires_edge_cols,
    }


def _method_entry(
    *,
    allowed_params: Set[str],
    required_params: Set[str],
    param_validators: Dict[str, Callable[[object], bool]],
    description: str,
    schema_effects: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "allowed_params": allowed_params,
        "required_params": required_params,
        "param_validators": param_validators,
        "description": description,
        "schema_effects": schema_effects,
    }


def _projection_row_entry(description: str) -> Dict[str, Any]:
    return _method_entry(
        allowed_params={"items"},
        required_params={"items"},
        param_validators={"items": is_projection_items},
        description=description,
        schema_effects=_schema_effects(adds_node_cols=_select_added_node_cols),
    )
