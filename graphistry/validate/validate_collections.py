import json
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

from graphistry.client_session import strtobool
from graphistry.models.collections import CollectionsInput
from graphistry.models.types import ValidationMode, ValidationParam
from graphistry.util import warn as emit_warn
_ALLOWED_COLLECTION_FIELDS_ORDER = (
    'type',
    'id',
    'name',
    'description',
    'node_color',
    'edge_color',
    'expr',
)
_ALLOWED_COLLECTION_FIELDS_SET = set(_ALLOWED_COLLECTION_FIELDS_ORDER)


def normalize_validation_params(
    validate: ValidationParam = 'autofix',
    warn: bool = True
) -> Tuple[ValidationMode, bool]:
    if validate is True:
        validate_mode: ValidationMode = 'strict'
    elif validate is False:
        validate_mode = 'autofix'
        warn = False
    else:
        validate_mode = validate
    return validate_mode, warn


def encode_collections(collections: List[Dict[str, Any]]) -> str:
    json_str = json.dumps(collections, separators=(',', ':'), ensure_ascii=True)
    return quote(json_str, safe='')


def _issue(
    message: str,
    data: Optional[Dict[str, Any]],
    validate_mode: ValidationMode,
    warn: bool
) -> None:
    error = ValueError({'message': message, 'data': data} if data else {'message': message})
    if validate_mode in ('strict', 'strict-fast'):
        raise error
    if warn and validate_mode == 'autofix':
        emit_warn(f"Collections validation warning: {message} ({data})")


def _coerce_collection_list(
    parsed: Any,
    validate_mode: ValidationMode,
    warn: bool
) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    _issue('Collections JSON must be a list or dict', {'type': type(parsed).__name__}, validate_mode, warn)
    return []


def _parse_collections_input(
    collections: CollectionsInput,
    validate_mode: ValidationMode,
    warn: bool
) -> List[Dict[str, Any]]:
    if isinstance(collections, list):
        return _coerce_collection_list(collections, validate_mode, warn)
    if isinstance(collections, dict):
        return _coerce_collection_list(collections, validate_mode, warn)
    if isinstance(collections, str):
        try:
            parsed = json.loads(collections)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(unquote(collections))
            except json.JSONDecodeError as exc:
                _issue('Collections string must be JSON or URL-encoded JSON', {'error': str(exc)}, validate_mode, warn)
                return []
        return _coerce_collection_list(parsed, validate_mode, warn)
    _issue('Collections must be a list, dict, or JSON string', {'type': type(collections).__name__}, validate_mode, warn)
    return []


def _normalize_str_field(
    entry: Dict[str, Any],
    key: str,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int,
    autofix_drop: bool
) -> None:
    if key not in entry or entry[key] is None:
        return
    if isinstance(entry[key], str):
        return
    _issue(
        f'Collection field "{key}" should be a string',
        {'index': entry_index, 'value': entry[key], 'type': type(entry[key]).__name__},
        validate_mode,
        warn
    )
    if validate_mode == 'autofix':
        if autofix_drop:
            entry.pop(key, None)
        else:
            entry[key] = str(entry[key])


def _normalize_sets_list(
    sets_value: Any,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int
) -> Optional[List[str]]:
    if not isinstance(sets_value, list):
        _issue(
            'Intersection sets must be a list of strings',
            {'index': entry_index, 'value': sets_value, 'type': type(sets_value).__name__},
            validate_mode,
            warn
        )
        return None
    out: List[str] = []
    for set_id in sets_value:
        if isinstance(set_id, str):
            out.append(set_id)
            continue
        _issue(
            'Intersection set IDs must be strings',
            {'index': entry_index, 'value': set_id, 'type': type(set_id).__name__},
            validate_mode,
            warn
        )
        if validate_mode == 'autofix':
            out.append(str(set_id))
    return out


def _normalize_gfql_ops(
    gfql_ops: Any,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int
) -> Optional[List[Dict[str, Any]]]:
    from graphistry.compute.ast import ASTObject, from_json as ast_from_json
    from graphistry.compute.chain import Chain

    if gfql_ops is None:
        _issue('GFQL chain is missing', {'index': entry_index}, validate_mode, warn)
        return None
    if isinstance(gfql_ops, str):
        try:
            gfql_ops = json.loads(gfql_ops)
        except json.JSONDecodeError as exc:
            _issue('GFQL chain string must be JSON', {'index': entry_index, 'error': str(exc)}, validate_mode, warn)
            return None

    def _normalize_op(op: object) -> Optional[Dict[str, Any]]:
        if isinstance(op, ASTObject):
            return op.to_json()
        if isinstance(op, dict):
            try:
                parsed = ast_from_json(op, validate=True)
            except Exception as exc:
                _issue(
                    'Invalid GFQL operation in collection',
                    {'index': entry_index, 'op': op, 'error': str(exc)},
                    validate_mode,
                    warn
                )
                return None
            return parsed.to_json()
        _issue(
            'GFQL operations must be AST objects or dicts',
            {'index': entry_index, 'value': op, 'type': type(op).__name__},
            validate_mode,
            warn
        )
        return None

    def _normalize_ops_list(raw: object) -> Optional[List[Dict[str, Any]]]:
        if isinstance(raw, list):
            normalized_ops: List[Dict[str, Any]] = []
            for op in raw:
                normalized_op = _normalize_op(op)
                if normalized_op is None:
                    return None
                normalized_ops.append(normalized_op)
            if len(normalized_ops) == 0:
                _issue('GFQL chain is empty', {'index': entry_index}, validate_mode, warn)
                return None
            return normalized_ops
        if isinstance(raw, (ASTObject, dict)):
            normalized_op = _normalize_op(raw)
            if normalized_op is None:
                return None
            return [normalized_op]
        _issue(
            'GFQL operations must be a list, AST object, or dict',
            {'index': entry_index, 'value': raw, 'type': type(raw).__name__},
            validate_mode,
            warn
        )
        return None

    def _extract_ops_value(raw: object) -> object:
        if isinstance(raw, Chain):
            return raw.chain
        if isinstance(raw, ASTObject):
            return raw
        if isinstance(raw, dict):
            if raw.get('type') == 'gfql_chain' and 'gfql' in raw:
                return raw.get('gfql')
            if raw.get('type') == 'Chain' and 'chain' in raw:
                return raw.get('chain')
            if 'gfql' in raw:
                return raw.get('gfql')
            if 'chain' in raw:
                return raw.get('chain')
            return raw
        if isinstance(raw, list):
            return raw
        return raw

    return _normalize_ops_list(_extract_ops_value(gfql_ops))

    _issue(
        'GFQL operations must be a Chain, AST object, list, or dict',
        {'index': entry_index, 'type': type(gfql_ops).__name__},
        validate_mode,
        warn
    )
    return None



def _normalize_gfql_expr(
    expr: Any,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int
) -> Optional[Dict[str, Any]]:
    if isinstance(expr, dict) and expr.get('type') == 'intersection':
        _issue('Set collection expr cannot be intersection', {'index': entry_index}, validate_mode, warn)
        return None
    ops = _normalize_gfql_ops(expr, validate_mode, warn, entry_index)
    if ops is None:
        return None
    return {'type': 'gfql_chain', 'gfql': ops}


def _normalize_intersection_expr(
    expr: Any,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int
) -> Optional[Dict[str, Any]]:
    if not isinstance(expr, dict):
        _issue('Intersection expr must be a dict', {'index': entry_index}, validate_mode, warn)
        return None
    expr_type = expr.get('type', 'intersection')
    if expr_type != 'intersection':
        _issue(
            'Intersection expr type must be "intersection"',
            {'index': entry_index, 'value': expr_type},
            validate_mode,
            warn
        )
        return None
    sets_value = expr.get('sets', expr.get('intersection'))
    if sets_value is None:
        _issue('Intersection expr missing "sets"', {'index': entry_index}, validate_mode, warn)
        return None
    sets_list = _normalize_sets_list(sets_value, validate_mode, warn, entry_index)
    if sets_list is None:
        return None
    return {'type': 'intersection', 'sets': sets_list}


def normalize_collections(
    collections: CollectionsInput,
    validate: ValidationParam = 'autofix',
    warn: bool = True
) -> List[Dict[str, Any]]:
    validate_mode, warn = normalize_validation_params(validate, warn)
    items = _parse_collections_input(collections, validate_mode, warn)

    normalized: List[Dict[str, Any]] = []
    for idx, entry in enumerate(items):
        if not isinstance(entry, dict):
            _issue(
                'Collection entries must be dictionaries',
                {'index': idx, 'type': type(entry).__name__},
                validate_mode,
                warn
            )
            if validate_mode == 'autofix':
                continue
            return []

        unexpected_fields = [key for key in entry.keys() if key not in _ALLOWED_COLLECTION_FIELDS_SET]
        if unexpected_fields:
            _issue(
                'Unexpected fields in collection',
                {'index': idx, 'fields': unexpected_fields},
                validate_mode,
                warn
            )

        normalized_entry = {key: entry[key] for key in _ALLOWED_COLLECTION_FIELDS_ORDER if key in entry}
        collection_type = normalized_entry.get('type', 'set')
        if not isinstance(collection_type, str):
            _issue(
                'Collection type must be a string',
                {'index': idx, 'value': collection_type, 'type': type(collection_type).__name__},
                validate_mode,
                warn
            )
            if validate_mode == 'autofix':
                collection_type = str(collection_type)
            else:
                continue
        collection_type = collection_type.lower()
        normalized_entry['type'] = collection_type

        if collection_type not in ('set', 'intersection'):
            _issue(
                'Collection type must be "set" or "intersection"',
                {'index': idx, 'value': collection_type},
                validate_mode,
                warn
            )
            if validate_mode == 'autofix':
                continue
            return []

        for field in ('id', 'name', 'description'):
            _normalize_str_field(normalized_entry, field, validate_mode, warn, idx, autofix_drop=False)
        for field in ('node_color', 'edge_color'):
            _normalize_str_field(normalized_entry, field, validate_mode, warn, idx, autofix_drop=True)

        expr = normalized_entry.get('expr')
        if collection_type == 'intersection':
            normalized_expr = _normalize_intersection_expr(expr, validate_mode, warn, idx)
        else:
            normalized_expr = _normalize_gfql_expr(expr, validate_mode, warn, idx)
        if normalized_expr is None:
            if validate_mode == 'autofix':
                continue
            return []
        normalized_entry['expr'] = normalized_expr
        normalized_entry = {
            key: normalized_entry[key]
            for key in _ALLOWED_COLLECTION_FIELDS_ORDER
            if key in normalized_entry
        }
        normalized.append(normalized_entry)

    return normalized


def normalize_collections_url_params(
    url_params: Dict[str, Any],
    validate: ValidationParam = 'autofix',
    warn: bool = True
) -> Dict[str, Any]:
    validate_mode, warn = normalize_validation_params(validate, warn)
    updated = dict(url_params)

    if 'collections' in updated:
        normalized = normalize_collections(updated['collections'], validate_mode, warn)
        if len(normalized) > 0:
            updated['collections'] = encode_collections(normalized)
        else:
            if validate_mode in ('strict', 'strict-fast'):
                return updated
            updated.pop('collections', None)

    if 'showCollections' in updated:
        value = updated['showCollections']
        if isinstance(value, bool):
            pass
        else:
            try:
                updated['showCollections'] = strtobool(value)
            except Exception as exc:
                _issue(
                    'showCollections must be a boolean',
                    {'value': value, 'error': str(exc)},
                    validate_mode,
                    warn
                )
                if validate_mode == 'autofix':
                    updated.pop('showCollections', None)

    for color_key in ('collectionsGlobalNodeColor', 'collectionsGlobalEdgeColor'):
        if color_key in updated:
            value = updated[color_key]
            if value is None:
                updated.pop(color_key, None)
                continue
            if not isinstance(value, str):
                _issue(
                    f'{color_key} must be a string',
                    {'value': value, 'type': type(value).__name__},
                    validate_mode,
                    warn
                )
                if validate_mode == 'autofix':
                    value = str(value)
            if isinstance(value, str) and value.startswith('#'):
                value = value[1:]
            updated[color_key] = value

    return updated
