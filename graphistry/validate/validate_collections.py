import json
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import quote, unquote

from graphistry.client_session import strtobool
from graphistry.compute.exceptions import GFQLValidationError
from graphistry.models.collections import Collection, CollectionsInput
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


def _parse_collections_input(
    collections: CollectionsInput,
    validate_mode: ValidationMode,
    warn: bool
) -> Union[List[Dict[str, Any]], List[Collection]]:
    """Parse collections input to a list of dicts, handling list/dict/JSON string inputs."""
    if isinstance(collections, list):
        return collections
    if isinstance(collections, dict):
        return [collections]
    if isinstance(collections, str):
        try:
            parsed = json.loads(collections)
        except json.JSONDecodeError:
            try:
                parsed = json.loads(unquote(collections))
            except json.JSONDecodeError as exc:
                _issue('Collections string must be JSON or URL-encoded JSON', {'error': str(exc)}, validate_mode, warn)
                return []
        # Coerce parsed JSON to list
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        _issue('Collections JSON must be a list or dict', {'type': type(parsed).__name__}, validate_mode, warn)
        return []
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

    if len(out) == 0:
        _issue(
            'Intersection sets list cannot be empty',
            {'index': entry_index},
            validate_mode,
            warn
        )
        return None

    return out


def _normalize_gfql_ops(
    gfql_ops: Any,
    validate_mode: ValidationMode,
    warn: bool,
    entry_index: int
) -> Optional[List[Dict[str, Any]]]:
    """
    Normalize GFQL operations to a list of JSON-serializable dicts.

    Uses _wrap_gfql_expr from collections.py as the canonical implementation,
    wrapping with error handling for validation modes.
    """
    if gfql_ops is None:
        _issue('GFQL chain is missing', {'index': entry_index}, validate_mode, warn)
        return None

    # Handle JSON string input
    if isinstance(gfql_ops, str):
        try:
            gfql_ops = json.loads(gfql_ops)
        except json.JSONDecodeError as exc:
            _issue('GFQL chain string must be JSON', {'index': entry_index, 'error': str(exc)}, validate_mode, warn)
            return None

    # Use canonical implementation from collections.py
    try:
        from graphistry.collections import _wrap_gfql_expr
        result = _wrap_gfql_expr(gfql_ops)
        ops_raw = result.get('gfql', [])
        if not isinstance(ops_raw, list):
            _issue('GFQL chain must be a list', {'index': entry_index}, validate_mode, warn)
            return None
        ops: List[Dict[str, Any]] = ops_raw
        if len(ops) == 0:
            _issue('GFQL chain is empty', {'index': entry_index}, validate_mode, warn)
            return None
        return ops
    except (TypeError, ValueError, GFQLValidationError) as exc:
        # Precise exception handling for GFQL parsing errors
        _issue(
            'Invalid GFQL operation in collection',
            {'index': entry_index, 'error': str(exc)},
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

        # Convert to plain dict for uniform handling (TypedDicts become regular dicts)
        entry_dict: Dict[str, Any] = dict(entry)

        unexpected_fields = [key for key in entry_dict.keys() if key not in _ALLOWED_COLLECTION_FIELDS_SET]
        if unexpected_fields:
            _issue(
                'Unexpected fields in collection',
                {'index': idx, 'fields': unexpected_fields},
                validate_mode,
                warn
            )

        normalized_entry = {key: entry_dict[key] for key in _ALLOWED_COLLECTION_FIELDS_ORDER if key in entry_dict}
        collection_type = normalized_entry.get('type', 'set')
        if not isinstance(collection_type, str):
            _issue(
                'Collection type must be a string',
                {'index': idx, 'value': collection_type, 'type': type(collection_type).__name__},
                validate_mode,
                warn
            )
            # str() coercion is pointless - it won't produce 'set' or 'intersection'
            # so we skip this entry in autofix mode, or fail in strict mode
            if validate_mode == 'autofix':
                continue
            return []
        collection_type = collection_type.lower()

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

        normalized_entry['type'] = collection_type

        for field in ('id', 'name', 'description'):
            _normalize_str_field(normalized_entry, field, validate_mode, warn, idx, autofix_drop=False)
        for field in ('node_color', 'edge_color'):
            _normalize_str_field(normalized_entry, field, validate_mode, warn, idx, autofix_drop=True)

        # Validate id field - required for sets (server requires it, and intersections reference by ID)
        # Note: We warn but don't auto-generate IDs - user must provide meaningful IDs
        if collection_type == 'set':
            if 'id' not in normalized_entry or normalized_entry.get('id') is None:
                _issue(
                    'Set collection requires an id field (server requires it for subgraph storage)',
                    {'index': idx},
                    validate_mode,
                    warn
                )
                # In autofix mode, skip this collection rather than generate arbitrary IDs
                # User should provide meaningful IDs they control
                if validate_mode == 'autofix':
                    continue
                else:
                    continue

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

    # Cross-validate intersection set references
    normalized = _validate_intersection_references(normalized, validate_mode, warn)

    return normalized


def _validate_intersection_references(
    collections: List[Dict[str, Any]],
    validate_mode: ValidationMode,
    warn: bool
) -> List[Dict[str, Any]]:
    """
    Validate that intersection set IDs reference actual collection IDs.

    Dangling references (e.g., sets: ["nonexistent"]) cause backend errors.
    In strict mode, raise on first issue. In autofix mode, drop invalid intersections.
    """
    # Collect all set IDs (only from 'set' type collections, not intersections)
    set_ids = {
        c.get('id')
        for c in collections
        if c.get('type') == 'set' and c.get('id')
    }

    valid_collections: List[Dict[str, Any]] = []
    for idx, collection in enumerate(collections):
        if collection.get('type') == 'intersection':
            expr = collection.get('expr', {})
            referenced_sets = expr.get('sets', [])
            missing = [sid for sid in referenced_sets if sid not in set_ids]
            if missing:
                _issue(
                    'Intersection references non-existent set IDs',
                    {'index': idx, 'missing_sets': missing, 'available_sets': list(set_ids)},
                    validate_mode,
                    warn
                )
                if validate_mode == 'autofix':
                    continue  # Drop invalid intersection
                # In strict mode, we already raised in _issue
                return []
        valid_collections.append(collection)

    return valid_collections


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
