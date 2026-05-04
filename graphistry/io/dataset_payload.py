"""
Dataset endpoint payload coercion to PlottableMetadata.

This module is the boundary for server dataset payload contracts (wrapped,
direct, and legacy forms). It intentionally stays separate from
`graphistry.io.metadata`, which handles only Plottable metadata
serialization/deserialization.
"""
from typing import Any, Dict, cast
import copy

from graphistry.io.dataset_payload_types import DatasetLegacyPayload, SIMPLE_ENCODING_SERVER_TO_CLIENT_MAP
from graphistry.io.types import ComplexEncodingModes, EncodingsDict, MetadataDict, PlottableMetadata
from graphistry.validate import normalize_url_params


def _is_plottable_metadata_dict(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if any(k in payload for k in ('bindings', 'encodings', 'style', 'url_params')):
        return True
    if 'metadata' in payload and isinstance(payload.get('metadata'), dict):
        nested = payload['metadata']
        return any(k in nested for k in ('name', 'description'))
    return False


def _coerce_complex_mode(payload: Any) -> ComplexEncodingModes:
    out: ComplexEncodingModes = {'default': {}, 'current': {}}
    if isinstance(payload, dict):
        complex_payload = payload.get('complex')
        if isinstance(complex_payload, dict):
            default_payload = complex_payload.get('default')
            current_payload = complex_payload.get('current')
            if isinstance(default_payload, dict):
                out['default'] = copy.deepcopy(default_payload)
            if isinstance(current_payload, dict):
                out['current'] = copy.deepcopy(current_payload)
    return out


def coerce_dataset_payload_to_plottable_metadata(dataset_payload: DatasetLegacyPayload) -> PlottableMetadata:
    """Best-effort normalize dataset API payloads into PlottableMetadata."""
    metadata_payload = dataset_payload.get('metadata')
    if _is_plottable_metadata_dict(metadata_payload):
        return cast(PlottableMetadata, copy.deepcopy(metadata_payload))

    out: PlottableMetadata = {}

    direct_bindings = dataset_payload.get('bindings')
    if isinstance(direct_bindings, dict):
        out['bindings'] = cast(Dict[str, str], copy.deepcopy(direct_bindings))

    direct_encodings = dataset_payload.get('encodings')
    if isinstance(direct_encodings, dict):
        out['encodings'] = cast(EncodingsDict, copy.deepcopy(direct_encodings))

    node_encodings = dataset_payload.get('node_encodings')
    edge_encodings = dataset_payload.get('edge_encodings')
    node_bindings = node_encodings.get('bindings') if isinstance(node_encodings, dict) else None
    edge_bindings = edge_encodings.get('bindings') if isinstance(edge_encodings, dict) else None
    node_bindings_dict: Dict[str, str] = cast(Dict[str, str], node_bindings) if isinstance(node_bindings, dict) else {}
    edge_bindings_dict: Dict[str, str] = cast(Dict[str, str], edge_bindings) if isinstance(edge_bindings, dict) else {}

    bindings: Dict[str, str] = {}
    for key in ['node', 'source', 'destination', 'edge']:
        if key in node_bindings_dict and node_bindings_dict[key] is not None:
            bindings[key] = node_bindings_dict[key]
        elif key in edge_bindings_dict and edge_bindings_dict[key] is not None:
            bindings[key] = edge_bindings_dict[key]
    if bindings and 'bindings' not in out:
        out['bindings'] = bindings

    encodings: EncodingsDict = {}
    for server_key, encoding_key in SIMPLE_ENCODING_SERVER_TO_CLIENT_MAP.items():
        if server_key in node_bindings_dict and node_bindings_dict[server_key] is not None:
            encodings[encoding_key] = node_bindings_dict[server_key]  # type: ignore[literal-required]
        elif server_key in edge_bindings_dict and edge_bindings_dict[server_key] is not None:
            encodings[encoding_key] = edge_bindings_dict[server_key]  # type: ignore[literal-required]

    node_complex = _coerce_complex_mode(node_encodings)
    edge_complex = _coerce_complex_mode(edge_encodings)
    if node_complex['default'] or node_complex['current'] or edge_complex['default'] or edge_complex['current']:
        encodings['complex_encodings'] = {
            'node_encodings': node_complex,
            'edge_encodings': edge_complex,
        }
    if encodings and 'encodings' not in out:
        out['encodings'] = encodings

    metadata_obj: MetadataDict = {}
    name = dataset_payload.get('name')
    description = dataset_payload.get('description')
    if isinstance(metadata_payload, dict):
        if 'name' in metadata_payload and metadata_payload.get('name') is not None:
            name = metadata_payload.get('name')
        if 'description' in metadata_payload and metadata_payload.get('description') is not None:
            description = metadata_payload.get('description')
    if isinstance(name, str) and name != '':
        metadata_obj['name'] = name
    if isinstance(description, str) and description != '':
        metadata_obj['description'] = description
    if metadata_obj:
        out['metadata'] = metadata_obj

    style = dataset_payload.get('style')
    if isinstance(style, dict):
        out['style'] = copy.deepcopy(style)

    url_params = dataset_payload.get('url_params')
    if isinstance(url_params, dict):
        out['url_params'] = normalize_url_params(url_params, validate="autofix", warn=False)

    return out
