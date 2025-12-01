"""
Metadata serialization and deserialization for Plottable objects.

This module provides a unified interface for converting Plottable metadata
to/from JSON format for server communication (uploads, GFQL responses, etc.).

Format mirrors the arrow uploader metadata structure:
- bindings: node, source, destination, edge column names
- encodings: simple (point_color, etc.) and complex encodings
- metadata: name, description
- style: visualization styles
"""
from typing import Any, Dict, List, TYPE_CHECKING
import copy
import warnings

from graphistry.io.types import (
    ComplexEncodingsDict,
    EncodingsDict,
    MetadataDict,
    NodeEdgeEncodingsDict,
    PlottableMetadata,
)

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable


def serialize_bindings(g: 'Plottable', field_mapping: List[List[str]]) -> Dict[str, str]:
    """Extract bindings from Plottable using field mapping.

    :param g: Plottable object
    :param field_mapping: List of (plottable_attr, json_key) tuples
    :return: Dictionary of bindings (column name mappings)
    """
    out: Dict[str, str] = {}
    for old_field_name, new_field_name in field_mapping:
        try:
            val = getattr(g, old_field_name)
            if val is None:
                continue
            else:
                out[new_field_name] = val
        except AttributeError:
            continue
    return out


def serialize_node_bindings(g: 'Plottable') -> Dict[str, str]:
    """Extract node bindings from Plottable.

    Maps internal Plottable attributes (_node, _point_color, etc.)
    to server format (node, node_color, etc.).

    :param g: Plottable object
    :return: Dictionary of node bindings (column name mappings)

    **Example**

    ::

        bindings = serialize_node_bindings(g)
        # {'node': 'id', 'node_color': 'category', ...}
    """
    return serialize_bindings(g, [
        ['_node', 'node'],
        ['_point_color', 'node_color'],
        ['_point_label', 'node_label'],
        ['_point_opacity', 'node_opacity'],
        ['_point_size', 'node_size'],
        ['_point_title', 'node_title'],
        ['_point_weight', 'node_weight'],
        ['_point_icon', 'node_icon'],
        ['_point_x', 'node_x'],
        ['_point_y', 'node_y'],
        ['_point_longitude', 'node_longitude'],
        ['_point_latitude', 'node_latitude']
    ])


def serialize_edge_bindings(g: 'Plottable') -> Dict[str, str]:
    """Extract edge bindings from Plottable.

    Maps internal Plottable attributes (_source, _edge_color, etc.)
    to server format (source, edge_color, etc.).

    :param g: Plottable object
    :return: Dictionary of edge bindings (column name mappings)

    **Example**

    ::

        bindings = serialize_edge_bindings(g)
        # {'source': 'src', 'destination': 'dst', ...}
    """
    return serialize_bindings(g, [
        ['_source', 'source'],
        ['_destination', 'destination'],
        ['_edge_color', 'edge_color'],
        ['_edge_source_color', 'edge_source_color'],
        ['_edge_destination_color', 'edge_destination_color'],
        ['_edge_label', 'edge_label'],
        ['_edge_opacity', 'edge_opacity'],
        ['_edge_size', 'edge_size'],
        ['_edge_title', 'edge_title'],
        ['_edge_weight', 'edge_weight'],
        ['_edge_icon', 'edge_icon']
    ])


def serialize_node_encodings(g: 'Plottable') -> NodeEdgeEncodingsDict:
    """Extract node encodings (bindings + complex) from Plottable.

    :param g: Plottable object
    :return: Dictionary with 'bindings' and optionally 'complex' keys

    **Example**

    ::

        encodings = serialize_node_encodings(g)
        # {'bindings': {...}, 'complex': {'default': {...}, 'current': {...}}}
    """
    encodings: NodeEdgeEncodingsDict = {
        'bindings': serialize_node_bindings(g)
    }
    # Check current mode
    if len(g._complex_encodings['node_encodings']['current'].keys()) > 0:
        if not ('complex' in encodings):
            encodings['complex'] = {}
        encodings['complex']['current'] = g._complex_encodings['node_encodings']['current']
    # Check default mode
    if len(g._complex_encodings['node_encodings']['default'].keys()) > 0:
        if not ('complex' in encodings):
            encodings['complex'] = {}
        encodings['complex']['default'] = g._complex_encodings['node_encodings']['default']
    return encodings


def serialize_edge_encodings(g: 'Plottable') -> NodeEdgeEncodingsDict:
    """Extract edge encodings (bindings + complex) from Plottable.

    :param g: Plottable object
    :return: Dictionary with 'bindings' and optionally 'complex' keys

    **Example**

    ::

        encodings = serialize_edge_encodings(g)
        # {'bindings': {...}, 'complex': {'default': {...}, 'current': {...}}}
    """
    encodings: NodeEdgeEncodingsDict = {
        'bindings': serialize_edge_bindings(g)
    }
    # Check current mode
    if len(g._complex_encodings['edge_encodings']['current'].keys()) > 0:
        if not ('complex' in encodings):
            encodings['complex'] = {}
        encodings['complex']['current'] = g._complex_encodings['edge_encodings']['current']
    # Check default mode
    if len(g._complex_encodings['edge_encodings']['default'].keys()) > 0:
        if not ('complex' in encodings):
            encodings['complex'] = {}
        encodings['complex']['default'] = g._complex_encodings['edge_encodings']['default']
    return encodings


def serialize_plottable_metadata(g: 'Plottable') -> PlottableMetadata:
    """Serialize complete Plottable metadata to JSON format.

    Extracts all metadata that should be sent to the server during uploads
    or that the server might return after GFQL operations.

    :param g: Plottable object
    :return: PlottableMetadata with bindings, encodings, metadata, style

    **Example**

    ::

        metadata = serialize_plottable_metadata(g)
        # {
        #   'bindings': {'node': 'id', 'source': 'src', ...},
        #   'encodings': {'point_color': 'category', 'complex_encodings': {...}},
        #   'metadata': {'name': 'My Graph', 'description': '...'},
        #   'style': {'bg': {'color': '#000000'}}
        # }
    """
    # Collect all bindings (both node and edge)
    bindings: Dict[str, str] = {}
    bindings.update(serialize_node_bindings(g))
    bindings.update(serialize_edge_bindings(g))

    # Collect all simple encodings
    encodings: EncodingsDict = {}
    node_bindings: Dict[str, str] = serialize_node_bindings(g)
    edge_bindings: Dict[str, str] = serialize_edge_bindings(g)

    # Map server format back to simple encoding names
    simple_encoding_map: Dict[str, str] = {
        'node_color': 'point_color',
        'node_size': 'point_size',
        'node_title': 'point_title',
        'node_label': 'point_label',
        'node_icon': 'point_icon',
        'node_opacity': 'point_opacity',
        'node_x': 'point_x',
        'node_y': 'point_y',
        'edge_color': 'edge_color',
        'edge_size': 'edge_size',
        'edge_title': 'edge_title',
        'edge_label': 'edge_label',
        'edge_icon': 'edge_icon',
        'edge_opacity': 'edge_opacity',
        'edge_source_color': 'edge_source_color',
        'edge_destination_color': 'edge_destination_color',
        'edge_weight': 'edge_weight'
    }

    for server_key, encoding_key in simple_encoding_map.items():
        if server_key in node_bindings:
            encodings[encoding_key] = node_bindings[server_key]  # type: ignore[literal-required]
        elif server_key in edge_bindings:
            encodings[encoding_key] = edge_bindings[server_key]  # type: ignore[literal-required]

    # Add complex encodings
    if hasattr(g, '_complex_encodings') and g._complex_encodings:
        complex_encs: ComplexEncodingsDict = g._complex_encodings
        encodings['complex_encodings'] = complex_encs

    # Build metadata
    metadata_obj: MetadataDict = {}
    if hasattr(g, '_name') and g._name:
        metadata_obj['name'] = g._name
    if hasattr(g, '_description') and g._description:
        metadata_obj['description'] = g._description

    # Build style
    style: Dict[str, Any] = {}
    if hasattr(g, '_style') and g._style:
        style = g._style

    result: PlottableMetadata = {}
    if bindings:
        result['bindings'] = bindings
    if encodings:
        result['encodings'] = encodings
    if metadata_obj:
        result['metadata'] = metadata_obj
    if style:
        result['style'] = style

    return result


def deserialize_plottable_metadata(metadata: PlottableMetadata, g: 'Plottable') -> 'Plottable':
    """Deserialize JSON metadata back into Plottable.

    Applies metadata from server responses (e.g., after GFQL operations)
    back into the Plottable object. Handles bindings, encodings, metadata,
    and style gracefully with error handling.

    :param metadata: Server metadata (PlottableMetadata structure)
    :param g: Plottable object to hydrate
    :return: New Plottable with hydrated metadata

    **Example**

    ::

        # Server returned metadata with updated bindings
        metadata = {
            'bindings': {'source': 'umap_src', 'destination': 'umap_dst'},
            'encodings': {'point_color': 'umap_cluster'}
        }
        g2 = deserialize_plottable_metadata(metadata, g1)
        # g2._source == 'umap_src', g2._point_color == 'umap_cluster'
    """
    res: 'Plottable' = g

    # Hydrate bindings
    if 'bindings' in metadata:
        try:
            bindings: Dict[str, str] = metadata['bindings']
            if isinstance(bindings, dict):
                bind_kwargs: Dict[str, str] = {}
                if 'node' in bindings and bindings['node'] is not None:
                    bind_kwargs['node'] = bindings['node']
                if 'source' in bindings and bindings['source'] is not None:
                    bind_kwargs['source'] = bindings['source']
                if 'destination' in bindings and bindings['destination'] is not None:
                    bind_kwargs['destination'] = bindings['destination']
                if 'edge' in bindings and bindings['edge'] is not None:
                    bind_kwargs['edge'] = bindings['edge']

                if bind_kwargs:
                    res = res.bind(**bind_kwargs)
        except Exception as e:
            warnings.warn(f"Failed to hydrate bindings from metadata: {e}", UserWarning, stacklevel=2)

    # Hydrate simple encodings
    if 'encodings' in metadata:
        try:
            encodings: EncodingsDict = metadata['encodings']
            if isinstance(encodings, dict):
                encode_kwargs: Dict[str, str] = {}

                simple_encoding_keys: List[str] = [
                    'point_color', 'point_size', 'point_title', 'point_label',
                    'point_icon', 'point_badge', 'point_opacity', 'point_x', 'point_y',
                    'edge_color', 'edge_size', 'edge_title', 'edge_label',
                    'edge_icon', 'edge_badge', 'edge_opacity', 'edge_source_color',
                    'edge_destination_color', 'edge_weight'
                ]

                for key in simple_encoding_keys:
                    if key in encodings and encodings.get(key) is not None:  # type: ignore[misc]
                        encode_kwargs[key] = encodings[key]  # type: ignore[literal-required, typeddict-item]

                if encode_kwargs:
                    res = res.bind(**encode_kwargs)

                # Complex encodings (direct assignment)
                if 'complex_encodings' in encodings:
                    res = copy.copy(res)
                    complex_encs: ComplexEncodingsDict = encodings['complex_encodings']  # type: ignore[typeddict-item]
                    res._complex_encodings = complex_encs

        except Exception as e:
            warnings.warn(f"Failed to hydrate encodings from metadata: {e}", UserWarning, stacklevel=2)

    if 'metadata' in metadata:
        try:
            meta: MetadataDict = metadata['metadata']
            if isinstance(meta, dict):
                if 'name' in meta and meta.get('name') is not None:
                    res = res.name(meta['name'])  # type: ignore[typeddict-item]
                if 'description' in meta and meta.get('description') is not None:
                    res = res.description(meta['description'])  # type: ignore[typeddict-item]
        except Exception as e:
            warnings.warn(f"Failed to hydrate name/description from metadata: {e}", UserWarning, stacklevel=2)

    if 'style' in metadata:
        try:
            style: Dict[str, Any] = metadata['style']
            if isinstance(style, dict):
                res = res.style(**style)
        except Exception as e:
            warnings.warn(f"Failed to hydrate style from metadata: {e}", UserWarning, stacklevel=2)

    return res
