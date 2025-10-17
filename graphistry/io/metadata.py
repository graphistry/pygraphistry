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
from typing import Any, Dict, Optional
import copy


def serialize_bindings(g, field_mapping) -> Dict[str, Any]:
    """Extract bindings from Plottable using field mapping.

    :param g: Plottable object
    :param field_mapping: List of (plottable_attr, json_key) tuples
    :return: Dictionary of bindings
    """
    out = {}
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


def serialize_node_bindings(g) -> Dict[str, Any]:
    """Extract node bindings from Plottable.

    Maps internal Plottable attributes (_node, _point_color, etc.)
    to server format (node, node_color, etc.).

    :param g: Plottable object
    :return: Dictionary of node bindings

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
        ['_point_y', 'node_y']
    ])


def serialize_edge_bindings(g) -> Dict[str, Any]:
    """Extract edge bindings from Plottable.

    Maps internal Plottable attributes (_source, _edge_color, etc.)
    to server format (source, edge_color, etc.).

    :param g: Plottable object
    :return: Dictionary of edge bindings

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


def serialize_node_encodings(g) -> Dict[str, Any]:
    """Extract node encodings (bindings + complex) from Plottable.

    :param g: Plottable object
    :return: Dictionary with 'bindings' and optionally 'complex' keys

    **Example**

    ::

        encodings = serialize_node_encodings(g)
        # {'bindings': {...}, 'complex': {'default': {...}, 'current': {...}}}
    """
    encodings = {
        'bindings': serialize_node_bindings(g)
    }
    for mode in ['current', 'default']:
        if len(g._complex_encodings['node_encodings'][mode].keys()) > 0:
            if not ('complex' in encodings):
                encodings['complex'] = {}
            encodings['complex'][mode] = g._complex_encodings['node_encodings'][mode]
    return encodings


def serialize_edge_encodings(g) -> Dict[str, Any]:
    """Extract edge encodings (bindings + complex) from Plottable.

    :param g: Plottable object
    :return: Dictionary with 'bindings' and optionally 'complex' keys

    **Example**

    ::

        encodings = serialize_edge_encodings(g)
        # {'bindings': {...}, 'complex': {'default': {...}, 'current': {...}}}
    """
    encodings = {
        'bindings': serialize_edge_bindings(g)
    }
    for mode in ['current', 'default']:
        if len(g._complex_encodings['edge_encodings'][mode].keys()) > 0:
            if not ('complex' in encodings):
                encodings['complex'] = {}
            encodings['complex'][mode] = g._complex_encodings['edge_encodings'][mode]
    return encodings


def serialize_plottable_metadata(g) -> Dict[str, Any]:
    """Serialize complete Plottable metadata to JSON format.

    Extracts all metadata that should be sent to the server during uploads
    or that the server might return after GFQL operations.

    :param g: Plottable object
    :return: Dictionary with bindings, encodings, metadata, style

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
    bindings = {}
    bindings.update(serialize_node_bindings(g))
    bindings.update(serialize_edge_bindings(g))

    # Collect all simple encodings
    encodings = {}
    node_bindings = serialize_node_bindings(g)
    edge_bindings = serialize_edge_bindings(g)

    # Map server format back to simple encoding names
    # Node bindings: node_color -> point_color, etc.
    simple_encoding_map = {
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
            encodings[encoding_key] = node_bindings[server_key]
        elif server_key in edge_bindings:
            encodings[encoding_key] = edge_bindings[server_key]

    # Add complex encodings
    if hasattr(g, '_complex_encodings') and g._complex_encodings:
        encodings['complex_encodings'] = g._complex_encodings

    # Build metadata
    metadata_obj = {}
    if hasattr(g, '_name') and g._name:
        metadata_obj['name'] = g._name
    if hasattr(g, '_description') and g._description:
        metadata_obj['description'] = g._description

    # Build style
    style = {}
    if hasattr(g, '_style') and g._style:
        style = g._style

    result = {}
    if bindings:
        result['bindings'] = bindings
    if encodings:
        result['encodings'] = encodings
    if metadata_obj:
        result['metadata'] = metadata_obj
    if style:
        result['style'] = style

    return result


def deserialize_plottable_metadata(metadata: dict, g: 'Plottable') -> 'Plottable':
    """Deserialize JSON metadata back into Plottable.

    Applies metadata from server responses (e.g., after GFQL operations)
    back into the Plottable object. Handles bindings, encodings, metadata,
    and style gracefully with error handling.

    :param metadata: Server metadata dictionary
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
    import warnings

    res = g

    # Hydrate bindings
    if 'bindings' in metadata:
        try:
            bindings = metadata['bindings']
            if isinstance(bindings, dict):
                bind_kwargs = {}
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
            encodings = metadata['encodings']
            if isinstance(encodings, dict):
                encode_kwargs = {}

                # Simple encodings that bind() supports
                simple_encoding_keys = [
                    'point_color', 'point_size', 'point_title', 'point_label',
                    'point_icon', 'point_badge', 'point_opacity', 'point_x', 'point_y',
                    'edge_color', 'edge_size', 'edge_title', 'edge_label',
                    'edge_icon', 'edge_badge', 'edge_opacity', 'edge_source_color',
                    'edge_destination_color', 'edge_weight'
                ]

                for key in simple_encoding_keys:
                    if key in encodings and encodings[key] is not None:
                        encode_kwargs[key] = encodings[key]

                if encode_kwargs:
                    res = res.bind(**encode_kwargs)

                # Complex encodings (direct assignment)
                if 'complex_encodings' in encodings:
                    res = copy.copy(res)
                    res._complex_encodings = encodings['complex_encodings']

        except Exception as e:
            warnings.warn(f"Failed to hydrate encodings from metadata: {e}", UserWarning, stacklevel=2)

    # Hydrate metadata (name, description)
    if 'metadata' in metadata:
        try:
            meta = metadata['metadata']
            if isinstance(meta, dict):
                if 'name' in meta and meta['name'] is not None:
                    res = res.name(meta['name'])
                if 'description' in meta and meta['description'] is not None:
                    res = res.description(meta['description'])
        except Exception as e:
            warnings.warn(f"Failed to hydrate name/description from metadata: {e}", UserWarning, stacklevel=2)

    # Hydrate style
    if 'style' in metadata:
        try:
            style = metadata['style']
            if isinstance(style, dict):
                res = res.style(**style)
        except Exception as e:
            warnings.warn(f"Failed to hydrate style from metadata: {e}", UserWarning, stacklevel=2)

    return res
