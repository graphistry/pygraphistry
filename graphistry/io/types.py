"""
Type definitions for Plottable metadata serialization/deserialization.

This module contains TypedDict definitions that lock in the exact structure
of metadata JSON used for server communication (uploads, GFQL responses, etc.).
"""
from typing import Any, Dict, Tuple, TypedDict

from graphistry.validate import URLParamsDict


# Complex Encodings Structure
class ComplexEncodingModes(TypedDict):
    """Current and default modes for complex encodings.

    Both keys are always present (may be empty dicts).

    :field default: Default encoding settings
    :field current: Current encoding settings (may override defaults)
    """
    default: Dict[str, Any]
    current: Dict[str, Any]


class ComplexEncodingsDict(TypedDict):
    """Complete complex encodings structure.

    Both keys are always present.

    :field node_encodings: Complex encodings for node/point visualizations
    :field edge_encodings: Complex encodings for edge visualizations
    """
    node_encodings: ComplexEncodingModes
    edge_encodings: ComplexEncodingModes


# Simple Metadata Structures
class MetadataDict(TypedDict, total=False):
    """Name and description metadata.

    All fields are optional - only present fields are included in serialization.

    :field name: Graph name
    :field description: Graph description
    """
    name: str
    description: str


class BindingsDict(TypedDict, total=False):
    node: str
    source: str
    destination: str
    edge: str
    node_color: str
    node_size: str
    node_title: str
    node_label: str
    node_icon: str
    node_opacity: str
    node_weight: str
    node_x: str
    node_y: str
    node_longitude: str
    node_latitude: str
    edge_color: str
    edge_size: str
    edge_title: str
    edge_label: str
    edge_icon: str
    edge_opacity: str
    edge_source_color: str
    edge_destination_color: str
    edge_weight: str


class EncodingsDict(TypedDict, total=False):
    """Visual encodings for nodes and edges.

    All fields are optional - only present fields are included in serialization.
    Maps column names to encoding attributes (colors, sizes, labels, etc.).

    :field point_color: Node color column name
    :field point_size: Node size column name
    :field point_title: Node title column name
    :field point_label: Node label column name
    :field point_icon: Node icon column name
    :field point_opacity: Node opacity column name
    :field point_x: Node X position column name
    :field point_y: Node Y position column name
    :field edge_color: Edge color column name
    :field edge_size: Edge size column name
    :field edge_title: Edge title column name
    :field edge_label: Edge label column name
    :field edge_icon: Edge icon column name
    :field edge_opacity: Edge opacity column name
    :field edge_source_color: Edge source color column name
    :field edge_destination_color: Edge destination color column name
    :field edge_weight: Edge weight column name
    :field complex_encodings: Complex encoding configurations
    """
    # Node encodings
    point_color: str
    point_size: str
    point_title: str
    point_label: str
    point_icon: str
    point_badge: str
    point_opacity: str
    point_x: str
    point_y: str
    # Edge encodings
    edge_color: str
    edge_size: str
    edge_title: str
    edge_label: str
    edge_icon: str
    edge_badge: str
    edge_opacity: str
    edge_source_color: str
    edge_destination_color: str
    edge_weight: str
    # Complex encodings (nested structure)
    complex_encodings: ComplexEncodingsDict


SIMPLE_ENCODING_CLIENT_KEYS: Tuple[str, ...] = (
    'point_color',
    'point_size',
    'point_title',
    'point_label',
    'point_icon',
    'point_badge',
    'point_opacity',
    'point_x',
    'point_y',
    'edge_color',
    'edge_size',
    'edge_title',
    'edge_label',
    'edge_icon',
    'edge_badge',
    'edge_opacity',
    'edge_source_color',
    'edge_destination_color',
    'edge_weight',
)


class NodeEdgeEncodingsDict(TypedDict, total=False):
    """Intermediate encoding structure with bindings and complex encodings.

    Used internally by serialize_node_encodings() and serialize_edge_encodings().

    :field bindings: Column name mappings for the entity (node or edge)
    :field complex: Complex encoding modes (default and/or current)
    """
    bindings: Dict[str, str]
    complex: Dict[str, Any]  # ComplexEncodingModes with dynamic keys


class PlottableMetadata(TypedDict, total=False):
    """Complete Plottable metadata structure for JSON serialization.

    All fields are optional - only present fields are included in serialization.
    This structure mirrors the arrow uploader format and is used for both
    upload metadata and GFQL response metadata.

    :field bindings: Column name mappings (node, source, destination, edge)
    :field encodings: Visual encoding mappings (colors, sizes, labels, complex)
    :field metadata: Graph metadata (name, description)
    :field style: Visualization styles (background, layout, etc.)
    :field url_params: Visualization URL parameter defaults

    **Example**

    ::

        metadata: PlottableMetadata = {
            'bindings': {'node': 'id', 'source': 'src', 'destination': 'dst'},
            'encodings': {
                'point_color': 'category',
                'complex_encodings': {
                    'node_encodings': {
                        'default': {'pointColorEncoding': {...}},
                        'current': {}
                    },
                    'edge_encodings': {
                        'default': {},
                        'current': {}
                    }
                }
            },
            'metadata': {'name': 'My Graph', 'description': 'A sample graph'},
            'style': {'bg': {'color': '#000000'}}
        }
    """
    bindings: Dict[str, str]
    encodings: EncodingsDict
    metadata: MetadataDict
    style: Dict[str, Any]
    url_params: URLParamsDict
