"""
Graphistry server dataset response contract.

These types model dataset metadata payloads retrieved through server-backed
routes (for example, Nexus dataset records surfaced through Forge ETL paths).
"""

from typing import Any, Dict, TypedDict

from graphistry.io.types import BindingsDict, EncodingsDict
from graphistry.validate import URLParamsDict


GRAPHISTRY_SERVER_SIMPLE_ENCODING_BINDING_MAP: Dict[str, str] = {
    "node_color": "point_color",
    "node_size": "point_size",
    "node_title": "point_title",
    "node_label": "point_label",
    "node_icon": "point_icon",
    "node_opacity": "point_opacity",
    "node_x": "point_x",
    "node_y": "point_y",
    "edge_color": "edge_color",
    "edge_size": "edge_size",
    "edge_title": "edge_title",
    "edge_label": "edge_label",
    "edge_icon": "edge_icon",
    "edge_opacity": "edge_opacity",
    "edge_source_color": "edge_source_color",
    "edge_destination_color": "edge_destination_color",
    "edge_weight": "edge_weight",
}


class GraphistryDatasetNodeEdgeEncodingPayload(TypedDict, total=False):
    bindings: BindingsDict
    complex: Dict[str, Any]


class GraphistryDatasetResponsePayload(TypedDict, total=False):
    dataset_id: str
    bindings: BindingsDict
    encodings: EncodingsDict
    metadata: Dict[str, Any]
    node_encodings: GraphistryDatasetNodeEdgeEncodingPayload
    edge_encodings: GraphistryDatasetNodeEdgeEncodingPayload
    name: str
    description: str
    style: Dict[str, Any]
    url_params: URLParamsDict
