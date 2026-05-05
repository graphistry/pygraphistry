"""
Graphistry server dataset metadata response contract.

These types model dataset metadata payloads retrieved through server-backed
routes (for example, Nexus dataset records surfaced through Forge ETL paths).

Contract-bundle note:
- If exported mapping/payload shapes change here, update
  `graphistry/io/contracts/graphistry_server/contract_version.py`.
"""

from typing import Any, Dict, TypedDict

from graphistry.io.types import BindingsDict, EncodingsDict
from graphistry.models.surfaces.graphistry_frontend.url_params import URLParamsDict


GRAPHISTRY_SERVER_BINDING_TO_PLOTTABLE_ENCODING_MAP: Dict[str, str] = {
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


class GraphistryServerNodeEdgeEncodingsPayload(TypedDict, total=False):
    bindings: BindingsDict
    complex: Dict[str, Any]


class GraphistryServerDatasetPayload(TypedDict, total=False):
    dataset_id: str
    bindings: BindingsDict
    encodings: EncodingsDict
    metadata: Dict[str, Any]
    node_encodings: GraphistryServerNodeEdgeEncodingsPayload
    edge_encodings: GraphistryServerNodeEdgeEncodingsPayload
    name: str
    description: str
    style: Dict[str, Any]
    url_params: URLParamsDict
