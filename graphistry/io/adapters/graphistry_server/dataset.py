"""
Adapter from Graphistry server dataset payloads to PlottableMetadata.

This adapter is intentionally separate from `graphistry.io.metadata`, which
handles only Plottable metadata serialization/deserialization.
"""

from typing import Any, Dict, cast
import copy

from graphistry.io.contracts.graphistry_server.dataset import (
    GRAPHISTRY_SERVER_BINDING_TO_PLOTTABLE_ENCODING_MAP,
    GraphistryServerDatasetPayload,
)
from graphistry.io.types import ComplexEncodingModes, EncodingsDict, MetadataDict, PlottableMetadata
from graphistry.validate import normalize_url_params


def _looks_like_plottable_metadata_dict(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    if any(k in value for k in ("bindings", "encodings", "style", "url_params")):
        return True
    nested_metadata = value.get("metadata")
    if isinstance(nested_metadata, dict):
        return any(k in nested_metadata for k in ("name", "description"))
    return False


def _copy_complex_encoding_modes(value: Any) -> ComplexEncodingModes:
    complex_modes: ComplexEncodingModes = {"default": {}, "current": {}}
    if isinstance(value, dict):
        complex_payload = value.get("complex")
        if isinstance(complex_payload, dict):
            default_payload = complex_payload.get("default")
            current_payload = complex_payload.get("current")
            if isinstance(default_payload, dict):
                complex_modes["default"] = copy.deepcopy(default_payload)
            if isinstance(current_payload, dict):
                complex_modes["current"] = copy.deepcopy(current_payload)
    return complex_modes


def adapt_graphistry_server_dataset_payload_to_plottable_metadata(
    server_dataset_payload: GraphistryServerDatasetPayload,
) -> PlottableMetadata:
    """Best-effort normalize Graphistry server dataset payloads into PlottableMetadata."""
    server_metadata_payload = server_dataset_payload.get("metadata")
    if _looks_like_plottable_metadata_dict(server_metadata_payload):
        return cast(PlottableMetadata, copy.deepcopy(server_metadata_payload))

    plottable_metadata: PlottableMetadata = {}

    server_bindings = server_dataset_payload.get("bindings")
    if isinstance(server_bindings, dict):
        plottable_metadata["bindings"] = cast(Dict[str, str], copy.deepcopy(server_bindings))

    server_encodings = server_dataset_payload.get("encodings")
    if isinstance(server_encodings, dict):
        plottable_metadata["encodings"] = cast(EncodingsDict, copy.deepcopy(server_encodings))

    server_node_encodings = server_dataset_payload.get("node_encodings")
    server_edge_encodings = server_dataset_payload.get("edge_encodings")
    server_node_bindings = (
        server_node_encodings.get("bindings") if isinstance(server_node_encodings, dict) else None
    )
    server_edge_bindings = (
        server_edge_encodings.get("bindings") if isinstance(server_edge_encodings, dict) else None
    )
    server_node_bindings_dict: Dict[str, str] = (
        cast(Dict[str, str], server_node_bindings) if isinstance(server_node_bindings, dict) else {}
    )
    server_edge_bindings_dict: Dict[str, str] = (
        cast(Dict[str, str], server_edge_bindings) if isinstance(server_edge_bindings, dict) else {}
    )

    core_bindings: Dict[str, str] = {}
    for binding_name in ["node", "source", "destination", "edge"]:
        if binding_name in server_node_bindings_dict and server_node_bindings_dict[binding_name] is not None:
            core_bindings[binding_name] = server_node_bindings_dict[binding_name]
        elif binding_name in server_edge_bindings_dict and server_edge_bindings_dict[binding_name] is not None:
            core_bindings[binding_name] = server_edge_bindings_dict[binding_name]
    if core_bindings and "bindings" not in plottable_metadata:
        plottable_metadata["bindings"] = core_bindings

    simple_encodings: EncodingsDict = {}
    for server_binding_key, plottable_encoding_key in (
        GRAPHISTRY_SERVER_BINDING_TO_PLOTTABLE_ENCODING_MAP.items()
    ):
        if (
            server_binding_key in server_node_bindings_dict
            and server_node_bindings_dict[server_binding_key] is not None
        ):
            simple_encodings[plottable_encoding_key] = server_node_bindings_dict[server_binding_key]  # type: ignore[literal-required]
        elif (
            server_binding_key in server_edge_bindings_dict
            and server_edge_bindings_dict[server_binding_key] is not None
        ):
            simple_encodings[plottable_encoding_key] = server_edge_bindings_dict[server_binding_key]  # type: ignore[literal-required]

    node_complex_modes = _copy_complex_encoding_modes(server_node_encodings)
    edge_complex_modes = _copy_complex_encoding_modes(server_edge_encodings)
    if (
        node_complex_modes["default"]
        or node_complex_modes["current"]
        or edge_complex_modes["default"]
        or edge_complex_modes["current"]
    ):
        simple_encodings["complex_encodings"] = {
            "node_encodings": node_complex_modes,
            "edge_encodings": edge_complex_modes,
        }
    if simple_encodings and "encodings" not in plottable_metadata:
        plottable_metadata["encodings"] = simple_encodings

    normalized_metadata_obj: MetadataDict = {}
    dataset_name = server_dataset_payload.get("name")
    dataset_description = server_dataset_payload.get("description")
    if isinstance(server_metadata_payload, dict):
        if "name" in server_metadata_payload and server_metadata_payload.get("name") is not None:
            dataset_name = server_metadata_payload.get("name")
        if "description" in server_metadata_payload and server_metadata_payload.get("description") is not None:
            dataset_description = server_metadata_payload.get("description")
    if isinstance(dataset_name, str) and dataset_name != "":
        normalized_metadata_obj["name"] = dataset_name
    if isinstance(dataset_description, str) and dataset_description != "":
        normalized_metadata_obj["description"] = dataset_description
    if normalized_metadata_obj:
        plottable_metadata["metadata"] = normalized_metadata_obj

    server_style = server_dataset_payload.get("style")
    if isinstance(server_style, dict):
        plottable_metadata["style"] = copy.deepcopy(server_style)

    server_url_params = server_dataset_payload.get("url_params")
    if isinstance(server_url_params, dict):
        plottable_metadata["url_params"] = normalize_url_params(
            server_url_params,
            validate="autofix",
            warn=False,
        )

    return plottable_metadata
