"""Generate structural JSON Schema artifacts for public tooling contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from graphistry.io.types import PLOTTABLE_SIMPLE_ENCODING_BIND_KEYS
from graphistry.models.surfaces.graphistry_frontend import (
    APPLY_ENCODINGS_REACT_KEYS,
    GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE,
    GRAPHISTRY_FRONTEND_CONTRACT_VERSION,
    GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS,
    REACT_SETTING_NAMES,
    URL_PARAM_NAMES,
)

JsonDict = Dict[str, Any]

SCHEMA_FILENAMES: Mapping[str, str] = {
    "encodings": "encodings.schema.json",
    "react_settings": "react-settings.schema.json",
    "url_params": "url-params.schema.json",
}


def _settings_value_ref() -> JsonDict:
    return {"$ref": "#/$defs/settingsValue"}


def _contract_metadata() -> JsonDict:
    return {
        "graphistry_frontend_contract_version": GRAPHISTRY_FRONTEND_CONTRACT_VERSION,
        "graphistry_frontend_contract_signature": GRAPHISTRY_FRONTEND_CONTRACT_SIGNATURE,
        "graphistry_frontend_upstream_versions": dict(GRAPHISTRY_FRONTEND_UPSTREAM_VERSIONS),
    }


def _settings_value_def() -> JsonDict:
    return {
        "description": "Structural JSON value accepted by settings validators. Deeper semantics remain in pygraphistry validators.",
        "anyOf": [
            {"type": "null"},
            {"type": "string"},
            {"type": "number"},
            {"type": "boolean"},
            {"type": "array", "items": {"$ref": "#/$defs/settingsValue"}},
            {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/settingsValue"},
            },
        ],
    }


def _object_with_known_settings(keys: tuple[str, ...], title: str, description: str) -> JsonDict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": f"https://schemas.graphistry.com/pygraphistry/{title.lower().replace(' ', '-')}.schema.json",
        "title": title,
        "description": description,
        "type": "object",
        "additionalProperties": False,
        "properties": {key: _settings_value_ref() for key in keys},
        "$defs": {"settingsValue": _settings_value_def()},
        "x-graphistry": _contract_metadata(),
    }


def url_params_schema() -> JsonDict:
    return _object_with_known_settings(
        URL_PARAM_NAMES,
        "Graphistry URL Params",
        "Structural contract for graph.html/embed URL parameters accepted by pygraphistry settings(url_params=...).",
    )


def react_settings_schema() -> JsonDict:
    schema = _object_with_known_settings(
        REACT_SETTING_NAMES,
        "Graphistry React Settings",
        "Structural contract for React-facing visualization settings. Semantic validation remains in pygraphistry validators.",
    )
    schema["x-graphistry"]["apply_encodings_react_keys"] = list(APPLY_ENCODINGS_REACT_KEYS)
    return schema


def _complex_encoding_schema() -> JsonDict:
    return {
        "type": "object",
        "description": "Structural complex encoding payload. Detailed semantic checks remain in validate_encodings().",
        "required": ["attribute", "variation"],
        "properties": {
            "attribute": {"type": "string"},
            "variation": {"enum": ["categorical", "continuous"]},
            "graphType": {"enum": ["point", "edge"]},
            "encodingType": {"type": "string"},
            "mapping": {"type": "object", "additionalProperties": True},
            "colors": {"type": "array", "items": {"type": "string"}},
            "asText": {"type": "boolean"},
            "style": {"type": "object", "additionalProperties": {"type": "number"}},
            "blendMode": {"type": "string"},
            "border": {"type": "object", "additionalProperties": True},
            "shape": {"type": "string"},
            "name": {"type": "string"},
        },
        "additionalProperties": True,
    }


def encodings_schema() -> JsonDict:
    simple_properties = {
        key: {"type": "string"}
        for key in PLOTTABLE_SIMPLE_ENCODING_BIND_KEYS
    }
    complex_modes = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "default": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/complexEncoding"},
            },
            "current": {
                "type": "object",
                "additionalProperties": {"$ref": "#/$defs/complexEncoding"},
            },
        },
    }
    node_edge_properties: JsonDict = {
        "bindings": {
            "type": "object",
            "additionalProperties": {"type": "string"},
        },
        "complex": {"$ref": "#/$defs/complexModes"},
    }
    node_edge_encodings = {
        "type": "object",
        "required": ["bindings"],
        "properties": node_edge_properties,
        "additionalProperties": True,
    }
    edge_encodings = {
        **node_edge_encodings,
        "properties": {
            **node_edge_properties,
            "bindings": {
                "type": "object",
                "required": ["source", "destination"],
                "additionalProperties": {"type": "string"},
            },
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.graphistry.com/pygraphistry/encodings.schema.json",
        "title": "Graphistry Encodings",
        "description": "Structural contract for simple and complex visualization encodings. Runtime validators remain canonical for semantic checks.",
        "oneOf": [
            {"$ref": "#/$defs/metadataEncodings"},
            {"$ref": "#/$defs/nodeEdgeEncodingsPayload"},
        ],
        "$defs": {
            "metadataEncodings": {
                "type": "object",
                "properties": {
                    **simple_properties,
                    "complex_encodings": {"$ref": "#/$defs/complexEncodings"},
                },
                "additionalProperties": False,
            },
            "nodeEdgeEncodingsPayload": {
                "type": "object",
                "required": ["node_encodings", "edge_encodings"],
                "properties": {
                    "node_encodings": node_edge_encodings,
                    "edge_encodings": edge_encodings,
                },
                "additionalProperties": False,
            },
            "complexEncodings": {
                "type": "object",
                "required": ["node_encodings", "edge_encodings"],
                "properties": {
                    "node_encodings": {"$ref": "#/$defs/complexModes"},
                    "edge_encodings": {"$ref": "#/$defs/complexModes"},
                },
                "additionalProperties": False,
            },
            "complexModes": complex_modes,
            "complexEncoding": _complex_encoding_schema(),
        },
        "x-graphistry": _contract_metadata(),
    }


def build_schemas() -> Mapping[str, JsonDict]:
    return {
        "encodings": encodings_schema(),
        "react_settings": react_settings_schema(),
        "url_params": url_params_schema(),
    }


def _render(schema: JsonDict) -> str:
    return json.dumps(schema, indent=2, sort_keys=True) + "\n"


def export_schemas(output_dir: Path, check: bool = False) -> bool:
    output_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    for name, schema in build_schemas().items():
        path = output_dir / SCHEMA_FILENAMES[name]
        rendered = _render(schema)
        if check:
            current = path.read_text(encoding="utf-8") if path.exists() else None
            if current != rendered:
                print(f"schema drift: {path}")
                ok = False
        else:
            path.write_text(rendered, encoding="utf-8")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="schemas", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    return 0 if export_schemas(args.output_dir, check=args.check) else 1


if __name__ == "__main__":
    raise SystemExit(main())
