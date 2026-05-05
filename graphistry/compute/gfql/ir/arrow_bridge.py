"""Arrow/schema bridge helpers for GFQL logical types.

T4 contract surface (#1312 under #1262 / #1046):
- ``to_arrow()`` exports a ``RowSchema`` + confidence map into a ``pyarrow.Schema``.
- ``from_arrow()`` imports a ``pyarrow.Schema`` back into ``RowSchema`` + confidence map.

The bridge is deterministic and explicit about coercion:
- ``coercion='strict'`` rejects unsupported type mappings.
- ``coercion='widen'`` falls back to representable Arrow/logical forms and records
  canonical metadata so round-trips remain stable.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Literal, Mapping, Optional, Tuple, cast

from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import EdgeRef, ListType, LogicalType, NodeRef, PathType, ScalarType

SchemaConfidence = Literal["declared", "propagated", "inferred"]
CoercionMode = Literal["strict", "widen"]

_METADATA_VERSION_KEY = b"gfql.arrow_bridge.version"
_METADATA_VERSION_VALUE = b"1"
_METADATA_LOGICAL_TYPE_KEY = b"gfql.logical_type"
_METADATA_CONFIDENCE_KEY = b"gfql.schema_confidence"

_CONFIDENCE_VALUES = {"declared", "propagated", "inferred"}

_SCALAR_KIND_TO_ARROW_FACTORY = {
    "bool": lambda pa: pa.bool_(),
    "boolean": lambda pa: pa.bool_(),
    "int8": lambda pa: pa.int8(),
    "int16": lambda pa: pa.int16(),
    "int32": lambda pa: pa.int32(),
    "int": lambda pa: pa.int64(),
    "int64": lambda pa: pa.int64(),
    "long": lambda pa: pa.int64(),
    "uint8": lambda pa: pa.uint8(),
    "uint16": lambda pa: pa.uint16(),
    "uint32": lambda pa: pa.uint32(),
    "uint64": lambda pa: pa.uint64(),
    "float": lambda pa: pa.float64(),
    "float32": lambda pa: pa.float32(),
    "float64": lambda pa: pa.float64(),
    "double": lambda pa: pa.float64(),
    "string": lambda pa: pa.large_string(),
    "str": lambda pa: pa.large_string(),
    "text": lambda pa: pa.large_string(),
    "binary": lambda pa: pa.large_binary(),
    "bytes": lambda pa: pa.large_binary(),
    "date": lambda pa: pa.date32(),
    "date32": lambda pa: pa.date32(),
    "date64": lambda pa: pa.date64(),
    "timestamp": lambda pa: pa.timestamp("us"),
    "time32": lambda pa: pa.time32("ms"),
    "time64": lambda pa: pa.time64("us"),
    "duration": lambda pa: pa.duration("us"),
    "null": lambda pa: pa.null(),
}


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError(
            "pyarrow is required for graphistry.compute.gfql.ir.arrow_bridge"
        ) from exc
    return pa


def _ensure_confidence(
    value: str,
    *,
    field_name: str,
    coercion: CoercionMode,
    fallback: SchemaConfidence,
) -> SchemaConfidence:
    if value in _CONFIDENCE_VALUES:
        return cast(SchemaConfidence, value)
    if coercion == "strict":
        raise ValueError(
            f"Unsupported schema confidence {value!r} for field {field_name!r}; "
            f"expected one of {sorted(_CONFIDENCE_VALUES)}"
        )
    return fallback


def _logical_type_to_payload(logical_type: LogicalType) -> Dict[str, Any]:
    if isinstance(logical_type, ScalarType):
        return {
            "family": "scalar",
            "kind": logical_type.kind,
            "nullable": logical_type.nullable,
        }
    if isinstance(logical_type, NodeRef):
        return {
            "family": "node",
            "labels": sorted(logical_type.labels),
        }
    if isinstance(logical_type, EdgeRef):
        return {
            "family": "edge",
            "type": logical_type.type,
            "src_label": logical_type.src_label,
            "dst_label": logical_type.dst_label,
        }
    if isinstance(logical_type, PathType):
        return {
            "family": "path",
            "min_hops": logical_type.min_hops,
            "max_hops": logical_type.max_hops,
        }
    if isinstance(logical_type, ListType):
        return {
            "family": "list",
            "element": _logical_type_to_payload(logical_type.element_type),
        }
    raise TypeError(f"Unsupported logical type: {type(logical_type)!r}")


def _payload_to_logical_type(payload: Mapping[str, Any]) -> LogicalType:
    family = str(payload.get("family", "")).strip().lower()
    if family == "scalar":
        kind = str(payload.get("kind", "unknown"))
        nullable = bool(payload.get("nullable", True))
        return ScalarType(kind=kind, nullable=nullable)
    if family == "node":
        labels = payload.get("labels", [])
        if not isinstance(labels, list):
            raise ValueError(f"Invalid node labels payload: {labels!r}")
        return NodeRef(labels=frozenset(str(v) for v in labels))
    if family == "edge":
        typ = payload.get("type")
        src_label = payload.get("src_label")
        dst_label = payload.get("dst_label")
        return EdgeRef(
            type=str(typ) if typ is not None else None,
            src_label=str(src_label) if src_label is not None else None,
            dst_label=str(dst_label) if dst_label is not None else None,
        )
    if family == "path":
        min_hops = int(payload.get("min_hops", 1))
        max_hops_raw = payload.get("max_hops", 1)
        max_hops = None if max_hops_raw is None else int(max_hops_raw)
        return PathType(min_hops=min_hops, max_hops=max_hops)
    if family == "list":
        element = payload.get("element")
        if not isinstance(element, Mapping):
            raise ValueError(f"Invalid list element payload: {element!r}")
        return ListType(element_type=_payload_to_logical_type(element))
    raise ValueError(f"Unsupported logical type family payload: {family!r}")


def _logical_type_to_arrow_type(
    logical_type: LogicalType,
    *,
    coercion: CoercionMode,
    pa: Any,
) -> Any:
    if isinstance(logical_type, ScalarType):
        factory = _SCALAR_KIND_TO_ARROW_FACTORY.get(logical_type.kind.lower())
        if factory is not None:
            return factory(pa)
        if coercion == "strict":
            raise ValueError(
                f"Unsupported ScalarType.kind for strict Arrow export: {logical_type.kind!r}"
            )
        return pa.large_string()

    if isinstance(logical_type, ListType):
        element_arrow = _logical_type_to_arrow_type(
            logical_type.element_type,
            coercion=coercion,
            pa=pa,
        )
        return pa.list_(element_arrow)

    if isinstance(logical_type, (NodeRef, EdgeRef, PathType)):
        if coercion == "strict":
            raise ValueError(
                f"Strict Arrow export does not support {type(logical_type).__name__}; "
                "use coercion='widen' for string bridge encoding"
            )
        return pa.large_string()

    raise TypeError(f"Unsupported logical type: {type(logical_type)!r}")


def _arrow_type_to_logical_type(
    arrow_type: Any,
    *,
    nullable: bool,
    coercion: CoercionMode,
    pa: Any,
) -> LogicalType:
    types = pa.types

    if types.is_null(arrow_type):
        return ScalarType(kind="null", nullable=True)
    if types.is_boolean(arrow_type):
        return ScalarType(kind="bool", nullable=nullable)
    if types.is_int8(arrow_type):
        return ScalarType(kind="int8", nullable=nullable)
    if types.is_int16(arrow_type):
        return ScalarType(kind="int16", nullable=nullable)
    if types.is_int32(arrow_type):
        return ScalarType(kind="int32", nullable=nullable)
    if types.is_int64(arrow_type):
        return ScalarType(kind="int64", nullable=nullable)
    if types.is_uint8(arrow_type):
        return ScalarType(kind="uint8", nullable=nullable)
    if types.is_uint16(arrow_type):
        return ScalarType(kind="uint16", nullable=nullable)
    if types.is_uint32(arrow_type):
        return ScalarType(kind="uint32", nullable=nullable)
    if types.is_uint64(arrow_type):
        return ScalarType(kind="uint64", nullable=nullable)
    if types.is_float32(arrow_type):
        return ScalarType(kind="float32", nullable=nullable)
    if types.is_float64(arrow_type):
        return ScalarType(kind="float64", nullable=nullable)
    if types.is_string(arrow_type) or types.is_large_string(arrow_type):
        return ScalarType(kind="string", nullable=nullable)
    if types.is_binary(arrow_type) or types.is_large_binary(arrow_type):
        return ScalarType(kind="binary", nullable=nullable)
    if types.is_date32(arrow_type):
        return ScalarType(kind="date32", nullable=nullable)
    if types.is_date64(arrow_type):
        return ScalarType(kind="date64", nullable=nullable)
    if types.is_timestamp(arrow_type):
        return ScalarType(kind="timestamp", nullable=nullable)
    if types.is_time32(arrow_type):
        return ScalarType(kind="time32", nullable=nullable)
    if types.is_time64(arrow_type):
        return ScalarType(kind="time64", nullable=nullable)
    if types.is_duration(arrow_type):
        return ScalarType(kind="duration", nullable=nullable)
    if types.is_list(arrow_type) or types.is_large_list(arrow_type):
        element = _arrow_type_to_logical_type(
            arrow_type.value_type,
            nullable=True,
            coercion=coercion,
            pa=pa,
        )
        return ListType(element_type=element)

    if coercion == "strict":
        raise ValueError(f"Unsupported Arrow type for strict import: {arrow_type}")
    return ScalarType(kind="unknown", nullable=nullable)


def to_arrow(
    schema: RowSchema,
    *,
    confidence: Optional[Mapping[str, str]] = None,
    coercion: CoercionMode = "widen",
) -> Any:
    """Export a GFQL ``RowSchema`` into ``pyarrow.Schema``.

    Metadata contract per field:
    - ``gfql.logical_type``: JSON payload for exact logical-type round-trip.
    - ``gfql.schema_confidence``: one of ``declared|propagated|inferred``.

    Nullability contract:
    - ``ScalarType`` columns preserve ``ScalarType.nullable``.
    - Structural columns (``NodeRef``/``EdgeRef``/``PathType``) are exported as
      nullable string-encoded bridge fields when coercion is ``widen``.
    """
    pa = _require_pyarrow()
    confidence = confidence or {}

    fields = []
    for column_name, logical_type in schema.columns.items():
        arrow_type = _logical_type_to_arrow_type(logical_type, coercion=coercion, pa=pa)
        field_nullable = logical_type.nullable if isinstance(logical_type, ScalarType) else True

        field_confidence = _ensure_confidence(
            str(confidence.get(column_name, "inferred")),
            field_name=column_name,
            coercion=coercion,
            fallback="inferred",
        )

        metadata_payload = {
            _METADATA_LOGICAL_TYPE_KEY: json.dumps(
                _logical_type_to_payload(logical_type), sort_keys=True
            ).encode("utf-8"),
            _METADATA_CONFIDENCE_KEY: field_confidence.encode("utf-8"),
        }
        fields.append(
            pa.field(
                column_name,
                arrow_type,
                nullable=field_nullable,
                metadata=metadata_payload,
            )
        )

    return pa.schema(fields, metadata={_METADATA_VERSION_KEY: _METADATA_VERSION_VALUE})


def from_arrow(
    schema: Any,
    *,
    coercion: CoercionMode = "widen",
    default_confidence: SchemaConfidence = "inferred",
) -> Tuple[RowSchema, Dict[str, SchemaConfidence]]:
    """Import ``pyarrow.Schema`` into ``RowSchema`` + confidence map.

    Import prefers field metadata ``gfql.logical_type`` when present for exact
    round-tripping. When missing, logical types are inferred from Arrow dtypes.
    """
    pa = _require_pyarrow()
    if not isinstance(schema, pa.Schema):
        raise TypeError(f"from_arrow expected pyarrow.Schema, got {type(schema)!r}")

    out_columns: Dict[str, LogicalType] = {}
    out_confidence: Dict[str, SchemaConfidence] = {}

    for field in schema:
        metadata = field.metadata or {}
        logical_type: LogicalType

        raw_payload = metadata.get(_METADATA_LOGICAL_TYPE_KEY)
        if raw_payload is not None:
            payload = json.loads(raw_payload.decode("utf-8"))
            logical_type = _payload_to_logical_type(payload)
        else:
            logical_type = _arrow_type_to_logical_type(
                field.type,
                nullable=bool(field.nullable),
                coercion=coercion,
                pa=pa,
            )

        raw_confidence = metadata.get(_METADATA_CONFIDENCE_KEY)
        confidence_value = default_confidence
        if raw_confidence is not None:
            confidence_value = _ensure_confidence(
                raw_confidence.decode("utf-8"),
                field_name=field.name,
                coercion=coercion,
                fallback=default_confidence,
            )

        out_columns[field.name] = logical_type
        out_confidence[field.name] = confidence_value

    return RowSchema(columns=out_columns), out_confidence


__all__ = [
    "CoercionMode",
    "SchemaConfidence",
    "to_arrow",
    "from_arrow",
]
