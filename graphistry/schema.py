"""Experimental public declarative graph schema model for GFQL validation.

The import path is public, but this schema surface is still experimental while
inference, coercion, remote transport, and planner use are developed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple, Union

from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog
from graphistry.compute.gfql.ir.arrow_bridge import CoercionMode, from_arrow, to_arrow
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import EdgeRef, ListType, LogicalType, NodeRef, PathType, ScalarType


NodeRefInput = Union["NodeType", str, Iterable[str]]
PropertySchemaInput = Union[Mapping[str, Any], RowSchema, Any]


def _is_arrow_schema(value: Any) -> bool:
    try:
        import pyarrow as pa
    except ImportError:
        return False
    return isinstance(value, pa.Schema)


def _is_arrow_field(value: Any) -> bool:
    try:
        import pyarrow as pa
    except ImportError:
        return False
    return isinstance(value, pa.Field)


def _is_arrow_data_type(value: Any) -> bool:
    try:
        import pyarrow as pa
    except ImportError:
        return False
    return isinstance(value, pa.DataType)


def _logical_type_from_annotation(value: Any) -> LogicalType:
    if isinstance(value, (EdgeRef, ListType, NodeRef, PathType, ScalarType)):
        return value

    if value is str:
        return ScalarType("string")
    if value is int:
        return ScalarType("int64")
    if value is float:
        return ScalarType("float64")
    if value is bool:
        return ScalarType("bool")
    if value is bytes:
        return ScalarType("binary")

    if isinstance(value, str):
        return ScalarType(value)

    if _is_arrow_field(value):
        import pyarrow as pa

        row_schema, _ = from_arrow(pa.schema([value]), default_confidence="declared")
        return row_schema.columns[str(value.name)]

    if _is_arrow_data_type(value):
        import pyarrow as pa

        row_schema, _ = from_arrow(
            pa.schema([pa.field("__property__", value)]),
            default_confidence="declared",
        )
        return row_schema.columns["__property__"]

    return ScalarType("unknown")


def _normalize_properties(properties: Optional[PropertySchemaInput]) -> Dict[str, LogicalType]:
    if properties is None:
        return {}

    if isinstance(properties, RowSchema):
        return {str(name): logical_type for name, logical_type in properties.columns.items()}

    if _is_arrow_schema(properties):
        row_schema, _ = from_arrow(properties, default_confidence="declared")
        return {str(name): logical_type for name, logical_type in row_schema.columns.items()}

    if not isinstance(properties, Mapping):
        raise TypeError(
            "properties must be a mapping, pyarrow.Schema, or GFQL RowSchema; "
            f"got {type(properties)!r}"
        )

    return {
        str(name): _logical_type_from_annotation(value)
        for name, value in dict(properties).items()
    }


def _label_columns(labels: Iterable[str]) -> Dict[str, LogicalType]:
    return {f"label__{label}": ScalarType("bool") for label in labels}


def _normalize_labels(name: str, labels: Optional[Iterable[str]]) -> FrozenSet[str]:
    raw = (name,) if labels is None else tuple(labels)
    return frozenset(str(label) for label in raw if str(label))


def _labels_from_node_ref(value: NodeRefInput) -> FrozenSet[str]:
    if isinstance(value, NodeType):
        return value.labels
    if isinstance(value, str):
        return frozenset((value,))
    return frozenset(str(label) for label in value if str(label))


@dataclass(frozen=True)
class NodeType:
    """Experimental declarative node contract for GFQL schema validation.

    ``name`` is the stable user-facing type name. ``labels`` defaults to
    ``(name,)`` and maps to GFQL's existing ``label__<Label>`` convention.
    """

    name: str
    properties: Mapping[str, LogicalType] = field(default_factory=dict)
    labels: FrozenSet[str] = field(default_factory=frozenset)

    def __init__(
        self,
        name: str,
        properties: Optional[PropertySchemaInput] = None,
        labels: Optional[Iterable[str]] = None,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "properties", _normalize_properties(properties))
        object.__setattr__(self, "labels", _normalize_labels(str(name), labels))

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "properties", _normalize_properties(self.properties))
        object.__setattr__(self, "labels", _normalize_labels(self.name, self.labels))

    @property
    def columns(self) -> FrozenSet[str]:
        """Columns admitted for this node type, including label columns."""
        property_columns = frozenset(str(name) for name in self.properties.keys())
        label_columns = frozenset(f"label__{label}" for label in self.labels)
        return property_columns | label_columns

    def to_row_schema(self, *, include_labels: bool = True) -> RowSchema:
        """Export this node contract as GFQL's Arrow-bridge row schema."""
        columns = dict(self.properties)
        if include_labels:
            columns.update(_label_columns(self.labels))
        return RowSchema(columns=columns)

    def to_arrow(
        self,
        *,
        include_labels: bool = True,
        coercion: CoercionMode = "widen",
    ) -> Any:
        """Export this node contract as a ``pyarrow.Schema``."""
        return to_arrow(self.to_row_schema(include_labels=include_labels), coercion=coercion)


@dataclass(frozen=True)
class EdgeTopology:
    """Experimental source/destination-label contract for a relationship type."""

    relationship_type: str
    source_labels: FrozenSet[str]
    destination_labels: FrozenSet[str]

    def as_metadata(self) -> Dict[str, object]:
        return {
            "relationship_type": self.relationship_type,
            "source_labels": tuple(sorted(self.source_labels)),
            "destination_labels": tuple(sorted(self.destination_labels)),
        }


@dataclass(frozen=True)
class EdgeType:
    """Experimental declarative edge contract with topology constraints."""

    name: str
    source: FrozenSet[str]
    destination: FrozenSet[str]
    properties: Mapping[str, LogicalType] = field(default_factory=dict)

    def __init__(
        self,
        name: str,
        source: NodeRefInput,
        destination: NodeRefInput,
        properties: Optional[PropertySchemaInput] = None,
    ) -> None:
        object.__setattr__(self, "name", str(name))
        object.__setattr__(self, "properties", _normalize_properties(properties))
        object.__setattr__(self, "source", _labels_from_node_ref(source))
        object.__setattr__(self, "destination", _labels_from_node_ref(destination))

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", str(self.name))
        object.__setattr__(self, "properties", _normalize_properties(self.properties))
        object.__setattr__(self, "source", _labels_from_node_ref(self.source))
        object.__setattr__(self, "destination", _labels_from_node_ref(self.destination))

    @property
    def columns(self) -> FrozenSet[str]:
        """Columns admitted for this edge type, including relationship label."""
        property_columns = frozenset(str(name) for name in self.properties.keys())
        return property_columns | frozenset((f"label__{self.name}",))

    @property
    def topology(self) -> EdgeTopology:
        return EdgeTopology(
            relationship_type=self.name,
            source_labels=self.source,
            destination_labels=self.destination,
        )

    def to_row_schema(self, *, include_type_label: bool = True) -> RowSchema:
        """Export this edge contract as GFQL's Arrow-bridge row schema."""
        columns = dict(self.properties)
        if include_type_label:
            columns[f"label__{self.name}"] = ScalarType("bool")
        return RowSchema(columns=columns)

    def to_arrow(
        self,
        *,
        include_type_label: bool = True,
        coercion: CoercionMode = "widen",
    ) -> Any:
        """Export this edge contract as a ``pyarrow.Schema``."""
        return to_arrow(self.to_row_schema(include_type_label=include_type_label), coercion=coercion)


@dataclass(frozen=True)
class GraphSchema:
    """Experimental public graph schema contract for GFQL validation."""

    node_types: Tuple[NodeType, ...] = field(default_factory=tuple)
    edge_types: Tuple[EdgeType, ...] = field(default_factory=tuple)
    strict: bool = True
    node_id_column: Optional[str] = None
    edge_source_column: Optional[str] = None
    edge_destination_column: Optional[str] = None

    def __init__(
        self,
        node_types: Iterable[NodeType] = (),
        edge_types: Iterable[EdgeType] = (),
        *,
        strict: bool = True,
        node_id_column: Optional[str] = None,
        edge_source_column: Optional[str] = None,
        edge_destination_column: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "node_types", tuple(node_types))
        object.__setattr__(self, "edge_types", tuple(edge_types))
        object.__setattr__(self, "strict", bool(strict))
        object.__setattr__(self, "node_id_column", node_id_column)
        object.__setattr__(self, "edge_source_column", edge_source_column)
        object.__setattr__(self, "edge_destination_column", edge_destination_column)

    @property
    def node_columns(self) -> FrozenSet[str]:
        columns: FrozenSet[str] = frozenset()
        for node_type in self.node_types:
            columns = columns | node_type.columns
        return columns

    @property
    def edge_columns(self) -> FrozenSet[str]:
        columns: FrozenSet[str] = frozenset()
        for edge_type in self.edge_types:
            columns = columns | edge_type.columns
        return columns

    @property
    def node_columns_by_label(self) -> Dict[str, Tuple[str, ...]]:
        columns_by_label: Dict[str, FrozenSet[str]] = {}
        for node_type in self.node_types:
            for label in node_type.labels:
                columns_by_label[label] = columns_by_label.get(label, frozenset()) | node_type.columns
        return {label: tuple(sorted(columns)) for label, columns in sorted(columns_by_label.items())}

    @property
    def edge_columns_by_type(self) -> Dict[str, Tuple[str, ...]]:
        return {edge_type.name: tuple(sorted(edge_type.columns)) for edge_type in self.edge_types}

    def to_catalog(
        self,
        *,
        node_id_column: Optional[str] = None,
        edge_source_column: Optional[str] = None,
        edge_destination_column: Optional[str] = None,
        strict: Optional[bool] = None,
    ) -> GraphSchemaCatalog:
        """Adapt this public schema into the internal GFQL schema catalog."""
        metadata = {
            "schema_model": "graphistry.schema.GraphSchema",
            "strict": self.strict if strict is None else bool(strict),
            "node_types": tuple(node_type.name for node_type in self.node_types),
            "edge_types": tuple(edge_type.name for edge_type in self.edge_types),
            "edge_topologies": tuple(edge_type.topology.as_metadata() for edge_type in self.edge_types),
            "node_columns_by_label": self.node_columns_by_label,
            "edge_columns_by_type": self.edge_columns_by_type,
            "node_row_schemas": {
                node_type.name: node_type.to_row_schema(include_labels=False)
                for node_type in self.node_types
            },
            "edge_row_schemas": {
                edge_type.name: edge_type.to_row_schema(include_type_label=False)
                for edge_type in self.edge_types
            },
        }
        return GraphSchemaCatalog.from_schema_parts(
            node_columns=self.node_columns,
            edge_columns=self.edge_columns,
            node_id_column=node_id_column or self.node_id_column,
            edge_source_column=edge_source_column or self.edge_source_column,
            edge_destination_column=edge_destination_column or self.edge_destination_column,
            metadata=metadata,
        )


__all__ = ["EdgeTopology", "EdgeType", "GraphSchema", "NodeType"]
