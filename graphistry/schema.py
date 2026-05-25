"""Experimental public declarative graph schema model for GFQL validation.

The import path is public, but this schema surface is still experimental while
inference, coercion, remote transport, and planner use are developed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Tuple, Union, cast

from graphistry.compute.gfql.ir.compilation import GraphSchemaCatalog
from graphistry.compute.gfql.ir.arrow_bridge import CoercionMode, from_arrow, to_arrow
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import EdgeRef, ListType, LogicalType, NodeRef, PathType, ScalarType


NodeRefInput = Union["NodeType", str, Iterable[str]]
PropertySchemaInput = Union[Mapping[str, Any], RowSchema, Any]
GraphArrowDeclaration = Mapping[str, Any]


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


def _label_column(label: str) -> str:
    return f"label__{label}"


def _labels_from_arrow_schema(schema: Any) -> FrozenSet[str]:
    labels = []
    for arrow_field in schema:
        name = str(arrow_field.name)
        if name.startswith("label__") and len(name) > len("label__"):
            labels.append(name[len("label__"):])
    return frozenset(labels)


def _strip_label_properties(
    properties: Mapping[str, LogicalType],
    labels: Iterable[str],
) -> Dict[str, LogicalType]:
    label_columns = {_label_column(label) for label in labels}
    return {
        name: logical_type
        for name, logical_type in properties.items()
        if name not in label_columns
    }


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

    @classmethod
    def from_arrow(
        cls,
        name: str,
        schema: Any,
        *,
        labels: Optional[Iterable[str]] = None,
        include_labels: bool = True,
        coercion: CoercionMode = "widen",
    ) -> "NodeType":
        """Import a node contract from a ``pyarrow.Schema`` declaration."""
        row_schema, _ = from_arrow(schema, coercion=coercion, default_confidence="declared")
        normalized_labels = _normalize_labels(name, labels)
        if include_labels and labels is None:
            inferred_labels = _labels_from_arrow_schema(schema)
            if inferred_labels:
                normalized_labels = inferred_labels
        properties = row_schema.columns
        if include_labels:
            properties = _strip_label_properties(properties, normalized_labels)
        return cls(name, properties=properties, labels=normalized_labels)


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

    @classmethod
    def from_metadata(cls, value: Mapping[str, object]) -> "EdgeTopology":
        """Import topology from the metadata shape emitted by ``as_metadata()``."""
        return cls(
            relationship_type=str(value["relationship_type"]),
            source_labels=frozenset(str(label) for label in cast(Iterable[object], value["source_labels"])),
            destination_labels=frozenset(
                str(label) for label in cast(Iterable[object], value["destination_labels"])
            ),
        )


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

    @classmethod
    def from_arrow(
        cls,
        name: str,
        source: NodeRefInput,
        destination: NodeRefInput,
        schema: Any,
        *,
        include_type_label: bool = True,
        coercion: CoercionMode = "widen",
    ) -> "EdgeType":
        """Import an edge contract from a ``pyarrow.Schema`` declaration."""
        row_schema, _ = from_arrow(schema, coercion=coercion, default_confidence="declared")
        properties = dict(row_schema.columns)
        if include_type_label:
            properties.pop(_label_column(name), None)
        return cls(name, source=source, destination=destination, properties=properties)


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

    def node_arrow(
        self,
        *,
        include_labels: bool = True,
        coercion: CoercionMode = "widen",
    ) -> Any:
        """Export the union of node declarations as a ``pyarrow.Schema``."""
        return _merge_arrow_schemas(
            (
                node_type.to_arrow(include_labels=include_labels, coercion=coercion)
                for node_type in self.node_types
            ),
            kind="nodes",
        )

    def edge_arrow(
        self,
        *,
        include_type_labels: bool = True,
        coercion: CoercionMode = "widen",
    ) -> Any:
        """Export the union of edge declarations as a ``pyarrow.Schema``."""
        return _merge_arrow_schemas(
            (
                edge_type.to_arrow(include_type_label=include_type_labels, coercion=coercion)
                for edge_type in self.edge_types
            ),
            kind="edges",
        )

    def to_arrow(
        self,
        *,
        include_labels: bool = True,
        include_type_labels: bool = True,
        coercion: CoercionMode = "widen",
    ) -> Dict[str, Any]:
        """Export this graph schema as Arrow declarations.

        The payload is intentionally declaration-shaped rather than inferred:
        per-node and per-edge entries preserve public type names and topology,
        while ``nodes`` and ``edges`` expose table-level merged schemas for
        dataframe boundary validation.
        """
        return {
            "nodes": self.node_arrow(include_labels=include_labels, coercion=coercion),
            "edges": self.edge_arrow(include_type_labels=include_type_labels, coercion=coercion),
            "node_types": {
                node_type.name: node_type.to_arrow(include_labels=include_labels, coercion=coercion)
                for node_type in self.node_types
            },
            "edge_types": {
                edge_type.name: {
                    "source": tuple(sorted(edge_type.source)),
                    "destination": tuple(sorted(edge_type.destination)),
                    "schema": edge_type.to_arrow(
                        include_type_label=include_type_labels,
                        coercion=coercion,
                    ),
                }
                for edge_type in self.edge_types
            },
            "strict": self.strict,
            "node_id_column": self.node_id_column,
            "edge_source_column": self.edge_source_column,
            "edge_destination_column": self.edge_destination_column,
        }

    @classmethod
    def from_arrow(
        cls,
        declaration: Optional[GraphArrowDeclaration] = None,
        *,
        node_types: Optional[Mapping[str, Any]] = None,
        edge_types: Optional[Mapping[str, Any]] = None,
        strict: bool = True,
        node_id_column: Optional[str] = None,
        edge_source_column: Optional[str] = None,
        edge_destination_column: Optional[str] = None,
        coercion: CoercionMode = "widen",
    ) -> "GraphSchema":
        """Import graph schema declarations from Arrow schemas.

        This is not inference: callers provide node/edge names and edge
        topology, either directly or via the payload emitted by ``to_arrow()``.
        """
        if declaration is not None:
            node_types = cast(Optional[Mapping[str, Any]], declaration.get("node_types", node_types))
            edge_types = cast(Optional[Mapping[str, Any]], declaration.get("edge_types", edge_types))
            strict = bool(declaration.get("strict", strict))
            node_id_column = cast(Optional[str], declaration.get("node_id_column", node_id_column))
            edge_source_column = cast(Optional[str], declaration.get("edge_source_column", edge_source_column))
            edge_destination_column = cast(
                Optional[str],
                declaration.get("edge_destination_column", edge_destination_column),
            )

        nodes = tuple(
            NodeType.from_arrow(name, schema, coercion=coercion)
            for name, schema in (node_types or {}).items()
        )
        edges = tuple(
            _edge_type_from_arrow_entry(name, entry, coercion=coercion)
            for name, entry in (edge_types or {}).items()
        )
        return cls(
            node_types=nodes,
            edge_types=edges,
            strict=strict,
            node_id_column=node_id_column,
            edge_source_column=edge_source_column,
            edge_destination_column=edge_destination_column,
        )

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


def _merge_arrow_schemas(schemas: Iterable[Any], *, kind: str) -> Any:
    import pyarrow as pa

    fields: Dict[str, Any] = {}
    metadata: Optional[Mapping[bytes, bytes]] = None
    for schema in schemas:
        if schema.metadata is not None:
            if metadata is not None and metadata != schema.metadata:
                raise ValueError(f"Conflicting Arrow schema metadata for {kind} declarations")
            metadata = schema.metadata
        for arrow_field in schema:
            existing = fields.get(arrow_field.name)
            if existing is not None and existing != arrow_field:
                raise ValueError(
                    f"Conflicting Arrow declaration for {kind} column {arrow_field.name!r}: "
                    f"{existing!r} vs {arrow_field!r}"
                )
            fields[arrow_field.name] = arrow_field
    return pa.schema(list(fields.values()), metadata=metadata)


def _edge_type_from_arrow_entry(
    name: str,
    entry: Any,
    *,
    coercion: CoercionMode,
) -> EdgeType:
    if isinstance(entry, Mapping):
        source = cast(NodeRefInput, entry["source"])
        destination = cast(NodeRefInput, entry["destination"])
        schema = entry["schema"]
    else:
        source, destination, schema = entry
    return EdgeType.from_arrow(name, source, destination, schema, coercion=coercion)


__all__ = ["EdgeTopology", "EdgeType", "GraphSchema", "NodeType"]
