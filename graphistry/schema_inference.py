"""Experimental schema inference for public GFQL graph schemas."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Literal, Mapping, Optional, Tuple, Union

from graphistry.compute.gfql.ir.arrow_bridge import from_arrow
from graphistry.compute.gfql.ir.logical_plan import RowSchema
from graphistry.compute.gfql.ir.types import LogicalType, ScalarType
from graphistry.schema import EdgeType, GraphSchema, NodeType


PresenceState = Literal["required", "optional", "maybe_absent", "unknown"]

_LABEL_PREFIX = "label__"
_INFERENCE_METADATA_VERSION = 1


@dataclass(frozen=True)
class InferredProperty:
    """Experimental per-property inference detail."""

    logical_type: LogicalType
    presence: PresenceState


@dataclass(frozen=True)
class SchemaInferenceReport:
    """Experimental report for information not carried by ``GraphSchema``."""

    node_properties: Mapping[str, Mapping[str, InferredProperty]] = field(default_factory=dict)
    edge_properties: Mapping[str, Mapping[str, InferredProperty]] = field(default_factory=dict)


def _inference_metadata(source: str) -> Dict[str, Any]:
    return {
        "source": source,
        "inference": {
            "version": _INFERENCE_METADATA_VERSION,
            "label_column_prefix": _LABEL_PREFIX,
            "node_label_source": "node-label-columns",
            "edge_type_source": "edge-label-columns",
            "property_type_source": "dataframe-arrow-schema",
            "topology_source": "edge-endpoint-node-id-join",
        },
    }


def _columns(df: Any) -> Tuple[str, ...]:
    if df is None or not hasattr(df, "columns"):
        return tuple()
    return tuple(str(column) for column in df.columns)


def _label_names(df: Any) -> Tuple[str, ...]:
    return tuple(
        sorted(
            column[len(_LABEL_PREFIX):]
            for column in _columns(df)
            if column.startswith(_LABEL_PREFIX) and column[len(_LABEL_PREFIX):]
        )
    )


def _label_column(label: str) -> str:
    return f"{_LABEL_PREFIX}{label}"


def _as_bool(value: Any) -> bool:
    if hasattr(value, "item") and callable(getattr(value, "item")):
        return bool(value.item())
    return bool(value)


def _truth_mask(df: Any, label: str) -> Any:
    series = df[_label_column(label)]
    try:
        return (series == True).fillna(False)  # noqa: E712 - exact boolean label convention
    except (TypeError, ValueError):
        return series.fillna(False).astype("bool")


def _select_rows(df: Any, mask: Any) -> Any:
    return df.loc[mask]


def _select_columns(df: Any, columns: Iterable[str]) -> Any:
    selected = list(columns)
    if not selected:
        return df.iloc[:, 0:0]
    return df[selected]


def _series_has_nulls(series: Any) -> bool:
    if len(series) == 0:
        return True
    return _as_bool(series.isna().any())


def _series_has_values(series: Any) -> bool:
    if len(series) == 0:
        return False
    return not _as_bool(series.isna().all())


def _presence_for_series(series: Any) -> PresenceState:
    if len(series) == 0:
        return "unknown"
    if not _series_has_values(series):
        return "maybe_absent"
    if _series_has_nulls(series):
        return "optional"
    return "required"


def _to_arrow_schema(df: Any) -> Any:
    try:
        import pyarrow as pa
    except ImportError as exc:
        raise ImportError("pyarrow is required for graphistry.infer_schema()") from exc

    if hasattr(df, "to_arrow") and callable(getattr(df, "to_arrow")):
        try:
            arrow_obj = df.to_arrow(preserve_index=False)
        except TypeError:
            arrow_obj = df.to_arrow()
        return arrow_obj.schema
    return pa.Table.from_pandas(df, preserve_index=False).schema


def _row_schema_for_frame(df: Any, columns: Iterable[str]) -> RowSchema:
    selected_columns = tuple(columns)
    if not selected_columns:
        return RowSchema()
    row_schema, _ = from_arrow(_to_arrow_schema(_select_columns(df, selected_columns)))
    adjusted: Dict[str, LogicalType] = {}
    for column, logical_type in row_schema.columns.items():
        if isinstance(logical_type, ScalarType):
            adjusted[column] = ScalarType(logical_type.kind, nullable=_series_has_nulls(df[column]))
        else:
            adjusted[column] = logical_type
    return RowSchema(columns=adjusted)


def _property_schema_and_report(
    df: Any,
    *,
    candidates: Iterable[str],
) -> Tuple[Dict[str, LogicalType], Dict[str, InferredProperty]]:
    schema_columns = []
    report: Dict[str, InferredProperty] = {}
    for column in sorted(candidates):
        presence = _presence_for_series(df[column])
        if presence in {"required", "optional"}:
            schema_columns.append(column)

    row_schema = _row_schema_for_frame(df, schema_columns)
    for column in sorted(candidates):
        logical_type = row_schema.columns.get(column, ScalarType("unknown"))
        report[column] = InferredProperty(
            logical_type=logical_type,
            presence=_presence_for_series(df[column]),
        )
    return dict(row_schema.columns), report


def _infer_node_types(nodes_df: Any) -> Tuple[Tuple[NodeType, ...], Dict[str, Mapping[str, InferredProperty]]]:
    if nodes_df is None:
        return tuple(), {}

    label_columns = frozenset(_label_column(label) for label in _label_names(nodes_df))
    candidate_columns = tuple(column for column in _columns(nodes_df) if column not in label_columns)

    node_types = []
    report: Dict[str, Mapping[str, InferredProperty]] = {}
    for label in _label_names(nodes_df):
        selected = _select_rows(nodes_df, _truth_mask(nodes_df, label))
        properties, property_report = _property_schema_and_report(
            selected,
            candidates=candidate_columns,
        )
        node_types.append(NodeType(label, properties))
        report[label] = property_report
    return tuple(node_types), report


def _series_to_values(series: Any) -> Tuple[Any, ...]:
    if hasattr(series, "to_arrow") and callable(getattr(series, "to_arrow")):
        return tuple(series.to_arrow().to_pylist())
    if hasattr(series, "tolist") and callable(getattr(series, "tolist")):
        return tuple(series.tolist())
    return tuple(series)


def _node_labels_by_id(nodes_df: Any, *, node_id_column: Optional[str]) -> Dict[Any, FrozenSet[str]]:
    if nodes_df is None or node_id_column is None or node_id_column not in _columns(nodes_df):
        return {}

    out: Dict[Any, FrozenSet[str]] = {}
    for label in _label_names(nodes_df):
        selected = _select_rows(nodes_df, _truth_mask(nodes_df, label))
        for node_id in _series_to_values(selected[node_id_column]):
            out[node_id] = out.get(node_id, frozenset()) | frozenset((label,))
    return out


def _topology_labels(
    edges_df: Any,
    *,
    mask: Any,
    endpoint_column: Optional[str],
    labels_by_id: Mapping[Any, FrozenSet[str]],
) -> FrozenSet[str]:
    if edges_df is None or endpoint_column is None or endpoint_column not in _columns(edges_df):
        return frozenset()
    labels: FrozenSet[str] = frozenset()
    selected = _select_rows(edges_df, mask)
    for node_id in _series_to_values(selected[endpoint_column]):
        labels = labels | labels_by_id.get(node_id, frozenset())
    return labels


def _infer_edge_types(
    edges_df: Any,
    *,
    nodes_df: Any,
    node_id_column: Optional[str],
    edge_source_column: Optional[str],
    edge_destination_column: Optional[str],
    edge_id_column: Optional[str],
) -> Tuple[Tuple[EdgeType, ...], Dict[str, Mapping[str, InferredProperty]]]:
    if edges_df is None:
        return tuple(), {}

    edge_label_columns = frozenset(_label_column(label) for label in _label_names(edges_df))
    structural_columns = {
        column
        for column in (edge_source_column, edge_destination_column, edge_id_column)
        if column is not None
    }
    candidate_columns = tuple(
        column
        for column in _columns(edges_df)
        if column not in edge_label_columns and column not in structural_columns
    )
    labels_by_id = _node_labels_by_id(nodes_df, node_id_column=node_id_column)

    edge_types = []
    report: Dict[str, Mapping[str, InferredProperty]] = {}
    for relationship_type in _label_names(edges_df):
        mask = _truth_mask(edges_df, relationship_type)
        selected = _select_rows(edges_df, mask)
        properties, property_report = _property_schema_and_report(
            selected,
            candidates=candidate_columns,
        )
        edge_types.append(
            EdgeType(
                relationship_type,
                source=_topology_labels(
                    edges_df,
                    mask=mask,
                    endpoint_column=edge_source_column,
                    labels_by_id=labels_by_id,
                ),
                destination=_topology_labels(
                    edges_df,
                    mask=mask,
                    endpoint_column=edge_destination_column,
                    labels_by_id=labels_by_id,
                ),
                properties=properties,
            )
        )
        report[relationship_type] = property_report
    return tuple(edge_types), report


def _merge_declared_schema(inferred: GraphSchema, declared: Optional[GraphSchema]) -> GraphSchema:
    if declared is None:
        return inferred

    declared_nodes = {node_type.name: node_type for node_type in declared.node_types}
    declared_edges = {edge_type.name: edge_type for edge_type in declared.edge_types}
    metadata = dict(declared.metadata)
    metadata.update(_inference_metadata("mixed"))
    node_types = tuple(declared_nodes.values()) + tuple(
        node_type for node_type in inferred.node_types if node_type.name not in declared_nodes
    )
    edge_types = tuple(declared_edges.values()) + tuple(
        edge_type for edge_type in inferred.edge_types if edge_type.name not in declared_edges
    )
    return GraphSchema(
        node_types=node_types,
        edge_types=edge_types,
        strict=declared.strict,
        node_id_column=declared.node_id_column or inferred.node_id_column,
        edge_source_column=declared.edge_source_column or inferred.edge_source_column,
        edge_destination_column=declared.edge_destination_column or inferred.edge_destination_column,
        metadata=metadata,
    )


def infer_schema(
    g: Any,
    *,
    schema: Optional[GraphSchema] = None,
    return_report: bool = False,
) -> Union[GraphSchema, Tuple[GraphSchema, SchemaInferenceReport]]:
    """Infer an experimental public ``GraphSchema`` from bound graph data.

    Inference is opt-in and returns the same public schema objects as declared
    schemas. When ``schema`` is provided, its node/edge declarations override
    inferred declarations with the same names.
    """
    nodes_df = getattr(g, "_nodes", None)
    edges_df = getattr(g, "_edges", None)
    node_id_column = getattr(g, "_node", None)
    edge_source_column = getattr(g, "_source", None)
    edge_destination_column = getattr(g, "_destination", None)
    edge_id_column = getattr(g, "_edge", None)

    node_types, node_report = _infer_node_types(nodes_df)
    edge_types, edge_report = _infer_edge_types(
        edges_df,
        nodes_df=nodes_df,
        node_id_column=node_id_column,
        edge_source_column=edge_source_column,
        edge_destination_column=edge_destination_column,
        edge_id_column=edge_id_column,
    )
    inferred = GraphSchema(
        node_types=node_types,
        edge_types=edge_types,
        node_id_column=node_id_column,
        edge_source_column=edge_source_column,
        edge_destination_column=edge_destination_column,
        metadata=_inference_metadata("inferred"),
    )
    merged = _merge_declared_schema(inferred, schema)
    report = SchemaInferenceReport(
        node_properties=node_report,
        edge_properties=edge_report,
    )
    return (merged, report) if return_report else merged


__all__ = [
    "InferredProperty",
    "PresenceState",
    "SchemaInferenceReport",
    "infer_schema",
]
