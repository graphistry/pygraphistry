"""Internal schema-effect propagation for graph-growing GFQL calls.

This module is deliberately private while typed-schema inference, Arrow
boundaries, remote transport, and graph-value planning settle. It updates a
bound public ``GraphSchema`` after graph-growing calls succeed, so later local
validation sees properties written by earlier operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple, cast

from typing_extensions import Literal

from graphistry.Plottable import Plottable
from graphistry.compute.gfql.ir.types import LogicalType, ScalarType
from graphistry.schema import EdgeType, GraphSchema, NodeType


SchemaEffectConfidence = Literal["declared", "inferred", "observed", "user_refined"]
SchemaEffectUpdateMap = Mapping[str, Mapping[str, LogicalType]]

_CONFIDENCE_VALUES: FrozenSet[str] = frozenset(
    ("declared", "inferred", "observed", "user_refined")
)


@dataclass(frozen=True)
class SchemaEffect:
    """Private graph-schema delta for graph-growing transforms.

    The shape mirrors the typed-schema research model but remains internal.
    Initial call integrations only use property additions; ``updates`` and
    ``drops`` are included now so effect records have stable internal semantics
    before public exposure is considered.
    """

    requires: Optional[GraphSchema] = None
    adds_node_properties: Mapping[str, LogicalType] = field(default_factory=dict)
    adds_edge_properties: Mapping[str, LogicalType] = field(default_factory=dict)
    updates: SchemaEffectUpdateMap = field(default_factory=dict)
    drops: Tuple[str, ...] = ()
    confidence: SchemaEffectConfidence = "declared"

    def __post_init__(self) -> None:
        if self.confidence not in _CONFIDENCE_VALUES:
            raise ValueError(
                "SchemaEffect.confidence must be one of "
                f"{tuple(sorted(_CONFIDENCE_VALUES))}; got {self.confidence!r}"
            )


_INT32 = ScalarType("int32")
_INT64 = ScalarType("int64")
_FLOAT64 = ScalarType("float64")
_STRING = ScalarType("string")

_NODE_FLOAT_ALGS: FrozenSet[str] = frozenset(
    (
        "authority_score",
        "betweenness",
        "betweenness_centrality",
        "closeness",
        "closeness_centrality",
        "constraint",
        "degree_centrality",
        "eigenvector_centrality",
        "harmonic_centrality",
        "hub_score",
        "hits",
        "katz_centrality",
        "pagerank",
        "personalized_pagerank",
    )
)
_NODE_INT_ALGS: FrozenSet[str] = frozenset(
    (
        "articulation_points",
        "clusters",
        "community_edge_betweenness",
        "community_fastgreedy",
        "community_infomap",
        "community_label_propagation",
        "community_leading_eigenvector",
        "community_leiden",
        "community_multilevel",
        "community_optimal_modularity",
        "community_spinglass",
        "community_walktrap",
        "connected_components",
        "core_number",
        "coreness",
        "ecg",
        "leiden",
        "louvain",
        "spectralBalancedCutClustering",
        "spectralModularityMaximizationClustering",
        "strongly_connected_components",
    )
)
_NODE_STRING_ALGS: FrozenSet[str] = frozenset(("bfs", "bfs_edges", "shortest_path", "sssp"))
_EDGE_FLOAT_ALGS: FrozenSet[str] = frozenset(
    (
        "edge_betweenness_centrality",
        "jaccard",
        "jaccard_w",
        "overlap",
        "overlap_coefficient",
        "overlap_w",
        "sorensen",
        "sorensen_coefficient",
        "sorensen_w",
    )
)
_CUGRAPH_EDGE_DEFAULT_COLUMNS: Mapping[str, str] = {
    "batched_ego_graphs": "unknown",
    "edge_betweenness_centrality": "betweenness_centrality",
    "jaccard": "jaccard_coeff",
    "jaccard_w": "jaccard_coeff",
    "overlap": "overlap_coeff",
    "overlap_coefficient": "overlap_coefficient",
    "overlap_w": "overlap_coeff",
    "sorensen": "sorensen_coeff",
    "sorensen_coefficient": "sorensen_coeff",
    "sorensen_w": "sorensen_coeff",
}


def _typed_properties(columns: Iterable[str], logical_type: LogicalType) -> Dict[str, LogicalType]:
    return {str(column): logical_type for column in columns if str(column)}


def _property_type_for_algorithm(algorithm: Optional[str], *, table: Literal["nodes", "edges"]) -> LogicalType:
    if algorithm is None:
        return _STRING
    if table == "edges":
        return _FLOAT64 if algorithm in _EDGE_FLOAT_ALGS else _STRING
    if algorithm in _NODE_FLOAT_ALGS:
        return _FLOAT64
    if algorithm in _NODE_INT_ALGS:
        return _INT64
    if algorithm in _NODE_STRING_ALGS:
        return _STRING
    return _STRING


def schema_effect_for_call(function: str, params: Mapping[str, Any]) -> Optional[SchemaEffect]:
    """Return an internal effect for a successful safelisted ``call(...)``."""
    if function == "get_degrees":
        return SchemaEffect(
            adds_node_properties={
                str(params.get("col", "degree")): _INT32,
                str(params.get("degree_in", "degree_in")): _INT32,
                str(params.get("degree_out", "degree_out")): _INT32,
            },
            confidence="declared",
        )
    if function == "get_indegrees":
        return SchemaEffect(
            adds_node_properties={str(params.get("col", "degree_in")): _INT32},
            confidence="declared",
        )
    if function == "get_outdegrees":
        return SchemaEffect(
            adds_node_properties={str(params.get("col", "degree_out")): _INT32},
            confidence="declared",
        )
    if function == "compute_cugraph":
        algorithm = params.get("alg")
        if not isinstance(algorithm, str):
            return None
        if algorithm in _CUGRAPH_EDGE_DEFAULT_COLUMNS:
            out_col = cast(Optional[str], params.get("out_col")) or _CUGRAPH_EDGE_DEFAULT_COLUMNS[algorithm]
            return SchemaEffect(
                adds_edge_properties={
                    str(out_col): _property_type_for_algorithm(algorithm, table="edges")
                },
                confidence="declared",
            )
        out_col = cast(Optional[str], params.get("out_col")) or algorithm
        if out_col is None:
            return None
        properties = {str(out_col): _property_type_for_algorithm(algorithm, table="nodes")}
        if algorithm == "hits":
            properties["authorities"] = _FLOAT64
        return SchemaEffect(adds_node_properties=properties, confidence="declared")
    if function == "compute_igraph":
        algorithm = params.get("alg")
        if not isinstance(algorithm, str):
            return None
        out_col = cast(Optional[str], params.get("out_col")) or algorithm
        return SchemaEffect(
            adds_node_properties={
                str(out_col): _property_type_for_algorithm(algorithm, table="nodes")
            },
            confidence="declared",
        )
    return None


def schema_effect_for_procedure_output(
    *,
    backend: str,
    algorithm: Optional[str],
    row_kind: str,
    value_columns: Sequence[str],
) -> Optional[SchemaEffect]:
    """Return an internal effect for a successful Cypher graph-write procedure."""
    if backend == "degree":
        return SchemaEffect(
            adds_node_properties={
                "degree": _INT32,
                "degree_in": _INT32,
                "degree_out": _INT32,
            },
            confidence="declared",
        )
    if not value_columns:
        return None
    if row_kind == "edge":
        return SchemaEffect(
            adds_edge_properties=_typed_properties(
                value_columns,
                _property_type_for_algorithm(algorithm, table="edges"),
            ),
            confidence="declared",
        )
    if row_kind in {"node", "node_or_graph"}:
        return SchemaEffect(
            adds_node_properties=_typed_properties(
                value_columns,
                _property_type_for_algorithm(algorithm, table="nodes"),
            ),
            confidence="declared",
        )
    return None


def apply_schema_effect(schema: GraphSchema, effect: SchemaEffect) -> GraphSchema:
    """Apply an internal effect to a public ``GraphSchema`` snapshot."""
    node_types = tuple(
        _apply_node_properties(node_type, effect.adds_node_properties)
        for node_type in schema.node_types
    )
    edge_types = tuple(
        _apply_edge_properties(edge_type, effect.adds_edge_properties)
        for edge_type in schema.edge_types
    )
    return GraphSchema(
        node_types=node_types,
        edge_types=edge_types,
        strict=schema.strict,
        node_id_column=schema.node_id_column,
        edge_source_column=schema.edge_source_column,
        edge_destination_column=schema.edge_destination_column,
    )


def apply_call_schema_effect(
    input_graph: Plottable,
    result_graph: Plottable,
    function: str,
    params: Mapping[str, Any],
) -> Plottable:
    return apply_graph_schema_effect(
        input_graph,
        result_graph,
        schema_effect_for_call(function, params),
    )


def apply_graph_schema_effect(
    input_graph: Plottable,
    result_graph: Plottable,
    effect: Optional[SchemaEffect],
) -> Plottable:
    if effect is None:
        return result_graph
    bound_schema = getattr(input_graph, "_gfql_schema", None)
    if not isinstance(bound_schema, GraphSchema):
        return result_graph

    filtered_effect = _filter_effect_to_result_columns(result_graph, effect)
    if not filtered_effect.adds_node_properties and not filtered_effect.adds_edge_properties:
        return result_graph

    return result_graph.bind(schema=apply_schema_effect(bound_schema, filtered_effect))


def _apply_node_properties(
    node_type: NodeType,
    additions: Mapping[str, LogicalType],
) -> NodeType:
    if not additions:
        return node_type
    properties = dict(node_type.properties)
    for name, logical_type in additions.items():
        properties.setdefault(str(name), logical_type)
    return NodeType(node_type.name, properties=properties, labels=node_type.labels)


def _apply_edge_properties(
    edge_type: EdgeType,
    additions: Mapping[str, LogicalType],
) -> EdgeType:
    if not additions:
        return edge_type
    properties = dict(edge_type.properties)
    for name, logical_type in additions.items():
        properties.setdefault(str(name), logical_type)
    return EdgeType(edge_type.name, edge_type.source, edge_type.destination, properties=properties)


def _filter_effect_to_result_columns(result_graph: Plottable, effect: SchemaEffect) -> SchemaEffect:
    nodes_df = getattr(result_graph, "_nodes", None)
    edges_df = getattr(result_graph, "_edges", None)
    node_columns = set(nodes_df.columns) if nodes_df is not None else set()
    edge_columns = set(edges_df.columns) if edges_df is not None else set()
    return SchemaEffect(
        requires=effect.requires,
        adds_node_properties={
            name: logical_type
            for name, logical_type in effect.adds_node_properties.items()
            if name in node_columns
        },
        adds_edge_properties={
            name: logical_type
            for name, logical_type in effect.adds_edge_properties.items()
            if name in edge_columns
        },
        updates=effect.updates,
        drops=effect.drops,
        confidence=effect.confidence,
    )
