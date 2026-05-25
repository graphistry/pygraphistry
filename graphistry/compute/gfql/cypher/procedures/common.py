from __future__ import annotations

from typing import Any, NoReturn, Sequence, Tuple, cast

import pandas as pd

from graphistry.Engine import safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.typing import DataFrameT


def raise_missing_backend_dependency(
    compiled_call: Any,
    *,
    dependency: str,
    suggestion: str,
    exc: Exception,
) -> NoReturn:
    raise GFQLValidationError(
        ErrorCode.E108,
        f"{compiled_call.procedure} requires the optional '{dependency}' dependency",
        field="call",
        value=compiled_call.procedure,
        suggestion=suggestion,
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    ) from exc


def materialized_graph(base_graph: Plottable) -> Plottable:
    graph_with_nodes = base_graph.materialize_nodes()
    if graph_with_nodes._nodes is None or graph_with_nodes._node is None:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Graphistry Cypher CALL requires a materialized node table",
            field="call",
            value="nodes",
            suggestion="Bind or materialize nodes before using graphistry.* CALL procedures.",
            language="cypher",
        )
    return graph_with_nodes


def as_pandas_frame(df: Any, columns: Tuple[str, ...]) -> pd.DataFrame:
    subset = df[list(columns)]
    if hasattr(subset, "to_pandas"):
        return cast(pd.DataFrame, subset.to_pandas())
    return pd.DataFrame(subset)


def merge_node_property_rows(
    base_graph: Plottable,
    rows_df: DataFrameT,
    *,
    source_value_col: str,
    output_value_col: str,
) -> Plottable:
    graph_with_nodes = materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    node_col = graph_with_nodes._node
    nodes_df = graph_with_nodes._nodes
    merge_base = cast(DataFrameT, nodes_df.drop(columns=[output_value_col])) if output_value_col in nodes_df.columns else nodes_df
    projected_rows = rows_df.rename(columns={"nodeId": node_col, source_value_col: output_value_col})[
        [node_col, output_value_col]
    ]
    merged_nodes = cast(DataFrameT, safe_merge(merge_base, projected_rows, on=node_col, how="left"))
    return graph_with_nodes.nodes(merged_nodes, node=node_col)


def merge_node_property_columns(
    base_graph: Plottable,
    rows_df: DataFrameT,
    *,
    value_columns: Sequence[str],
) -> Plottable:
    graph_with_nodes = materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    node_col = graph_with_nodes._node
    nodes_df = graph_with_nodes._nodes
    existing_value_columns = [col for col in value_columns if col in nodes_df.columns]
    merge_base = cast(DataFrameT, nodes_df.drop(columns=existing_value_columns)) if existing_value_columns else nodes_df
    projected_rows = rows_df.rename(columns={"nodeId": node_col})[[node_col, *value_columns]]
    merged_nodes = cast(DataFrameT, safe_merge(merge_base, projected_rows, on=node_col, how="left"))
    return graph_with_nodes.nodes(merged_nodes, node=node_col)


def merge_edge_property_rows(
    base_graph: Plottable,
    rows_df: DataFrameT,
    *,
    source_value_col: str,
    output_value_col: str,
) -> Plottable:
    graph_with_edges = base_graph.bind()
    if (
        graph_with_edges._edges is None
        or graph_with_edges._source is None
        or graph_with_edges._destination is None
    ):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Graphistry Cypher CALL requires an edge table for edge-enriching procedures",
            field="call",
            value="edges",
            suggestion="Bind edges before using edge-returning graphistry.* CALL procedures.",
            language="cypher",
        )
    source_col = graph_with_edges._source
    destination_col = graph_with_edges._destination
    edges_df = graph_with_edges._edges
    merge_base = (
        cast(DataFrameT, edges_df.drop(columns=[output_value_col]))
        if output_value_col in edges_df.columns
        else edges_df
    )
    projected_rows = rows_df.rename(
        columns={
            "source": source_col,
            "destination": destination_col,
            source_value_col: output_value_col,
        }
    )[[source_col, destination_col, output_value_col]]
    merged_edges = cast(
        DataFrameT,
        safe_merge(merge_base, projected_rows, on=[source_col, destination_col], how="left"),
    )
    return graph_with_edges.edges(merged_edges, source_col, destination_col)
