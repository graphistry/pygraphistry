from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Literal, Mapping, Optional, Tuple, cast

import pandas as pd

from graphistry.Engine import safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.procedures.common import (
    as_pandas_frame,
    materialized_graph,
    merge_edge_property_rows,
    merge_node_property_columns,
    merge_node_property_rows,
    raise_missing_backend_dependency,
)
from graphistry.compute.typing import DataFrameT
from graphistry.plugins.networkx.policy import networkx_version_error, scipy_version_error

if TYPE_CHECKING:
    from graphistry.compute.gfql.cypher.call_procedures import CompiledCypherProcedureCall


NETWORKX_RESERVED_KEYS = frozenset({"out_col", "params", "directed"})
NETWORKX_PROCEDURES: Dict[str, Tuple[Literal["node", "edge", "graph_only"], Tuple[str, ...]]] = {
    "pagerank": ("node", ("pagerank",)),
    "betweenness_centrality": ("node", ("betweenness_centrality",)),
    "closeness_centrality": ("node", ("closeness_centrality",)),
    "connected_components": ("node", ("labels",)),
    "core_number": ("node", ("core_number",)),
    "degree_centrality": ("node", ("degree_centrality",)),
    "eigenvector_centrality": ("node", ("eigenvector_centrality",)),
    "edge_betweenness_centrality": ("edge", ("edge_betweenness_centrality",)),
    "hits": ("node", ("hubs", "authorities")),
    "katz_centrality": ("node", ("katz_centrality",)),
    "k_core": ("graph_only", ()),
    "strongly_connected_components": ("node", ("labels",)),
}


def networkx_source_value_columns(algorithm: Optional[str]) -> Tuple[str, ...]:
    if algorithm is None:
        return ()
    nx_spec = NETWORKX_PROCEDURES.get(algorithm)
    return () if nx_spec is None else nx_spec[1]


def networkx_normalized_value_columns(compiled_call: CompiledCypherProcedureCall) -> Tuple[str, ...]:
    value_cols = list(networkx_source_value_columns(compiled_call.algorithm))
    out_col = compiled_call.call_params.get("out_col")
    if value_cols and out_col is not None:
        if len(value_cols) > 1:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Graphistry Cypher CALL does not allow out_col for multi-column procedures",
                field="call.args.out_col",
                value=out_col,
                suggestion="Remove out_col or choose a procedure with a single value column.",
                language="cypher",
            )
        value_cols[0] = cast(str, out_col)
    return tuple(value_cols)


def _networkx_module(compiled_call: CompiledCypherProcedureCall) -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise_missing_backend_dependency(
            compiled_call,
            dependency="networkx",
            suggestion="Install networkx or use graphistry.igraph.* / graphistry.cugraph.* procedures.",
            exc=exc,
        )
    _ensure_networkx_version_policy(compiled_call, nx)
    return nx


def _ensure_networkx_version_policy(compiled_call: CompiledCypherProcedureCall, nx: Any) -> None:
    error = networkx_version_error(getattr(nx, "__version__", None))
    if error is None:
        return
    raise GFQLValidationError(
        ErrorCode.E108,
        f"{compiled_call.procedure} requires a supported NetworkX version",
        field="call",
        value=compiled_call.procedure,
        suggestion=error,
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    )


def _ensure_scipy_version_policy(compiled_call: CompiledCypherProcedureCall, scipy_module: Any) -> None:
    error = scipy_version_error(getattr(scipy_module, "__version__", None))
    if error is None:
        return
    raise GFQLValidationError(
        ErrorCode.E108,
        f"{compiled_call.procedure} requires a supported SciPy version when SciPy is installed",
        field="call",
        value=compiled_call.procedure,
        suggestion=error,
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    )


def _optional_scipy_module() -> Optional[Any]:
    try:
        import scipy
    except ImportError:
        return None
    return scipy


def _ensure_networkx_feature(
    has_feature: bool,
    compiled_call: CompiledCypherProcedureCall,
    feature_name: str,
    nx: Any,
) -> None:
    if has_feature:
        return
    nx_version = getattr(nx, "__version__", "unknown")
    raise GFQLValidationError(
        ErrorCode.E108,
        f"{compiled_call.procedure} requires {feature_name}, which is unavailable in the installed NetworkX version",
        field="call",
        value=compiled_call.procedure,
        suggestion=f"Upgrade networkx or use a different graphistry.nx procedure. Installed networkx version: {nx_version}.",
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    )


def _networkx_algorithm_result(
    compiled_call: CompiledCypherProcedureCall,
    graph_nx: Any,
    params: Mapping[str, Any],
) -> Any:
    algorithm = compiled_call.algorithm
    try:
        if algorithm == "pagerank":
            return _networkx_pagerank_scores(graph_nx, **params)
        nx = _networkx_module(compiled_call)
        if algorithm == "betweenness_centrality":
            _ensure_networkx_feature(hasattr(nx, "betweenness_centrality"), compiled_call, "networkx.betweenness_centrality", nx)
            return nx.betweenness_centrality(graph_nx, **params)
        if algorithm == "closeness_centrality":
            _ensure_networkx_feature(hasattr(nx, "closeness_centrality"), compiled_call, "networkx.closeness_centrality", nx)
            return nx.closeness_centrality(graph_nx, **params)
        if algorithm == "connected_components":
            _raise_networkx_no_params(params)
            if graph_nx.is_directed():
                _ensure_networkx_feature(hasattr(nx, "weakly_connected_components"), compiled_call, "networkx.weakly_connected_components", nx)
                components = nx.weakly_connected_components(graph_nx)
            else:
                _ensure_networkx_feature(hasattr(nx, "connected_components"), compiled_call, "networkx.connected_components", nx)
                components = nx.connected_components(graph_nx)
            return _networkx_component_labels(components)
        if algorithm == "core_number":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(hasattr(nx, "core_number"), compiled_call, "networkx.core_number", nx)
            return nx.core_number(graph_nx, **params)
        if algorithm == "degree_centrality":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(hasattr(nx, "degree_centrality"), compiled_call, "networkx.degree_centrality", nx)
            return nx.degree_centrality(graph_nx)
        if algorithm == "eigenvector_centrality":
            _ensure_networkx_feature(hasattr(nx, "eigenvector_centrality"), compiled_call, "networkx.eigenvector_centrality", nx)
            return nx.eigenvector_centrality(graph_nx, **params)
        if algorithm == "edge_betweenness_centrality":
            _ensure_networkx_feature(hasattr(nx, "edge_betweenness_centrality"), compiled_call, "networkx.edge_betweenness_centrality", nx)
            return nx.edge_betweenness_centrality(graph_nx, **params)
        if algorithm == "hits":
            _ensure_networkx_feature(hasattr(nx, "hits"), compiled_call, "networkx.hits", nx)
            scipy_module = _optional_scipy_module()
            if scipy_module is not None:
                _ensure_scipy_version_policy(compiled_call, scipy_module)
            try:
                hubs, authorities = nx.hits(graph_nx, **params)
            except ModuleNotFoundError as exc:
                if exc.name != "scipy":
                    raise
                hubs, authorities = _networkx_hits_scores(graph_nx, **params)
            return {"hubs": hubs, "authorities": authorities}
        if algorithm == "katz_centrality":
            _ensure_networkx_feature(hasattr(nx, "katz_centrality"), compiled_call, "networkx.katz_centrality", nx)
            return nx.katz_centrality(graph_nx, **params)
        if algorithm == "k_core":
            _ensure_networkx_feature(hasattr(nx, "k_core"), compiled_call, "networkx.k_core", nx)
            return nx.k_core(graph_nx, **params)
        if algorithm == "strongly_connected_components":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(hasattr(nx, "strongly_connected_components"), compiled_call, "networkx.strongly_connected_components", nx)
            graph_for_components = graph_nx if graph_nx.is_directed() else graph_nx.to_directed()
            return _networkx_component_labels(nx.strongly_connected_components(graph_for_components))
    except TypeError as exc:
        raise GFQLValidationError(
            ErrorCode.E108,
            f"{compiled_call.procedure} received unsupported algorithm parameters",
            field="call.args",
            value=params,
            suggestion="Use parameters supported by the local graphistry.nx subset for this algorithm.",
            line=compiled_call.line,
            column=compiled_call.column,
            language="cypher",
        ) from exc
    raise ValueError(f"Unexpected NetworkX Cypher CALL algorithm: {algorithm}")


def _raise_networkx_no_params(params: Mapping[str, Any]) -> None:
    if params:
        raise TypeError("algorithm does not accept parameters")


def _networkx_component_labels(components: Any) -> Dict[Any, int]:
    labels: Dict[Any, int] = {}
    for label, component in enumerate(components):
        for node in component:
            labels[node] = label
    return labels


def _networkx_hits_scores(
    graph_nx: Any,
    *,
    max_iter: int = 100,
    tol: float = 1.0e-8,
    nstart: Optional[Mapping[Any, float]] = None,
    normalized: bool = True,
) -> Tuple[Dict[Any, float], Dict[Any, float]]:
    nodes = list(graph_nx.nodes())
    if not nodes:
        return {}, {}

    node_count = len(nodes)
    if nstart is None:
        hubs: Dict[Any, float] = dict.fromkeys(nodes, 1.0 / node_count)
    else:
        total = sum(float(nstart.get(node, 0.0)) for node in nodes)
        norm = total if total != 0 else 1.0
        hubs = {node: float(nstart.get(node, 0.0)) / norm for node in nodes}

    predecessor_fn = graph_nx.predecessors if graph_nx.is_directed() else graph_nx.neighbors
    successor_fn = graph_nx.successors if graph_nx.is_directed() else graph_nx.neighbors
    authorities: Dict[Any, float] = dict.fromkeys(nodes, 1.0 / node_count)

    for _ in range(max_iter):
        last_hubs = hubs
        authorities = {
            node: sum(last_hubs.get(pred, 0.0) for pred in predecessor_fn(node))
            for node in nodes
        }
        authority_norm = sum(abs(value) for value in authorities.values()) or 1.0
        authorities = {node: value / authority_norm for node, value in authorities.items()}
        hubs = {
            node: sum(authorities.get(succ, 0.0) for succ in successor_fn(node))
            for node in nodes
        }
        hub_norm = sum(abs(value) for value in hubs.values()) or 1.0
        hubs = {node: value / hub_norm for node, value in hubs.items()}
        if sum(abs(hubs[node] - last_hubs[node]) for node in nodes) < tol:
            break

    if normalized:
        hub_total = sum(hubs.values()) or 1.0
        authority_total = sum(authorities.values()) or 1.0
        hubs = {node: value / hub_total for node, value in hubs.items()}
        authorities = {node: value / authority_total for node, value in authorities.items()}

    return hubs, authorities


def _networkx_pagerank_scores(graph_nx: Any, *, alpha: float = 0.85, max_iter: int = 100, tol: float = 1.0e-6) -> Dict[Any, float]:
    nodes = list(graph_nx.nodes())
    if not nodes:
        return {}
    node_count = len(nodes)
    scores: Dict[Any, float] = dict.fromkeys(nodes, 1.0 / node_count)
    degree_fn, predecessor_fn = (graph_nx.out_degree, graph_nx.predecessors) if graph_nx.is_directed() else (graph_nx.degree, graph_nx.neighbors)
    for _ in range(max_iter):
        dangling_mass = alpha * sum(scores[node] for node in nodes if degree_fn(node) == 0) / node_count
        next_scores = {
            node: ((1.0 - alpha) / node_count) + dangling_mass + alpha * sum(scores[pred] / degree_fn(pred) for pred in predecessor_fn(node) if degree_fn(pred) > 0)
            for node in nodes
        }
        if sum(abs(next_scores[node] - scores[node]) for node in nodes) <= tol * node_count:
            return next_scores
        scores = next_scores
    return scores


def _networkx_out_col(compiled_call: CompiledCypherProcedureCall, default: str) -> str:
    out_col = compiled_call.call_params.get("out_col")
    if isinstance(out_col, str):
        return out_col
    return compiled_call.algorithm or default


def _networkx_graph(
    base_graph: Plottable,
    compiled_call: CompiledCypherProcedureCall,
    *,
    directed: bool,
) -> Tuple[Plottable, Any]:
    nx = _networkx_module(compiled_call)
    graph_with_nodes = materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None

    nodes_pdf = as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))
    edges_df = graph_with_nodes._edges
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else as_pandas_frame(edges_df, edge_columns)

    graph_nx = nx.DiGraph() if directed else nx.Graph()
    graph_nx.add_nodes_from(nodes_pdf[graph_with_nodes._node].tolist())
    if not edges_pdf.empty:
        graph_nx.add_edges_from(edges_pdf[list(edge_columns)].itertuples(index=False, name=None))
    return graph_with_nodes, graph_nx


def _networkx_common_inputs(compiled_call: CompiledCypherProcedureCall) -> Tuple[bool, Dict[str, Any]]:
    return cast(bool, compiled_call.call_params.get("directed", True)), dict(cast(Mapping[str, Any], compiled_call.call_params.get("params", {})))


def networkx_node_rows(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> DataFrameT:
    directed, params = _networkx_common_inputs(compiled_call)
    graph_with_nodes, graph_nx = _networkx_graph(base_graph, compiled_call, directed=directed)
    assert graph_with_nodes._node is not None
    nodes_pdf = as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))

    scores = _networkx_algorithm_result(compiled_call, graph_nx, params)
    value_columns = networkx_normalized_value_columns(compiled_call)
    rows_pdf = nodes_pdf.rename(columns={graph_with_nodes._node: "nodeId"}).copy()
    if isinstance(scores, Mapping) and all(isinstance(scores.get(col), Mapping) for col in value_columns):
        for col in value_columns:
            rows_pdf[col] = rows_pdf["nodeId"].map(scores[col]).fillna(0.0)
    else:
        rows_pdf[value_columns[0]] = rows_pdf["nodeId"].map(scores).fillna(0.0)
    return cast(DataFrameT, rows_pdf)


def networkx_edge_rows(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> DataFrameT:
    directed, params = _networkx_common_inputs(compiled_call)
    graph_with_nodes, graph_nx = _networkx_graph(base_graph, compiled_call, directed=directed)
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_df = graph_with_nodes._edges
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else as_pandas_frame(edges_df, edge_columns)

    scores = _networkx_algorithm_result(compiled_call, graph_nx, params)
    out_col = _networkx_out_col(compiled_call, "edge_betweenness_centrality")
    rows_pdf = edges_pdf.rename(columns={graph_with_nodes._source: "source", graph_with_nodes._destination: "destination"}).copy()
    if directed:
        rows_pdf[out_col] = rows_pdf.apply(lambda row: scores.get((row["source"], row["destination"]), 0.0), axis=1)
    else:
        rows_pdf[out_col] = rows_pdf.apply(
            lambda row: scores.get((row["source"], row["destination"]), scores.get((row["destination"], row["source"]), 0.0)),
            axis=1,
        )
    return cast(DataFrameT, rows_pdf)


def _merge_networkx_projected_graph(base_graph: Plottable, graph_nx: Any) -> Plottable:
    graph_with_nodes = materialized_graph(base_graph)
    projected = graph_with_nodes.from_networkx(graph_nx)

    if projected._nodes is not None and graph_with_nodes._nodes is not None and graph_with_nodes._node is not None:
        base_nodes_trimmed = graph_with_nodes._nodes[
            [x for x in graph_with_nodes._nodes.columns if x not in projected._nodes.columns or x == graph_with_nodes._node]
        ]
        merged_nodes = cast(DataFrameT, safe_merge(projected._nodes, base_nodes_trimmed, on=graph_with_nodes._node, how="left"))
        projected = projected.nodes(merged_nodes, graph_with_nodes._node)

    if (
        projected._edges is not None
        and graph_with_nodes._edges is not None
        and graph_with_nodes._source is not None
        and graph_with_nodes._destination is not None
    ):
        edge_keys = [graph_with_nodes._source, graph_with_nodes._destination]
        base_edges_trimmed = graph_with_nodes._edges[
            [x for x in graph_with_nodes._edges.columns if x not in projected._edges.columns or x in edge_keys]
        ]
        merged_edges = cast(DataFrameT, safe_merge(projected._edges, base_edges_trimmed, on=edge_keys, how="left"))
        projected = projected.edges(merged_edges, graph_with_nodes._source, graph_with_nodes._destination)

    return projected


def execute_networkx_graph_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.row_kind == "node":
        rows_df = networkx_node_rows(base_graph, compiled_call)
        value_columns = networkx_normalized_value_columns(compiled_call)
        if len(value_columns) > 1:
            return merge_node_property_columns(base_graph, rows_df, value_columns=value_columns)
        out_col = value_columns[0]
        return merge_node_property_rows(
            base_graph,
            rows_df,
            source_value_col=out_col,
            output_value_col=out_col,
        )
    if compiled_call.row_kind == "edge":
        rows_df = networkx_edge_rows(base_graph, compiled_call)
        out_col = _networkx_out_col(compiled_call, "edge_betweenness_centrality")
        return merge_edge_property_rows(
            base_graph,
            rows_df,
            source_value_col=out_col,
            output_value_col=out_col,
        )

    directed, params = _networkx_common_inputs(compiled_call)
    _, graph_nx = _networkx_graph(base_graph, compiled_call, directed=directed)
    projected_graph = _networkx_algorithm_result(compiled_call, graph_nx, params)
    return _merge_networkx_projected_graph(base_graph, projected_graph)
