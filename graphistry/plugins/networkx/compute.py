from __future__ import annotations

from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, cast

import pandas as pd

from graphistry.Engine import safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.engine_coercion import ensure_pandas, restore_engine
from graphistry.compute.typing import DataFrameT
from graphistry.plugins.networkx.policy import networkx_version_error, scipy_version_error


NetworkXResultKind = Literal["node", "edge", "graph"]

NETWORKX_COMPUTE_ALGORITHMS: Dict[str, Tuple[NetworkXResultKind, Tuple[str, ...]]] = {
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
    "k_core": ("graph", ()),
    "strongly_connected_components": ("node", ("labels",)),
}

compute_algs = tuple(NETWORKX_COMPUTE_ALGORITHMS.keys())


def _networkx_module() -> Any:
    try:
        import networkx as nx
    except ImportError as exc:
        raise ImportError(
            "compute_networkx() requires the optional 'networkx' dependency; install pygraphistry[networkx]."
        ) from exc
    error = networkx_version_error(getattr(nx, "__version__", None))
    if error is not None:
        raise ValueError(error)
    return nx


def _ensure_scipy_version_policy() -> None:
    try:
        import scipy
    except ImportError:
        return
    error = scipy_version_error(getattr(scipy, "__version__", None))
    if error is not None:
        raise ValueError(error)


def _ensure_networkx_feature(nx: Any, feature_name: str) -> None:
    if hasattr(nx, feature_name):
        return
    nx_version = getattr(nx, "__version__", "unknown")
    raise ValueError(
        f"compute_networkx() requires networkx.{feature_name}, which is unavailable in the installed NetworkX version "
        f"{nx_version}."
    )


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


def _networkx_pagerank_scores(
    graph_nx: Any,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> Dict[Any, float]:
    nodes = list(graph_nx.nodes())
    if not nodes:
        return {}
    node_count = len(nodes)
    scores: Dict[Any, float] = dict.fromkeys(nodes, 1.0 / node_count)
    degree_fn, predecessor_fn = (
        (graph_nx.out_degree, graph_nx.predecessors)
        if graph_nx.is_directed()
        else (graph_nx.degree, graph_nx.neighbors)
    )
    for _ in range(max_iter):
        dangling_mass = alpha * sum(scores[node] for node in nodes if degree_fn(node) == 0) / node_count
        next_scores = {
            node: ((1.0 - alpha) / node_count)
            + dangling_mass
            + alpha * sum(scores[pred] / degree_fn(pred) for pred in predecessor_fn(node) if degree_fn(pred) > 0)
            for node in nodes
        }
        if sum(abs(next_scores[node] - scores[node]) for node in nodes) <= tol * node_count:
            return next_scores
        scores = next_scores
    return scores


def _networkx_algorithm_result(alg: str, graph_nx: Any, params: Mapping[str, Any], nx: Any) -> Any:
    try:
        if alg == "pagerank":
            return _networkx_pagerank_scores(graph_nx, **params)
        if alg == "betweenness_centrality":
            _ensure_networkx_feature(nx, "betweenness_centrality")
            return nx.betweenness_centrality(graph_nx, **params)
        if alg == "closeness_centrality":
            _ensure_networkx_feature(nx, "closeness_centrality")
            return nx.closeness_centrality(graph_nx, **params)
        if alg == "connected_components":
            _raise_networkx_no_params(params)
            if graph_nx.is_directed():
                _ensure_networkx_feature(nx, "weakly_connected_components")
                components = nx.weakly_connected_components(graph_nx)
            else:
                _ensure_networkx_feature(nx, "connected_components")
                components = nx.connected_components(graph_nx)
            return _networkx_component_labels(components)
        if alg == "core_number":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(nx, "core_number")
            return nx.core_number(graph_nx)
        if alg == "degree_centrality":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(nx, "degree_centrality")
            return nx.degree_centrality(graph_nx)
        if alg == "eigenvector_centrality":
            _ensure_networkx_feature(nx, "eigenvector_centrality")
            return nx.eigenvector_centrality(graph_nx, **params)
        if alg == "edge_betweenness_centrality":
            _ensure_networkx_feature(nx, "edge_betweenness_centrality")
            return nx.edge_betweenness_centrality(graph_nx, **params)
        if alg == "hits":
            _ensure_networkx_feature(nx, "hits")
            _ensure_scipy_version_policy()
            try:
                hubs, authorities = nx.hits(graph_nx, **params)
            except ModuleNotFoundError as exc:
                if exc.name != "scipy":
                    raise
                hubs, authorities = _networkx_hits_scores(graph_nx, **params)
            return {"hubs": hubs, "authorities": authorities}
        if alg == "katz_centrality":
            _ensure_networkx_feature(nx, "katz_centrality")
            return nx.katz_centrality(graph_nx, **params)
        if alg == "k_core":
            _ensure_networkx_feature(nx, "k_core")
            return nx.k_core(graph_nx, **params)
        if alg == "strongly_connected_components":
            _raise_networkx_no_params(params)
            _ensure_networkx_feature(nx, "strongly_connected_components")
            graph_for_components = graph_nx if graph_nx.is_directed() else graph_nx.to_directed()
            return _networkx_component_labels(nx.strongly_connected_components(graph_for_components))
    except TypeError as exc:
        raise ValueError(f"compute_networkx({alg!r}) received unsupported algorithm parameters: {dict(params)!r}") from exc
    raise ValueError(f"Unsupported NetworkX algorithm: {alg}")


def _materialized_graph(base_graph: Plottable) -> Plottable:
    graph_with_nodes = base_graph.materialize_nodes()
    if graph_with_nodes._nodes is None or graph_with_nodes._node is None:
        raise ValueError("compute_networkx() requires a materialized node table; bind or materialize nodes first.")
    return graph_with_nodes


def _as_pandas_frame(df: Any, columns: Sequence[str]) -> pd.DataFrame:
    return ensure_pandas(df[list(columns)]).copy()


def _networkx_graph(base_graph: Plottable, nx: Any, *, directed: bool, G: Optional[Any]) -> Tuple[Plottable, Any]:
    graph_with_nodes = _materialized_graph(base_graph)
    if G is not None:
        return graph_with_nodes, G

    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None

    nodes_pdf = _as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_df = getattr(graph_with_nodes, "_edges", None)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else _as_pandas_frame(edges_df, edge_columns)

    graph_nx = nx.DiGraph() if directed else nx.Graph()
    graph_nx.add_nodes_from(nodes_pdf[graph_with_nodes._node].tolist())
    if not edges_pdf.empty:
        graph_nx.add_edges_from(edges_pdf[list(edge_columns)].itertuples(index=False, name=None))
    return graph_with_nodes, graph_nx


def _normalized_value_columns(alg: str, out_col: Optional[str]) -> Tuple[str, ...]:
    _, value_columns = NETWORKX_COMPUTE_ALGORITHMS[alg]
    if out_col is None or not value_columns:
        return value_columns
    if len(value_columns) > 1:
        raise ValueError("compute_networkx() does not allow out_col for multi-column algorithms such as hits")
    return (out_col,)


def _networkx_node_rows(
    graph_with_nodes: Plottable,
    graph_nx: Any,
    *,
    alg: str,
    params: Mapping[str, Any],
    value_columns: Tuple[str, ...],
    nx: Any,
) -> pd.DataFrame:
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    nodes_pdf = _as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))
    scores = _networkx_algorithm_result(alg, graph_nx, params, nx)
    rows_pdf = nodes_pdf.rename(columns={graph_with_nodes._node: "nodeId"}).copy()
    if isinstance(scores, Mapping) and all(isinstance(scores.get(col), Mapping) for col in value_columns):
        for col in value_columns:
            rows_pdf[col] = rows_pdf["nodeId"].map(scores[col]).fillna(0.0)
    else:
        rows_pdf[value_columns[0]] = rows_pdf["nodeId"].map(scores).fillna(0.0)
    return rows_pdf


def _networkx_edge_rows(
    graph_with_nodes: Plottable,
    graph_nx: Any,
    *,
    alg: str,
    params: Mapping[str, Any],
    out_col: str,
    directed: bool,
    nx: Any,
) -> pd.DataFrame:
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_df = getattr(graph_with_nodes, "_edges", None)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else _as_pandas_frame(edges_df, edge_columns)
    scores = _networkx_algorithm_result(alg, graph_nx, params, nx)
    rows_pdf = edges_pdf.rename(columns={graph_with_nodes._source: "source", graph_with_nodes._destination: "destination"}).copy()
    src_values = rows_pdf["source"].tolist()
    dst_values = rows_pdf["destination"].tolist()
    if directed:
        rows_pdf[out_col] = [scores.get((src, dst), 0.0) for src, dst in zip(src_values, dst_values)]
    else:
        rows_pdf[out_col] = [
            scores.get((src, dst), scores.get((dst, src), 0.0))
            for src, dst in zip(src_values, dst_values)
        ]
    return rows_pdf


def _merge_node_property_columns(base_graph: Plottable, rows_df: pd.DataFrame, value_columns: Tuple[str, ...]) -> Plottable:
    assert base_graph._nodes is not None
    assert base_graph._node is not None
    node_col = base_graph._node
    nodes_pdf = ensure_pandas(base_graph._nodes)
    existing_value_columns = [col for col in value_columns if col in nodes_pdf.columns]
    merge_base = nodes_pdf.drop(columns=existing_value_columns) if existing_value_columns else nodes_pdf
    projected_rows = rows_df.rename(columns={"nodeId": node_col})[[node_col, *value_columns]]
    merged_nodes = cast(DataFrameT, safe_merge(merge_base, projected_rows, on=node_col, how="left"))
    return base_graph.nodes(merged_nodes, node=node_col)


def _merge_edge_property_rows(base_graph: Plottable, rows_df: pd.DataFrame, source_value_col: str, output_value_col: str) -> Plottable:
    if base_graph._edges is None or base_graph._source is None or base_graph._destination is None:
        raise ValueError("compute_networkx() requires an edge table for edge-enriching algorithms.")
    source_col = base_graph._source
    destination_col = base_graph._destination
    edges_pdf = ensure_pandas(base_graph._edges)
    merge_base = edges_pdf.drop(columns=[output_value_col]) if output_value_col in edges_pdf.columns else edges_pdf
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
    return base_graph.edges(merged_edges, source_col, destination_col)


def _merge_undirected_reverse_edge_attrs(
    merged_edges: pd.DataFrame,
    base_edges: pd.DataFrame,
    edge_keys: Sequence[str],
) -> pd.DataFrame:
    source_col, destination_col = edge_keys
    value_columns = [col for col in base_edges.columns if col not in edge_keys]
    if not value_columns:
        return merged_edges

    reverse_source_col = "__graphistry_reverse_source__"
    reversed_base = base_edges.rename(
        columns={
            source_col: reverse_source_col,
            destination_col: source_col,
        }
    ).rename(columns={reverse_source_col: destination_col})
    reverse_value_columns = {col: f"__graphistry_reverse_{col}__" for col in value_columns}
    reversed_base = reversed_base.rename(columns=reverse_value_columns)
    reverse_projected = reversed_base[[source_col, destination_col, *reverse_value_columns.values()]]
    merged_with_reverse = cast(
        pd.DataFrame,
        safe_merge(merged_edges, reverse_projected, on=list(edge_keys), how="left"),
    )
    for col, reverse_col in reverse_value_columns.items():
        merged_with_reverse[col] = merged_with_reverse[col].where(
            merged_with_reverse[col].notna(),
            merged_with_reverse[reverse_col],
        )
    return merged_with_reverse.drop(columns=list(reverse_value_columns.values()))


def _merge_networkx_projected_graph(base_graph: Plottable, graph_nx: Any) -> Plottable:
    graph_with_nodes = _materialized_graph(base_graph)
    projected = graph_with_nodes.from_networkx(graph_nx)

    if projected._nodes is not None and graph_with_nodes._nodes is not None and graph_with_nodes._node is not None:
        projected_nodes_pdf = ensure_pandas(projected._nodes)
        base_nodes_pdf = ensure_pandas(graph_with_nodes._nodes)
        base_nodes_trimmed = base_nodes_pdf[
            [col for col in base_nodes_pdf.columns if col not in projected_nodes_pdf.columns or col == graph_with_nodes._node]
        ]
        merged_nodes = cast(DataFrameT, safe_merge(projected_nodes_pdf, base_nodes_trimmed, on=graph_with_nodes._node, how="left"))
        projected = projected.nodes(merged_nodes, graph_with_nodes._node)

    if (
        projected._edges is not None
        and graph_with_nodes._edges is not None
        and graph_with_nodes._source is not None
        and graph_with_nodes._destination is not None
    ):
        edge_keys = [graph_with_nodes._source, graph_with_nodes._destination]
        projected_edges_pdf = ensure_pandas(projected._edges)
        base_edges_pdf = ensure_pandas(graph_with_nodes._edges)
        base_edges_trimmed = base_edges_pdf[
            [col for col in base_edges_pdf.columns if col not in projected_edges_pdf.columns or col in edge_keys]
        ]
        merged_edges_pdf = cast(pd.DataFrame, safe_merge(projected_edges_pdf, base_edges_trimmed, on=edge_keys, how="left"))
        if not graph_nx.is_directed():
            merged_edges_pdf = _merge_undirected_reverse_edge_attrs(merged_edges_pdf, base_edges_trimmed, edge_keys)
        merged_edges = cast(DataFrameT, merged_edges_pdf)
        projected = projected.edges(merged_edges, graph_with_nodes._source, graph_with_nodes._destination)

    return projected


def compute_networkx(
    self: Plottable,
    alg: str,
    out_col: Optional[str] = None,
    params: Optional[Mapping[str, Any]] = None,
    directed: bool = True,
    G: Optional[Any] = None,
) -> Plottable:
    """Run a supported NetworkX algorithm and return an enriched PyGraphistry graph.

    :param alg: Explicitly supported NetworkX algorithm name.
    :type alg: str
    :param out_col: Optional output column name for single-column node/edge algorithms.
    :type out_col: Optional[str]
    :param params: Keyword parameters forwarded to the selected NetworkX algorithm.
    :type params: Optional[Mapping[str, Any]]
    :param directed: Whether to build a directed NetworkX graph from the current graph.
    :type directed: bool
    :param G: Optional prebuilt NetworkX graph to run instead of converting ``self``.
    :type G: Optional[Any]
    :returns: Plotter
    :rtype: Plotter

    **Example: Degree centrality**
        ::

            g2 = g.compute_networkx("degree_centrality", out_col="degree_score")

    **Example: HITS**
        ::

            g2 = g.compute_networkx("hits")
            assert "hubs" in g2._nodes.columns
            assert "authorities" in g2._nodes.columns
    """

    if alg not in NETWORKX_COMPUTE_ALGORITHMS:
        supported = ", ".join(compute_algs)
        raise ValueError(f"Unsupported NetworkX algorithm {alg!r}; supported algorithms: {supported}")

    params_dict = dict(params or {})
    original_nodes = self._nodes
    original_edges = self._edges
    nx = _networkx_module()
    graph_with_nodes, graph_nx = _networkx_graph(self, nx, directed=directed, G=G)
    result_kind, _ = NETWORKX_COMPUTE_ALGORITHMS[alg]
    value_columns = _normalized_value_columns(alg, out_col)

    if result_kind == "node":
        rows_df = _networkx_node_rows(
            graph_with_nodes,
            graph_nx,
            alg=alg,
            params=params_dict,
            value_columns=value_columns,
            nx=nx,
        )
        out = _merge_node_property_columns(graph_with_nodes, rows_df, value_columns)
        return restore_engine(out, original_nodes, original_edges)

    if result_kind == "edge":
        edge_out_col = value_columns[0]
        rows_df = _networkx_edge_rows(
            graph_with_nodes,
            graph_nx,
            alg=alg,
            params=params_dict,
            out_col=edge_out_col,
            directed=directed,
            nx=nx,
        )
        out = _merge_edge_property_rows(graph_with_nodes, rows_df, edge_out_col, edge_out_col)
        return restore_engine(out, original_nodes, original_edges)

    if out_col is not None:
        raise ValueError("compute_networkx() does not allow out_col for graph-returning algorithms")
    projected_graph = _networkx_algorithm_result(alg, graph_nx, params_dict, nx)
    out = _merge_networkx_projected_graph(graph_with_nodes, projected_graph)
    return restore_engine(out, original_nodes, original_edges)
