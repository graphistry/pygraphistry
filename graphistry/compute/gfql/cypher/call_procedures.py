from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple, cast

import pandas as pd

from graphistry.Engine import safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.ast import CallClause
from graphistry.compute.typing import DataFrameT
from graphistry.compute.util.generate_safe_column_name import generate_safe_column_name


@dataclass(frozen=True)
class ProcedureOutputColumn:
    source_name: str
    output_name: str


@dataclass(frozen=True)
class _ProcedureDefinition:
    procedure: str
    kind: str
    default_outputs: Tuple[str, ...] = ()
    result_kind: Literal["rows", "graph"] = "rows"
    write_property: Optional[str] = None


@dataclass(frozen=True)
class CompiledCypherProcedureCall:
    procedure: str
    kind: str
    output_columns: Tuple[ProcedureOutputColumn, ...] = ()
    result_kind: Literal["rows", "graph"] = "rows"
    write_property: Optional[str] = None


_PROCEDURE_DEFS: Dict[str, _ProcedureDefinition] = {
    "graphistry.degree": _ProcedureDefinition(
        procedure="graphistry.degree",
        kind="degree",
        default_outputs=("nodeId", "degree", "degree_in", "degree_out"),
    ),
    "graphistry.degree.write": _ProcedureDefinition(
        procedure="graphistry.degree.write",
        kind="degree_write",
        result_kind="graph",
    ),
    "graphistry.igraph.pagerank": _ProcedureDefinition(
        procedure="graphistry.igraph.pagerank",
        kind="igraph_pagerank",
        default_outputs=("nodeId", "score"),
    ),
    "graphistry.igraph.pagerank.write": _ProcedureDefinition(
        procedure="graphistry.igraph.pagerank.write",
        kind="igraph_pagerank_write",
        result_kind="graph",
        write_property="pagerank",
    ),
    "graphistry.cugraph.pagerank": _ProcedureDefinition(
        procedure="graphistry.cugraph.pagerank",
        kind="cugraph_pagerank",
        default_outputs=("nodeId", "score"),
    ),
    "graphistry.cugraph.pagerank.write": _ProcedureDefinition(
        procedure="graphistry.cugraph.pagerank.write",
        kind="cugraph_pagerank_write",
        result_kind="graph",
        write_property="pagerank",
    ),
    "graphistry.nx.pagerank": _ProcedureDefinition(
        procedure="graphistry.nx.pagerank",
        kind="networkx_pagerank",
        default_outputs=("nodeId", "score"),
    ),
    "graphistry.nx.pagerank.write": _ProcedureDefinition(
        procedure="graphistry.nx.pagerank.write",
        kind="networkx_pagerank_write",
        result_kind="graph",
        write_property="pagerank",
    ),
}


def _unsupported_call(message: str, *, call: CallClause, value: Any) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field="call",
        value=value,
        suggestion="Use one of the supported Graphistry procedures or remove unsupported CALL syntax.",
        line=call.span.line,
        column=call.span.column,
        language="cypher",
    )


def _compile_graph_call(
    definition: _ProcedureDefinition,
    call: CallClause,
) -> CompiledCypherProcedureCall:
    if call.args:
        raise _unsupported_call(
            "Graph-preserving Cypher CALL procedures do not accept arguments in the current local subset",
            call=call,
            value=call.procedure,
        )
    if call.yield_items:
        raise _unsupported_call(
            "Graph-preserving Cypher CALL procedures do not support YIELD in the current local subset",
            call=call,
            value=call.procedure,
        )
    return CompiledCypherProcedureCall(
        procedure=definition.procedure,
        kind=definition.kind,
        result_kind="graph",
        write_property=definition.write_property,
    )


def compile_cypher_call(call: CallClause) -> CompiledCypherProcedureCall:
    definition = _PROCEDURE_DEFS.get(call.procedure)
    if definition is None:
        raise _unsupported_call(
            "Unsupported Cypher CALL procedure in the local compiler",
            call=call,
            value=call.procedure,
        )
    if definition.result_kind == "graph":
        return _compile_graph_call(definition, call)
    if call.args:
        raise _unsupported_call(
            "Graphistry Cypher CALL procedures do not accept arguments in the current local subset",
            call=call,
            value=call.procedure,
        )
    available_outputs = set(definition.default_outputs)
    if not call.yield_items:
        output_columns = tuple(
            ProcedureOutputColumn(source_name=name, output_name=name)
            for name in definition.default_outputs
        )
        return CompiledCypherProcedureCall(
            procedure=definition.procedure,
            kind=definition.kind,
            output_columns=output_columns,
        )

    seen_output_names: set[str] = set()
    compiled_outputs = []
    for yield_item in call.yield_items:
        if yield_item.name not in available_outputs:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher YIELD references an unknown procedure output in the local compiler",
                field="yield",
                value=yield_item.name,
                suggestion=f"Use one of: {', '.join(definition.default_outputs)}.",
                line=yield_item.span.line,
                column=yield_item.span.column,
                language="cypher",
            )
        output_name = yield_item.alias or yield_item.name
        if output_name in seen_output_names:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher YIELD names must be unique in the local compiler",
                field="yield",
                value=output_name,
                suggestion="Rename duplicate YIELD outputs with AS or remove duplicates.",
                line=yield_item.span.line,
                column=yield_item.span.column,
                language="cypher",
            )
        seen_output_names.add(output_name)
        compiled_outputs.append(
            ProcedureOutputColumn(source_name=yield_item.name, output_name=output_name)
        )
    return CompiledCypherProcedureCall(
        procedure=definition.procedure,
        kind=definition.kind,
        output_columns=tuple(compiled_outputs),
    )


def _materialized_graph(base_graph: Plottable) -> Plottable:
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


def _graph_row_result(base_graph: Plottable, rows_df: DataFrameT) -> Plottable:
    out = base_graph.bind()
    out._nodes = rows_df
    edges_df = getattr(base_graph, "_edges", None)
    if edges_df is not None:
        out._edges = cast(DataFrameT, edges_df[:0])
    else:
        out._edges = cast(DataFrameT, rows_df[:0])
    out._node = str(rows_df.columns[0]) if len(rows_df.columns) > 0 else None
    return out


def _project_outputs(rows_df: DataFrameT, outputs: Tuple[ProcedureOutputColumn, ...]) -> DataFrameT:
    projected = {
        output.output_name: rows_df[output.source_name]
        for output in outputs
    }
    return cast(DataFrameT, rows_df.assign(**projected)[[output.output_name for output in outputs]])


def _degree_rows(base_graph: Plottable) -> DataFrameT:
    graph_with_degrees = _materialized_graph(base_graph).get_degrees()
    assert graph_with_degrees._nodes is not None
    assert graph_with_degrees._node is not None
    node_id_col = graph_with_degrees._node
    return cast(
        DataFrameT,
        graph_with_degrees._nodes.rename(columns={node_id_col: "nodeId"})[
            ["nodeId", "degree", "degree_in", "degree_out"]
        ],
    )


def _graph_algo_rows(base_graph: Plottable, *, function: str) -> DataFrameT:
    graph_with_nodes = _materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    score_col = generate_safe_column_name(
        "__cypher_call_score__",
        graph_with_nodes._nodes,
        prefix="__gfql_",
        suffix="__",
    )
    if function == "compute_igraph":
        computed = graph_with_nodes.compute_igraph("pagerank", out_col=score_col)
    elif function == "compute_cugraph":
        computed = graph_with_nodes.compute_cugraph("pagerank", out_col=score_col)
    else:
        raise ValueError(f"Unexpected graph algorithm function: {function}")
    assert computed._nodes is not None
    assert computed._node is not None
    return cast(
        DataFrameT,
        computed._nodes.rename(columns={computed._node: "nodeId", score_col: "score"})[
            ["nodeId", "score"]
        ],
    )


def _as_pandas_frame(df: Any, columns: Tuple[str, ...]) -> pd.DataFrame:
    subset = df[list(columns)]
    if hasattr(subset, "to_pandas"):
        return cast(pd.DataFrame, subset.to_pandas())
    return pd.DataFrame(subset)


def _networkx_pagerank_scores(
    graph_nx: Any,
    *,
    alpha: float = 0.85,
    max_iter: int = 100,
    tol: float = 1.0e-6,
) -> Dict[Any, float]:
    nodes = list(graph_nx.nodes())
    node_count = len(nodes)
    if node_count == 0:
        return {}

    base_score = 1.0 / node_count
    scores: Dict[Any, float] = {node: base_score for node in nodes}
    dangling_nodes = [node for node in nodes if graph_nx.out_degree(node) == 0]

    for _ in range(max_iter):
        dangling_mass = alpha * sum(scores[node] for node in dangling_nodes) / node_count
        next_scores: Dict[Any, float] = {}
        delta = 0.0
        for node in nodes:
            inbound_mass = 0.0
            for predecessor in graph_nx.predecessors(node):
                out_degree = graph_nx.out_degree(predecessor)
                if out_degree > 0:
                    inbound_mass += scores[predecessor] / out_degree
            next_score = ((1.0 - alpha) / node_count) + dangling_mass + (alpha * inbound_mass)
            next_scores[node] = next_score
            delta += abs(next_score - scores[node])
        scores = next_scores
        if delta <= tol * node_count:
            break
    return scores


def _networkx_pagerank_rows(base_graph: Plottable) -> DataFrameT:
    try:
        import networkx as nx
    except ImportError as exc:
        raise GFQLValidationError(
            ErrorCode.E108,
            "graphistry.nx.pagerank requires the optional 'networkx' dependency",
            field="call",
            value="graphistry.nx.pagerank",
            suggestion="Install networkx or use graphistry.igraph.pagerank / graphistry.cugraph.pagerank.",
            language="cypher",
        ) from exc

    graph_with_nodes = _materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None
    nodes_pdf = _as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))
    edges_df = getattr(graph_with_nodes, "_edges", None)
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else _as_pandas_frame(edges_df, edge_columns)

    graph_nx = nx.DiGraph()
    graph_nx.add_nodes_from(nodes_pdf[graph_with_nodes._node].tolist())
    if not edges_pdf.empty:
        graph_nx.add_edges_from(edges_pdf[list(edge_columns)].itertuples(index=False, name=None))
    scores = _networkx_pagerank_scores(graph_nx)
    rows_pdf = nodes_pdf.rename(columns={graph_with_nodes._node: "nodeId"}).copy()
    rows_pdf["score"] = rows_pdf["nodeId"].map(scores).fillna(0.0)
    return cast(DataFrameT, rows_pdf)


def _merge_node_property_rows(
    base_graph: Plottable,
    rows_df: DataFrameT,
    *,
    source_value_col: str,
    output_value_col: str,
) -> Plottable:
    graph_with_nodes = _materialized_graph(base_graph)
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


def _execute_graph_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.kind == "degree_write":
        return _materialized_graph(base_graph).get_degrees()
    if compiled_call.kind == "igraph_pagerank_write":
        return _materialized_graph(base_graph).compute_igraph(
            "pagerank",
            out_col=compiled_call.write_property or "pagerank",
        )
    if compiled_call.kind == "cugraph_pagerank_write":
        return _materialized_graph(base_graph).compute_cugraph(
            "pagerank",
            out_col=compiled_call.write_property or "pagerank",
        )
    if compiled_call.kind == "networkx_pagerank_write":
        rows_df = _networkx_pagerank_rows(base_graph)
        return _merge_node_property_rows(
            base_graph,
            rows_df,
            source_value_col="score",
            output_value_col=compiled_call.write_property or "pagerank",
        )
    raise ValueError(f"Unexpected graph-preserving Cypher CALL kind: {compiled_call.kind}")


def execute_cypher_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.result_kind == "graph":
        return _execute_graph_call(base_graph, compiled_call)
    if compiled_call.kind == "degree":
        default_rows = _degree_rows(base_graph)
    elif compiled_call.kind == "igraph_pagerank":
        try:
            default_rows = _graph_algo_rows(base_graph, function="compute_igraph")
        except ImportError as exc:
            raise GFQLValidationError(
                ErrorCode.E108,
                "graphistry.igraph.pagerank requires the optional 'igraph' dependency",
                field="call",
                value=compiled_call.procedure,
                suggestion="Install python-igraph or use another supported Graphistry CALL backend.",
                language="cypher",
            ) from exc
    elif compiled_call.kind == "cugraph_pagerank":
        default_rows = _graph_algo_rows(base_graph, function="compute_cugraph")
    elif compiled_call.kind == "networkx_pagerank":
        default_rows = _networkx_pagerank_rows(base_graph)
    else:
        raise ValueError(f"Unexpected compiled Cypher CALL kind: {compiled_call.kind}")

    return _graph_row_result(base_graph, _project_outputs(default_rows, compiled_call.output_columns))
