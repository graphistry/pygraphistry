from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, cast

import pandas as pd

from graphistry.Engine import safe_merge
from graphistry.Plottable import Plottable
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
from graphistry.compute.gfql.call.validation import validate_call_params
from graphistry.compute.gfql.cypher.ast import CallClause
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    ExprNode,
    FunctionCall,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    UnaryOp,
)
from graphistry.compute.typing import DataFrameT
from graphistry.plugins.cugraph import (
    compute_algs as _CUGRAPH_COMPUTE_ALGS,
    edge_compute_algs_to_attr,
    node_compute_algs_to_attr,
)
from graphistry.plugins.igraph import compute_algs as _IGRAPH_COMPUTE_ALGS


_ROW_KIND = Literal["degree", "node", "edge", "graph_only", "node_or_graph"]
_BACKEND = Literal["degree", "cugraph", "igraph", "networkx"]

_DEGREE_OUTPUTS: Tuple[str, ...] = ("nodeId", "degree", "degree_in", "degree_out")
_CUGRAPH_RESERVED_KEYS = frozenset({"out_col", "params", "kind", "directed", "G"})
_IGRAPH_RESERVED_KEYS = frozenset({"out_col", "directed", "use_vids", "params"})
_NETWORKX_RESERVED_KEYS = frozenset({"out_col", "params", "directed"})
_NETWORKX_NODE_ALGORITHMS: Dict[str, Tuple[str, ...]] = {
    "pagerank": ("pagerank",),
    "betweenness_centrality": ("betweenness_centrality",),
}
_NETWORKX_EDGE_ALGORITHMS: Dict[str, Tuple[str, ...]] = {
    "edge_betweenness_centrality": ("edge_betweenness_centrality",),
}
_NETWORKX_GRAPH_ALGORITHMS = frozenset({"k_core"})


@dataclass(frozen=True)
class ProcedureOutputColumn:
    source_name: str
    output_name: str


@dataclass(frozen=True)
class _ProcedureDefinition:
    procedure: str
    backend: _BACKEND
    algorithm: Optional[str] = None
    call_function: Optional[str] = None
    result_kind: Literal["rows", "graph"] = "rows"
    row_kind: _ROW_KIND = "node"


@dataclass(frozen=True)
class CompiledCypherProcedureCall:
    procedure: str
    backend: _BACKEND
    algorithm: Optional[str] = None
    output_columns: Tuple[ProcedureOutputColumn, ...] = ()
    result_kind: Literal["rows", "graph"] = "rows"
    row_kind: _ROW_KIND = "node"
    call_function: Optional[str] = None
    call_params: Mapping[str, Any] = field(default_factory=dict)
    line: int = 1
    column: int = 1


def _unsupported_call(message: str, *, call: CallClause, value: Any) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field="call",
        value=value,
        suggestion="Use graphistry.degree, supported graphistry.igraph.* / graphistry.cugraph.* procedures, or the limited graphistry.nx pagerank / betweenness / edge_betweenness / k_core subset.",
        line=call.span.line,
        column=call.span.column,
        language="cypher",
    )


def _invalid_call_argument(
    message: str,
    *,
    call: CallClause,
    value: Any,
    field: str = "call.args",
) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        message,
        field=field,
        value=value,
        suggestion="Use no arguments or a single literal / parameter map of compute_* options.",
        line=call.span.line,
        column=call.span.column,
        language="cypher",
    )


def _resolve_procedure_definition(call: CallClause) -> _ProcedureDefinition:
    if call.procedure == "graphistry.degree":
        return _ProcedureDefinition(
            procedure=call.procedure,
            backend="degree",
            result_kind="rows",
            row_kind="degree",
        )
    if call.procedure == "graphistry.degree.write":
        return _ProcedureDefinition(
            procedure=call.procedure,
            backend="degree",
            result_kind="graph",
            row_kind="degree",
        )

    parts = call.procedure.split(".")
    if len(parts) < 3 or parts[0] != "graphistry":
        raise _unsupported_call(
            "Unsupported Cypher CALL procedure in the local compiler",
            call=call,
            value=call.procedure,
        )

    is_write = parts[-1] == "write"
    backend_name = parts[1]
    if backend_name not in {"igraph", "cugraph", "nx"}:
        raise _unsupported_call(
            "Unsupported Cypher CALL procedure in the local compiler",
            call=call,
            value=call.procedure,
        )

    algorithm_parts = parts[2:-1] if is_write else parts[2:]
    if len(algorithm_parts) != 1:
        raise _unsupported_call(
            "Unsupported Graphistry procedure naming shape in the local compiler",
            call=call,
            value=call.procedure,
        )
    algorithm = algorithm_parts[0]

    if backend_name == "igraph":
        if algorithm not in _IGRAPH_COMPUTE_ALGS:
            raise _unsupported_call(
                "Unsupported graphistry.igraph.* procedure in the local compiler",
                call=call,
                value=call.procedure,
            )
        return _ProcedureDefinition(
            procedure=call.procedure,
            backend="igraph",
            algorithm=algorithm,
            call_function="compute_igraph",
            result_kind="graph" if is_write else "rows",
            row_kind="node_or_graph",
        )

    if backend_name == "nx":
        if algorithm in _NETWORKX_NODE_ALGORITHMS:
            nx_row_kind: _ROW_KIND = "node"
        elif algorithm in _NETWORKX_EDGE_ALGORITHMS:
            nx_row_kind = "edge"
        elif algorithm in _NETWORKX_GRAPH_ALGORITHMS:
            nx_row_kind = "graph_only"
        else:
            raise _unsupported_call(
                "Only the limited graphistry.nx pagerank / betweenness / edge_betweenness / k_core local Cypher CALL subset is currently supported",
                call=call,
                value=call.procedure,
            )
        return _ProcedureDefinition(
            procedure=call.procedure,
            backend="networkx",
            algorithm=algorithm,
            call_function="networkx",
            result_kind="graph" if is_write else "rows",
            row_kind=nx_row_kind,
        )

    assert backend_name == "cugraph"
    if algorithm not in _CUGRAPH_COMPUTE_ALGS:
        raise _unsupported_call(
            "Unsupported graphistry.cugraph.* procedure in the local compiler",
            call=call,
            value=call.procedure,
        )
    if algorithm in node_compute_algs_to_attr:
        cugraph_row_kind: _ROW_KIND = "node"
    elif algorithm in edge_compute_algs_to_attr:
        cugraph_row_kind = "edge"
    else:
        cugraph_row_kind = "graph_only"
    return _ProcedureDefinition(
        procedure=call.procedure,
        backend="cugraph",
        algorithm=algorithm,
        call_function="compute_cugraph",
        result_kind="graph" if is_write else "rows",
        row_kind=cugraph_row_kind,
    )


def _eval_constant_call_arg(
    node: ExprNode,
    *,
    call: CallClause,
    raw_text: str,
) -> Any:
    if isinstance(node, ExprLiteral):
        return node.value
    if isinstance(node, UnaryOp):
        value = _eval_constant_call_arg(node.operand, call=call, raw_text=raw_text)
        if node.op == "+":
            return +value
        if node.op == "-":
            return -value
        if node.op == "not":
            return not value
        raise _invalid_call_argument(
            "Unsupported unary function in Graphistry Cypher CALL arguments",
            call=call,
            value=raw_text,
        )
    if isinstance(node, BinaryOp):
        left = _eval_constant_call_arg(node.left, call=call, raw_text=raw_text)
        right = _eval_constant_call_arg(node.right, call=call, raw_text=raw_text)
        if node.op == "+":
            return left + right
        if node.op == "-":
            return left - right
        if node.op == "*":
            return left * right
        if node.op == "/":
            return left / right
        if node.op == "%":
            return left % right
        raise _invalid_call_argument(
            "Unsupported arithmetic in Graphistry Cypher CALL arguments",
            call=call,
            value=raw_text,
        )
    if isinstance(node, ListLiteral):
        return [_eval_constant_call_arg(item, call=call, raw_text=raw_text) for item in node.items]
    if isinstance(node, MapLiteral):
        return {
            key: _eval_constant_call_arg(value, call=call, raw_text=raw_text)
            for key, value in node.items
        }
    if isinstance(node, FunctionCall):
        args = tuple(_eval_constant_call_arg(arg, call=call, raw_text=raw_text) for arg in node.args)
        if node.name == "abs" and len(args) == 1:
            return abs(args[0])
        if node.name == "tointeger" and len(args) == 1:
            return int(args[0])
        if node.name == "tofloat" and len(args) == 1:
            return float(args[0])
        raise _invalid_call_argument(
            "Unsupported function in Graphistry Cypher CALL arguments",
            call=call,
            value=raw_text,
        )
    raise _invalid_call_argument(
        "Graphistry Cypher CALL arguments must be literal map/list/scalar values or parameters",
        call=call,
        value=raw_text,
    )


def _parse_call_options(
    definition: _ProcedureDefinition,
    call: CallClause,
    *,
    params: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if definition.backend == "degree":
        if call.args:
            raise _invalid_call_argument(
                "graphistry.degree procedures do not accept arguments in the local compiler",
                call=call,
                value=call.procedure,
            )
        return {}

    if len(call.args) > 1:
        raise _invalid_call_argument(
            "Graphistry Cypher CALL procedures accept at most one options map in the local compiler",
            call=call,
            value=call.procedure,
        )
    if not call.args:
        return {}

    arg = call.args[0]
    from graphistry.compute.gfql.cypher.lowering import _parse_row_expr

    node = _parse_row_expr(
        arg.text,
        params=params,
        field="call.args",
        line=arg.span.line,
        column=arg.span.column,
    )
    options = _eval_constant_call_arg(node, call=call, raw_text=arg.text)
    if not isinstance(options, dict):
        raise _invalid_call_argument(
            "Graphistry Cypher CALL expects a single options map when arguments are present",
            call=call,
            value=arg.text,
        )
    return dict(options)


def _normalized_value_columns(
    definition: _ProcedureDefinition,
    call_params: Mapping[str, Any],
) -> Tuple[str, ...]:
    if definition.backend == "degree":
        return _DEGREE_OUTPUTS[1:]

    if definition.backend == "igraph":
        assert definition.algorithm is not None
        return (cast(str, call_params.get("out_col", definition.algorithm)),)

    if definition.backend == "networkx":
        assert definition.algorithm is not None
        if definition.algorithm in _NETWORKX_NODE_ALGORITHMS:
            value_cols = list(_NETWORKX_NODE_ALGORITHMS[definition.algorithm])
        elif definition.algorithm in _NETWORKX_EDGE_ALGORITHMS:
            value_cols = list(_NETWORKX_EDGE_ALGORITHMS[definition.algorithm])
        else:
            return ()
        if call_params.get("out_col") is not None:
            if len(value_cols) > 1:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Graphistry Cypher CALL does not allow out_col for multi-column procedures",
                    field="call.args.out_col",
                    value=call_params["out_col"],
                    suggestion="Remove out_col or choose a procedure with a single value column.",
                    language="cypher",
                )
            value_cols[0] = cast(str, call_params["out_col"])
        return tuple(value_cols)

    assert definition.backend == "cugraph"
    assert definition.algorithm is not None

    if definition.algorithm in node_compute_algs_to_attr:
        raw_cols = node_compute_algs_to_attr[definition.algorithm]
    elif definition.algorithm in edge_compute_algs_to_attr:
        raw_cols = edge_compute_algs_to_attr[definition.algorithm]
    else:
        return ()

    value_cols = list(raw_cols) if isinstance(raw_cols, list) else [raw_cols]
    if call_params.get("out_col") is not None:
        if len(value_cols) > 1:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Graphistry Cypher CALL does not allow out_col for multi-column procedures",
                field="call.args.out_col",
                value=call_params["out_col"],
                suggestion="Remove out_col or choose a procedure with a single value column.",
                language="cypher",
            )
        value_cols[0] = cast(str, call_params["out_col"])
    elif definition.algorithm in node_compute_algs_to_attr and len(value_cols) == 1:
        value_cols[0] = definition.algorithm
    return tuple(value_cols)


def _default_output_names(
    definition: _ProcedureDefinition,
    call_params: Mapping[str, Any],
) -> Tuple[str, ...]:
    if definition.backend == "degree":
        return _DEGREE_OUTPUTS

    value_cols = _normalized_value_columns(definition, call_params)
    if definition.row_kind in {"node", "node_or_graph"}:
        return ("nodeId",) + value_cols
    if definition.row_kind == "edge":
        return ("source", "destination") + value_cols
    return ()


def _normalize_call_params(
    definition: _ProcedureDefinition,
    call: CallClause,
    *,
    params: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if definition.backend == "degree":
        return {}

    raw_options = _parse_call_options(definition, call, params=params)
    if definition.backend == "igraph":
        reserved_keys = _IGRAPH_RESERVED_KEYS
    elif definition.backend == "networkx":
        reserved_keys = _NETWORKX_RESERVED_KEYS
    else:
        reserved_keys = _CUGRAPH_RESERVED_KEYS

    top_level_params = {k: v for k, v in raw_options.items() if k in reserved_keys}
    nested_params = top_level_params.pop("params", None)
    if nested_params is None:
        algorithm_params: Dict[str, Any] = {}
    elif isinstance(nested_params, dict):
        algorithm_params = dict(nested_params)
    else:
        raise _invalid_call_argument(
            "Graphistry Cypher CALL expects params to be a dictionary when provided",
            call=call,
            value=nested_params,
            field="call.args.params",
        )

    extra_params = {k: v for k, v in raw_options.items() if k not in reserved_keys}
    duplicate_keys = set(algorithm_params).intersection(extra_params)
    if duplicate_keys:
        raise _invalid_call_argument(
            "Graphistry Cypher CALL options cannot repeat the same algorithm parameter in params and at top level",
            call=call,
            value=sorted(duplicate_keys),
        )
    if extra_params:
        algorithm_params.update(extra_params)

    call_params: Dict[str, Any] = dict(top_level_params)
    if algorithm_params:
        call_params["params"] = algorithm_params
    call_params["alg"] = definition.algorithm
    if definition.backend == "networkx":
        out_col = call_params.get("out_col")
        if out_col is not None and not isinstance(out_col, str):
            raise _invalid_call_argument(
                "graphistry.nx.* out_col must be a string when provided",
                call=call,
                value=out_col,
                field="call.args.out_col",
            )
        directed = call_params.get("directed")
        if directed is not None and not isinstance(directed, bool):
            raise _invalid_call_argument(
                "graphistry.nx.* directed must be a boolean when provided",
                call=call,
                value=directed,
                field="call.args.directed",
            )
        return call_params
    assert definition.call_function is not None
    try:
        return dict(validate_call_params(definition.call_function, call_params))
    except GFQLTypeError as exc:
        raise _invalid_call_argument(exc.message, call=call, value=raw_options) from exc


def compile_cypher_call(
    call: CallClause,
    *,
    params: Optional[Mapping[str, Any]] = None,
) -> CompiledCypherProcedureCall:
    definition = _resolve_procedure_definition(call)
    call_params = _normalize_call_params(definition, call, params=params)

    if definition.row_kind == "graph_only" and definition.result_kind != "graph":
        raise _unsupported_call(
            "Topology-returning Graphistry procedures require .write() in the local Cypher subset",
            call=call,
            value=call.procedure,
        )

    if definition.result_kind == "graph":
        if call.yield_items:
            raise _unsupported_call(
                "Graph-preserving Cypher CALL procedures do not support YIELD in the current local subset",
                call=call,
                value=call.procedure,
            )
        return CompiledCypherProcedureCall(
            procedure=definition.procedure,
            backend=definition.backend,
            algorithm=definition.algorithm,
            result_kind="graph",
            row_kind=definition.row_kind,
            call_function=definition.call_function,
            call_params=call_params,
            line=call.span.line,
            column=call.span.column,
        )

    available_outputs = _default_output_names(definition, call_params)
    if not call.yield_items:
        output_columns = tuple(
            ProcedureOutputColumn(source_name=name, output_name=name)
            for name in available_outputs
        )
        return CompiledCypherProcedureCall(
            procedure=definition.procedure,
            backend=definition.backend,
            algorithm=definition.algorithm,
            output_columns=output_columns,
            result_kind="rows",
            row_kind=definition.row_kind,
            call_function=definition.call_function,
            call_params=call_params,
            line=call.span.line,
            column=call.span.column,
        )

    known_outputs = set(available_outputs)
    seen_output_names: set[str] = set()
    compiled_outputs = []
    for yield_item in call.yield_items:
        if yield_item.name not in known_outputs:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher YIELD references an unknown procedure output in the local compiler",
                field="yield",
                value=yield_item.name,
                suggestion=f"Use one of: {', '.join(available_outputs)}.",
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
        backend=definition.backend,
        algorithm=definition.algorithm,
        output_columns=tuple(compiled_outputs),
        result_kind="rows",
        row_kind=definition.row_kind,
        call_function=definition.call_function,
        call_params=call_params,
        line=call.span.line,
        column=call.span.column,
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


def _as_pandas_frame(df: Any, columns: Tuple[str, ...]) -> pd.DataFrame:
    subset = df[list(columns)]
    if hasattr(subset, "to_pandas"):
        return cast(pd.DataFrame, subset.to_pandas())
    return pd.DataFrame(subset)


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


def _merge_edge_property_rows(
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


def _raise_missing_backend_dependency(
    compiled_call: CompiledCypherProcedureCall,
    *,
    dependency: str,
    suggestion: str,
    exc: Exception,
) -> None:
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
    if graph_nx.is_directed():
        degree_fn = graph_nx.out_degree
        predecessor_fn = graph_nx.predecessors
    else:
        degree_fn = graph_nx.degree
        predecessor_fn = graph_nx.neighbors
    dangling_nodes = [node for node in nodes if degree_fn(node) == 0]

    for _ in range(max_iter):
        dangling_mass = alpha * sum(scores[node] for node in dangling_nodes) / node_count
        next_scores: Dict[Any, float] = {}
        delta = 0.0
        for node in nodes:
            inbound_mass = 0.0
            for predecessor in predecessor_fn(node):
                out_degree = degree_fn(predecessor)
                if out_degree > 0:
                    inbound_mass += scores[predecessor] / out_degree
            next_score = ((1.0 - alpha) / node_count) + dangling_mass + (alpha * inbound_mass)
            next_scores[node] = next_score
            delta += abs(next_score - scores[node])
        scores = next_scores
        if delta <= tol * node_count:
            break
    return scores


def _networkx_graph(base_graph: Plottable, *, directed: bool) -> Tuple[Plottable, Any]:
    try:
        import networkx as nx
    except ImportError as exc:
        raise exc

    graph_with_nodes = _materialized_graph(base_graph)
    assert graph_with_nodes._nodes is not None
    assert graph_with_nodes._node is not None
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None

    nodes_pdf = _as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))
    edges_df = getattr(graph_with_nodes, "_edges", None)
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else _as_pandas_frame(edges_df, edge_columns)

    graph_nx = nx.DiGraph() if directed else nx.Graph()
    graph_nx.add_nodes_from(nodes_pdf[graph_with_nodes._node].tolist())
    if not edges_pdf.empty:
        graph_nx.add_edges_from(edges_pdf[list(edge_columns)].itertuples(index=False, name=None))
    return graph_with_nodes, graph_nx


def _networkx_common_inputs(compiled_call: CompiledCypherProcedureCall) -> Tuple[bool, Dict[str, Any]]:
    directed = cast(bool, compiled_call.call_params.get("directed", True))
    params = dict(cast(Mapping[str, Any], compiled_call.call_params.get("params", {})))
    return directed, params


def _networkx_node_rows(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> DataFrameT:
    try:
        import networkx as nx
    except ImportError as exc:
        _raise_missing_backend_dependency(
            compiled_call,
            dependency="networkx",
            suggestion="Install networkx or use graphistry.igraph.* / graphistry.cugraph.* procedures.",
            exc=exc,
        )

    directed, params = _networkx_common_inputs(compiled_call)
    graph_with_nodes, graph_nx = _networkx_graph(base_graph, directed=directed)
    assert graph_with_nodes._node is not None
    nodes_pdf = _as_pandas_frame(graph_with_nodes._nodes, (graph_with_nodes._node,))

    try:
        if compiled_call.algorithm == "pagerank":
            scores = _networkx_pagerank_scores(graph_nx, **params)
        else:
            scores = getattr(nx, cast(str, compiled_call.algorithm))(graph_nx, **params)
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

    out_col = cast(str, compiled_call.call_params.get("out_col", compiled_call.algorithm or "pagerank"))
    rows_pdf = nodes_pdf.rename(columns={graph_with_nodes._node: "nodeId"}).copy()
    rows_pdf[out_col] = rows_pdf["nodeId"].map(scores).fillna(0.0)
    return cast(DataFrameT, rows_pdf)


def _networkx_edge_rows(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> DataFrameT:
    try:
        import networkx as nx
    except ImportError as exc:
        _raise_missing_backend_dependency(
            compiled_call,
            dependency="networkx",
            suggestion="Install networkx or use graphistry.igraph.* / graphistry.cugraph.* procedures.",
            exc=exc,
        )

    directed, params = _networkx_common_inputs(compiled_call)
    graph_with_nodes, graph_nx = _networkx_graph(base_graph, directed=directed)
    assert graph_with_nodes._source is not None
    assert graph_with_nodes._destination is not None
    edge_columns = (graph_with_nodes._source, graph_with_nodes._destination)
    edges_df = getattr(graph_with_nodes, "_edges", None)
    edges_pdf = pd.DataFrame(columns=list(edge_columns)) if edges_df is None else _as_pandas_frame(edges_df, edge_columns)

    try:
        scores = getattr(nx, cast(str, compiled_call.algorithm))(graph_nx, **params)
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

    out_col = cast(str, compiled_call.call_params.get("out_col", compiled_call.algorithm or "edge_betweenness_centrality"))
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
    graph_with_nodes = _materialized_graph(base_graph)
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


def _execute_networkx_graph_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    try:
        import networkx as nx
    except ImportError as exc:
        _raise_missing_backend_dependency(
            compiled_call,
            dependency="networkx",
            suggestion="Install networkx or use graphistry.igraph.* / graphistry.cugraph.* procedures.",
            exc=exc,
        )

    if compiled_call.row_kind == "node":
        rows_df = _networkx_node_rows(base_graph, compiled_call)
        out_col = cast(str, compiled_call.call_params.get("out_col", compiled_call.algorithm or "pagerank"))
        return _merge_node_property_rows(
            base_graph,
            rows_df,
            source_value_col=out_col,
            output_value_col=out_col,
        )
    if compiled_call.row_kind == "edge":
        rows_df = _networkx_edge_rows(base_graph, compiled_call)
        out_col = cast(str, compiled_call.call_params.get("out_col", compiled_call.algorithm or "edge_betweenness_centrality"))
        return _merge_edge_property_rows(
            base_graph,
            rows_df,
            source_value_col=out_col,
            output_value_col=out_col,
        )

    directed, params = _networkx_common_inputs(compiled_call)
    _, graph_nx = _networkx_graph(base_graph, directed=directed)
    try:
        projected_graph = getattr(nx, cast(str, compiled_call.algorithm))(graph_nx, **params)
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
    return _merge_networkx_projected_graph(base_graph, projected_graph)


def _execute_backend_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.backend == "degree":
        return _materialized_graph(base_graph).get_degrees()

    if compiled_call.call_function == "networkx":
        return _execute_networkx_graph_call(base_graph, compiled_call)

    if compiled_call.call_function == "compute_igraph":
        try:
            return _materialized_graph(base_graph).compute_igraph(**dict(compiled_call.call_params))
        except ImportError as exc:
            _raise_missing_backend_dependency(
                compiled_call,
                dependency="python-igraph",
                suggestion="Install python-igraph or use graphistry.cugraph.* procedures.",
                exc=exc,
            )

    if compiled_call.call_function == "compute_cugraph":
        try:
            return base_graph.compute_cugraph(**dict(compiled_call.call_params))
        except ImportError as exc:
            _raise_missing_backend_dependency(
                compiled_call,
                dependency="cugraph",
                suggestion="Install RAPIDS cuGraph/cuDF or use graphistry.igraph.* procedures.",
                exc=exc,
            )

    raise ValueError(f"Unexpected Graphistry Cypher CALL function: {compiled_call.call_function}")


def _node_rows(
    computed_graph: Plottable,
    *,
    value_columns: Sequence[str],
) -> DataFrameT:
    if computed_graph._nodes is None or computed_graph._node is None:
        raise ValueError("Expected computed graph to have node rows")
    return cast(
        DataFrameT,
        computed_graph._nodes.rename(columns={computed_graph._node: "nodeId"})[
            ["nodeId", *value_columns]
        ],
    )


def _edge_rows(
    computed_graph: Plottable,
    *,
    value_columns: Sequence[str],
) -> DataFrameT:
    if (
        computed_graph._edges is None
        or computed_graph._source is None
        or computed_graph._destination is None
    ):
        raise ValueError("Expected computed graph to have edge rows")
    return cast(
        DataFrameT,
        computed_graph._edges.rename(
            columns={
                computed_graph._source: "source",
                computed_graph._destination: "destination",
            }
        )[["source", "destination", *value_columns]],
    )


def _graph_only_row_error(compiled_call: CompiledCypherProcedureCall) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        "Topology-returning Graphistry procedures require .write() in the local Cypher subset",
        field="call",
        value=compiled_call.procedure,
        suggestion="Use CALL ... .write() when the procedure returns a graph topology.",
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    )


def _write_only_igraph_row_error(compiled_call: CompiledCypherProcedureCall) -> GFQLValidationError:
    return GFQLValidationError(
        ErrorCode.E108,
        "This graphistry.igraph.* procedure returns a graph topology rather than row outputs",
        field="call",
        value=compiled_call.procedure,
        suggestion="Use CALL ... .write() for topology-returning igraph procedures such as spanning_tree or gomory_hu_tree.",
        line=compiled_call.line,
        column=compiled_call.column,
        language="cypher",
    )


def _execute_graph_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    return _execute_backend_call(base_graph, compiled_call)


def execute_cypher_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.result_kind == "graph":
        return _execute_graph_call(base_graph, compiled_call)

    if compiled_call.backend == "degree":
        default_rows = _degree_rows(base_graph)
    elif compiled_call.backend == "networkx":
        if compiled_call.row_kind == "graph_only":
            raise _graph_only_row_error(compiled_call)
        if compiled_call.row_kind == "edge":
            default_rows = _networkx_edge_rows(base_graph, compiled_call)
        else:
            default_rows = _networkx_node_rows(base_graph, compiled_call)
    else:
        computed = _execute_backend_call(base_graph, compiled_call)
        value_columns = _normalized_value_columns(
            _ProcedureDefinition(
                procedure=compiled_call.procedure,
                backend=compiled_call.backend,
                algorithm=compiled_call.algorithm,
                call_function=compiled_call.call_function,
                result_kind=compiled_call.result_kind,
                row_kind=compiled_call.row_kind,
            ),
            compiled_call.call_params,
        )
        if compiled_call.row_kind == "graph_only":
            raise _graph_only_row_error(compiled_call)
        if compiled_call.row_kind == "edge":
            default_rows = _edge_rows(computed, value_columns=value_columns)
        else:
            missing_columns = [
                col_name
                for col_name in value_columns
                if computed._nodes is None or col_name not in computed._nodes.columns
            ]
            if missing_columns:
                raise _write_only_igraph_row_error(compiled_call)
            default_rows = _node_rows(computed, value_columns=value_columns)

    return _graph_row_result(base_graph, _project_outputs(default_rows, compiled_call.output_columns))
