from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, cast

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
from graphistry.compute.gfql.cypher.procedures.common import (
    materialized_graph as _materialized_graph,
    raise_missing_backend_dependency as _raise_missing_backend_dependency,
)
from graphistry.compute.gfql.cypher.procedures.networkx import (
    NETWORKX_PROCEDURES as _NETWORKX_PROCEDURES,
    NETWORKX_RESERVED_KEYS as _NETWORKX_RESERVED_KEYS,
    execute_networkx_graph_call as _execute_networkx_graph_call,
    networkx_edge_rows as _networkx_edge_rows,
    networkx_node_rows as _networkx_node_rows,
    networkx_source_value_columns,
)
from graphistry.compute.gfql.schema_effects import (
    apply_graph_schema_effect,
    schema_effect_for_procedure_output,
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
        suggestion="Use graphistry.degree, supported graphistry.igraph.* / graphistry.cugraph.* procedures, or supported graphistry.nx procedures.",
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
        nx_spec = _NETWORKX_PROCEDURES.get(algorithm)
        if nx_spec is None:
            raise _unsupported_call(
                "Unsupported graphistry.nx.* procedure in the local compiler",
                call=call,
                value=call.procedure,
            )
        return _ProcedureDefinition(
            procedure=call.procedure,
            backend="networkx",
            algorithm=algorithm,
            call_function="networkx",
            result_kind="graph" if is_write else "rows",
            row_kind=nx_spec[0],
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
    value_cols = list(_source_value_columns(definition))
    if definition.backend == "cugraph" and value_cols:
        assert definition.algorithm is not None
        value_cols[0] = definition.algorithm
    if value_cols and call_params.get("out_col") is not None:
        if len(value_cols) > 1:
            _raise_multi_column_out_col(call_params["out_col"])
        value_cols[0] = cast(str, call_params["out_col"])
    return tuple(value_cols)


def _raise_multi_column_out_col(value: Any) -> None:
    raise GFQLValidationError(
        ErrorCode.E108,
        "Graphistry Cypher CALL does not allow out_col for multi-column procedures",
        field="call.args.out_col",
        value=value,
        suggestion="Remove out_col or choose a procedure with a single value column.",
        language="cypher",
    )


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


def _source_value_columns(definition: _ProcedureDefinition) -> Tuple[str, ...]:
    if definition.backend == "degree":
        return _DEGREE_OUTPUTS[1:]

    if definition.backend == "igraph":
        assert definition.algorithm is not None
        return (definition.algorithm,)

    if definition.backend == "networkx":
        return networkx_source_value_columns(definition.algorithm)

    assert definition.backend == "cugraph"
    assert definition.algorithm is not None

    if definition.algorithm in node_compute_algs_to_attr:
        raw_cols = node_compute_algs_to_attr[definition.algorithm]
    elif definition.algorithm in edge_compute_algs_to_attr:
        raw_cols = edge_compute_algs_to_attr[definition.algorithm]
    else:
        return ()
    return tuple(raw_cols) if isinstance(raw_cols, list) else (raw_cols,)


def _definition_from_compiled_call(compiled_call: CompiledCypherProcedureCall) -> _ProcedureDefinition:
    return _ProcedureDefinition(
        procedure=compiled_call.procedure,
        backend=compiled_call.backend,
        algorithm=compiled_call.algorithm,
        call_function=compiled_call.call_function,
        result_kind=compiled_call.result_kind,
        row_kind=compiled_call.row_kind,
    )


def _output_rename_map(frame: Any, source_columns: Tuple[str, ...], output_columns: Tuple[str, ...]) -> Dict[str, str]:
    return {
        source_name: output_name
        for source_name, output_name in zip(source_columns, output_columns)
        if source_name != output_name and source_name in frame.columns and output_name not in frame.columns
    }


def _align_computed_result_columns(
    computed_graph: Plottable,
    compiled_call: CompiledCypherProcedureCall,
) -> Plottable:
    # Keep the local Cypher CALL surface stable even when backend column names differ.
    definition = _definition_from_compiled_call(compiled_call)
    source_columns = _source_value_columns(definition)
    output_columns = _normalized_value_columns(definition, compiled_call.call_params)

    if len(source_columns) != len(output_columns):
        return computed_graph

    if compiled_call.row_kind == "edge":
        if (
            computed_graph._edges is None
            or computed_graph._source is None
            or computed_graph._destination is None
        ):
            return computed_graph
        rename_map = _output_rename_map(computed_graph._edges, source_columns, output_columns)
        if not rename_map:
            return computed_graph
        return cast(
            Plottable,
            computed_graph.edges(
                computed_graph._edges.rename(columns=rename_map),
                computed_graph._source,
                computed_graph._destination,
                computed_graph._edge,
            ),
        )

    if compiled_call.row_kind in {"node", "node_or_graph"}:
        if computed_graph._nodes is None or computed_graph._node is None:
            return computed_graph
        rename_map = _output_rename_map(computed_graph._nodes, source_columns, output_columns)
        if not rename_map:
            return computed_graph
        return cast(
            Plottable,
            computed_graph.nodes(
                computed_graph._nodes.rename(columns=rename_map),
                computed_graph._node,
            ),
        )

    return computed_graph


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


def _compiled_procedure_call(
    definition: _ProcedureDefinition,
    call: CallClause,
    *,
    call_params: Mapping[str, Any],
    output_columns: Tuple[ProcedureOutputColumn, ...] = (),
) -> CompiledCypherProcedureCall:
    return CompiledCypherProcedureCall(
        procedure=definition.procedure,
        backend=definition.backend,
        algorithm=definition.algorithm,
        output_columns=output_columns,
        result_kind=definition.result_kind,
        row_kind=definition.row_kind,
        call_function=definition.call_function,
        call_params=call_params,
        line=call.span.line,
        column=call.span.column,
    )


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
        return _compiled_procedure_call(definition, call, call_params=call_params)

    available_outputs = _default_output_names(definition, call_params)
    if not call.yield_items:
        output_columns = tuple(
            ProcedureOutputColumn(source_name=name, output_name=name)
            for name in available_outputs
        )
        return _compiled_procedure_call(
            definition,
            call,
            call_params=call_params,
            output_columns=output_columns,
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
    return _compiled_procedure_call(
        definition,
        call,
        call_params=call_params,
        output_columns=tuple(compiled_outputs),
    )


def _graph_row_result(base_graph: Plottable, rows_df: DataFrameT) -> Plottable:
    out = base_graph.bind()
    out._nodes = rows_df
    edges_df = base_graph._edges
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
            computed = base_graph.compute_cugraph(**dict(compiled_call.call_params))
            return _align_computed_result_columns(computed, compiled_call)
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


def execute_cypher_call(base_graph: Plottable, compiled_call: CompiledCypherProcedureCall) -> Plottable:
    if compiled_call.result_kind == "graph":
        result_graph = _execute_backend_call(base_graph, compiled_call)
        definition = _definition_from_compiled_call(compiled_call)
        effect = schema_effect_for_procedure_output(
            backend=compiled_call.backend,
            algorithm=compiled_call.algorithm,
            row_kind=compiled_call.row_kind,
            value_columns=_normalized_value_columns(definition, compiled_call.call_params),
        )
        return apply_graph_schema_effect(base_graph, result_graph, effect)

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
            _definition_from_compiled_call(compiled_call),
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
