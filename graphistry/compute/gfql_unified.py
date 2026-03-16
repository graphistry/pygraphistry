"""GFQL unified entrypoint for chains, DAGs, and local string-compiled queries."""
# ruff: noqa: E501

import re
from typing import Any, Dict, List, Literal, Mapping, Optional, Sequence, Union, cast
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract, df_concat, df_cons, resolve_engine
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTNode, ASTEdge, ASTCall
from .chain import Chain, chain as chain_impl
from .chain_let import chain_let as chain_let_impl
from .execution_context import ExecutionContext
from .gfql.policy import (
    PolicyContext,
    PolicyException,
    PolicyFunction,
    PolicyDict,
    QueryType,
    expand_policy
)
from graphistry.compute.gfql.same_path_types import (
    WhereComparison,
    normalize_where_entries,
    parse_where_json,
)
from graphistry.compute.exceptions import ErrorCode, GFQLValidationError
from graphistry.compute.gfql.cypher.api import compile_cypher
from graphistry.compute.gfql.cypher.lowering import CompiledCypherGraphQuery, CompiledCypherQuery, CompiledCypherUnionQuery
from graphistry.compute.gfql.cypher.call_procedures import execute_cypher_call
from graphistry.compute.gfql.cypher.result_postprocess import WholeRowProjectionMeta, apply_result_projection
from graphistry.compute.gfql.df_executor import (
    DFSamePathExecutor,
    build_same_path_inputs,
    execute_same_path_chain,
)
from graphistry.compute.gfql.row.pipeline import is_row_pipeline_call
from graphistry.compute.typing import DataFrameT, SeriesT
from graphistry.compute.util.generate_safe_column_name import generate_safe_column_name
from graphistry.compute.validate.validate_schema import validate_chain_schema
from graphistry.otel import otel_traced, otel_detail_enabled

logger = setup_logger(__name__)

_CYPHER_LEAD_RE = re.compile(
    r"^\s*(?:MATCH|OPTIONAL\s+MATCH|WITH|RETURN|UNWIND|CALL|CREATE|MERGE|DELETE|DETACH\s+DELETE|SET|REMOVE|FOREACH|GRAPH|USE)\b",
    re.IGNORECASE,
)


def _looks_like_cypher_query(query: str) -> bool:
    return _CYPHER_LEAD_RE.match(query) is not None


def _series_to_pylist(values: Any) -> List[Any]:
    if hasattr(values, "to_arrow"):
        try:
            return list(values.to_arrow().to_pylist())
        except Exception:
            pass
    if hasattr(values, "to_pandas"):
        try:
            return list(values.to_pandas().tolist())
        except Exception:
            pass
    if hasattr(values, "tolist"):
        try:
            return list(values.tolist())
        except Exception:
            pass
    return list(values)


def _apply_empty_result_row(
    result: Plottable,
    *,
    engine: Union[EngineAbstract, str],
    empty_result_row: Mapping[str, Any],
) -> Plottable:
    rows_df = getattr(result, "_nodes", None)
    if rows_df is not None and len(rows_df) > 0:
        return result
    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    out = result.bind()
    out._nodes = df_ctor({key: [value] for key, value in empty_result_row.items()})
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out


def _apply_optional_null_fill(
    result: Plottable,
    *,
    base_result: Plottable,
    alignment_result: Plottable,
    alignment_output_name: str,
    engine: Union[EngineAbstract, str],
    null_row: Mapping[str, Any],
) -> Plottable:
    base_rows_df = getattr(base_result, "_nodes", None)
    expected_rows = 0 if base_rows_df is None else len(base_rows_df)
    if expected_rows == 0:
        return result

    rows_df = getattr(result, "_nodes", None)
    actual_rows = 0 if rows_df is None else len(rows_df)
    entity_meta = cast(
        Optional[Dict[str, WholeRowProjectionMeta]],
        getattr(alignment_result, "_cypher_entity_projection_meta", None),
    )
    if not isinstance(entity_meta, dict) or alignment_output_name not in entity_meta:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not recover matched seed identities",
            field="match",
            value=alignment_output_name,
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )
    matched_ids = entity_meta[alignment_output_name]["ids"]
    if not hasattr(matched_ids, "tolist"):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not recover matched seed identities",
            field="match",
            value=alignment_output_name,
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )
    if actual_rows != len(matched_ids):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment produced inconsistent row counts",
            field="match",
            value={"matched_rows": actual_rows, "aligned_ids": len(matched_ids)},
            suggestion="Retry with a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )
    node_col = base_result._node
    if node_col is None or base_rows_df is None or node_col not in base_rows_df.columns:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not recover base seed identities",
            field="match",
            value=node_col,
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )

    base_ids = _series_to_pylist(base_rows_df[node_col])
    matched_id_list = _series_to_pylist(matched_ids)
    if len(base_ids) == actual_rows and base_ids == matched_id_list:
        return result

    concrete_engine = resolve_engine(cast(Any, engine), result)
    df_ctor = df_cons(concrete_engine)
    concat = df_concat(concrete_engine)
    fill_df = df_ctor({key: [value] for key, value in null_row.items()})
    segments = []
    matched_idx = 0
    for base_id in base_ids:
        group_start = matched_idx
        while matched_idx < len(matched_id_list) and matched_id_list[matched_idx] == base_id:
            matched_idx += 1
        if matched_idx > group_start:
            if rows_df is None:
                raise GFQLValidationError(
                    ErrorCode.E108,
                    "Cypher OPTIONAL MATCH null-row alignment lost the projected result rows",
                    field="match",
                    value=None,
                    suggestion="Retry with a simpler OPTIONAL MATCH projection shape in the local compiler.",
                    language="cypher",
                )
            segments.append(rows_df.iloc[group_start:matched_idx])
        else:
            segments.append(fill_df)
    if matched_idx != len(matched_id_list):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher OPTIONAL MATCH null-row alignment could not map matched rows back to the seed MATCH order",
            field="match",
            value={"mapped_rows": matched_idx, "matched_rows": len(matched_id_list)},
            suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
            language="cypher",
        )

    out = result.bind()
    out._nodes = concat(segments, ignore_index=True, sort=False) if segments else df_ctor()
    edges_df = getattr(result, "_edges", None)
    if edges_df is not None:
        out._edges = edges_df[:0]
    return out


def _apply_optional_projection_row_guard(
    result: Plottable,
    *,
    expected_rows: int,
) -> Plottable:
    if expected_rows == 0:
        return result

    rows_df = getattr(result, "_nodes", None)
    actual_rows = 0 if rows_df is None else len(rows_df)
    if actual_rows >= expected_rows:
        return result

    raise GFQLValidationError(
        ErrorCode.E108,
        "Cypher MATCH ... OPTIONAL MATCH projections over optional aliases would need null-extension rows that the local compiler cannot synthesize for this query shape",
        field="match",
        value={"expected_rows": expected_rows, "actual_rows": actual_rows},
        suggestion="Use a simpler OPTIONAL MATCH projection shape in the local compiler.",
        language="cypher",
    )


def _execute_graph_constructor_compiled(
    base_graph: Plottable,
    chain: Chain,
    *,
    procedure_call: Any = None,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a compiled graph constructor (MATCH-based or CALL-based)."""
    if procedure_call is not None:
        return execute_cypher_call(base_graph, procedure_call)
    return _chain_dispatch(base_graph, chain, engine, policy, context)


def _resolve_graph_bindings(
    base_graph: Plottable,
    bindings: tuple,
    scope: Optional[Dict[str, Plottable]] = None,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Dict[str, Plottable]:
    """Execute graph bindings in order, building a scope of named graphs.

    Each binding's USE clause (if present) is resolved against previously
    bound graphs in the scope. The resolved graph becomes the base for
    that binding's execution.
    """
    if scope is None:
        scope = {}
    for binding in bindings:
        target_graph = base_graph
        # USE ref inside the binding's constructor was already validated at
        # parse time. At runtime, resolve it against the scope.
        if binding.use_ref is not None:
            target_graph = scope.get(binding.use_ref.lower(), base_graph)
        result = _execute_graph_constructor_compiled(
            target_graph, binding.chain,
            procedure_call=binding.procedure_call,
            engine=engine, policy=policy, context=context,
        )
        scope[binding.name.lower()] = result
    return scope


def _execute_graph_query(
    base_graph: Plottable,
    compiled: CompiledCypherGraphQuery,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a standalone GRAPH { ... } query (returns graph state)."""
    scope = _resolve_graph_bindings(
        base_graph, compiled.graph_bindings,
        engine=engine, policy=policy, context=context,
    )
    # Resolve USE for the final constructor
    target_graph = base_graph
    if compiled.use_ref is not None:
        target_graph = scope.get(compiled.use_ref.lower(), base_graph)
    return _execute_graph_constructor_compiled(
        target_graph, compiled.chain,
        procedure_call=compiled.procedure_call,
        engine=engine, policy=policy, context=context,
    )


def _execute_query_with_graph_context(
    base_graph: Plottable,
    compiled: CompiledCypherQuery,
    *,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
) -> Plottable:
    """Execute a query that has GRAPH bindings and/or USE."""
    scope = _resolve_graph_bindings(
        base_graph, compiled.graph_bindings,
        engine=engine, policy=policy, context=context,
    )
    # If USE is specified, execute the main query against the USE'd graph
    if compiled.use_ref is not None:
        target_graph = scope.get(compiled.use_ref.lower(), base_graph)
    else:
        target_graph = base_graph
    # Strip graph context from the compiled query and execute normally
    plain_query = CompiledCypherQuery(
        chain=compiled.chain,
        seed_rows=compiled.seed_rows,
        procedure_call=compiled.procedure_call,
        result_projection=compiled.result_projection,
        empty_result_row=compiled.empty_result_row,
        optional_null_fill=compiled.optional_null_fill,
        optional_projection_row_guard=compiled.optional_projection_row_guard,
        start_nodes_query=compiled.start_nodes_query,
        start_nodes_output_name=compiled.start_nodes_output_name,
    )
    return _execute_compiled_query(
        target_graph,
        compiled_query=plain_query,
        engine=engine,
        policy=policy,
        context=context,
    )


def _execute_compiled_query(
    base_graph: Plottable,
    *,
    compiled_query: Union[CompiledCypherQuery, CompiledCypherUnionQuery],
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    if isinstance(compiled_query, CompiledCypherUnionQuery):
        concrete_engine = resolve_engine(cast(Any, engine), base_graph)
        df_ctor = df_cons(concrete_engine)
        concat = df_concat(concrete_engine)
        branch_results = [
            _execute_compiled_query(
                base_graph,
                compiled_query=branch,
                engine=engine,
                policy=policy,
                context=context,
                start_nodes=start_nodes,
            )
            for branch in compiled_query.branches
        ]
        row_frames = [cast(DataFrameT, getattr(result, "_nodes", None)) for result in branch_results if getattr(result, "_nodes", None) is not None]
        union_rows = df_ctor() if not row_frames else concat(row_frames, ignore_index=True, sort=False)
        if compiled_query.union_kind == "distinct" and len(union_rows) > 0:
            union_rows = cast(DataFrameT, union_rows.drop_duplicates(ignore_index=True))
        out = base_graph.bind()
        out._nodes = union_rows
        out._edges = df_ctor()
        return out

    dispatch_graph = base_graph
    if compiled_query.procedure_call is not None:
        dispatch_graph = execute_cypher_call(base_graph, compiled_query.procedure_call)
    elif compiled_query.seed_rows:
        concrete_engine = resolve_engine(cast(Any, engine), base_graph)
        df_ctor = df_cons(concrete_engine)
        dispatch_graph = base_graph.bind()
        dispatch_graph._nodes = df_ctor({"__cypher_seed_row__": [True]})
        dispatch_graph._edges = df_ctor()
    result = _chain_dispatch(dispatch_graph, compiled_query.chain, engine, policy, context, start_nodes=start_nodes)
    if compiled_query.empty_result_row is not None:
        result = _apply_empty_result_row(
            result,
            engine=engine,
            empty_result_row=compiled_query.empty_result_row,
        )
    if compiled_query.result_projection is not None:
        result = apply_result_projection(result, compiled_query.result_projection)
    if compiled_query.optional_projection_row_guard is not None:
        expected_rows = 1
        for base_chain in compiled_query.optional_projection_row_guard.base_chains:
            base_result = _chain_dispatch(
                base_graph,
                base_chain,
                engine,
                policy,
                context,
            )
            base_rows_df = getattr(base_result, "_nodes", None)
            expected_rows *= 0 if base_rows_df is None else len(base_rows_df)
            if expected_rows == 0:
                break
        result = _apply_optional_projection_row_guard(
            result,
            expected_rows=expected_rows,
        )
    if compiled_query.optional_null_fill is not None:
        base_result = _chain_dispatch(
            base_graph,
            compiled_query.optional_null_fill.base_chain,
            engine,
            policy,
            context,
        )
        alignment_result = apply_result_projection(
            _chain_dispatch(
                base_graph,
                compiled_query.optional_null_fill.alignment_chain,
                engine,
                policy,
                context,
            ),
            compiled_query.optional_null_fill.alignment_projection,
        )
        result = _apply_optional_null_fill(
            result,
            base_result=base_result,
            alignment_result=alignment_result,
            alignment_output_name=compiled_query.optional_null_fill.alignment_output_name,
            engine=engine,
            null_row=compiled_query.optional_null_fill.null_row,
        )
    return result


def _compiled_query_start_nodes(
    compiled_query: CompiledCypherQuery,
    prefix_result: Plottable,
    *,
    engine: Union[EngineAbstract, str],
) -> DataFrameT:
    output_name = compiled_query.start_nodes_output_name
    entity_meta = cast(
        Optional[Dict[str, WholeRowProjectionMeta]],
        getattr(prefix_result, "_cypher_entity_projection_meta", None),
    )
    if not isinstance(entity_meta, dict) or output_name not in entity_meta:
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher MATCH after WITH could not recover carried node identities from the prefix stage",
            field="with",
            value=output_name,
            suggestion="Carry a whole-row node alias through WITH before MATCH re-entry.",
            language="cypher",
        )
    meta = entity_meta[output_name]
    if meta["table"] != "nodes":
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher MATCH after WITH currently supports node re-entry only",
            field="with",
            value=output_name,
            suggestion="Carry a whole-row node alias through WITH before MATCH re-entry.",
            language="cypher",
        )
    ids = meta["ids"]
    id_column = meta["id_column"]
    if not hasattr(ids, "dropna"):
        raise GFQLValidationError(
            ErrorCode.E108,
            "Cypher MATCH after WITH could not recover carried node identities from the prefix stage",
            field="with",
            value=output_name,
            suggestion="Carry a whole-row node alias through WITH before MATCH re-entry.",
            language="cypher",
        )
    concrete_engine = resolve_engine(cast(Any, engine), prefix_result)
    df_ctor = df_cons(concrete_engine)
    carried_ids = cast(SeriesT, ids.dropna().reset_index(drop=True))
    return df_ctor({id_column: carried_ids})


def _materialize_split_alias_columns(
    result: Plottable,
    executor: DFSamePathExecutor,
) -> Plottable:
    if result._edges is not None and result._edge is None:
        edge_id_col = generate_safe_column_name("edge_index", result._edges, prefix="__gfql_", suffix="__")
        result._edges = result._edges.assign(**{edge_id_col: range(len(result._edges))})
        result._edge = edge_id_col

    node_updates: Dict[str, Any] = {}
    edge_updates: Dict[str, Any] = {}

    for alias, binding in executor.inputs.alias_bindings.items():
        frame = executor.alias_frames.get(alias)
        if frame is None:
            continue
        if binding.kind == "node":
            df = result._nodes
            result_id_col = result._node
            frame_id_col = executor._node_column
        else:
            df = result._edges
            result_id_col = result._edge
            frame_id_col = executor._edge_column
        if (
            df is None
            or result_id_col is None
            or frame_id_col is None
            or result_id_col not in df.columns
            or frame_id_col not in frame.columns
        ):
            continue
        mask = df[result_id_col].isin(frame[frame_id_col])
        if binding.kind == "node":
            node_updates[alias] = mask
        else:
            edge_updates[alias] = mask

    if node_updates and result._nodes is not None:
        result._nodes = result._nodes.assign(**node_updates)
    if edge_updates and result._edges is not None:
        result._edges = result._edges.assign(**edge_updates)
    return result


def _gfql_otel_attrs(
    self: Plottable,
    query: Union[ASTObject, List[ASTObject], ASTLet, Chain, dict, str],
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
    output: Optional[str] = None,
    policy: Optional[Dict[str, PolicyFunction]] = None,
    where: Optional[Sequence[WhereComparison]] = None,
    language: Optional[Literal["cypher", "gremlin"]] = None,
    params: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    if isinstance(query, dict):
        query_type = "chain" if "chain" in query else "dag"
    else:
        query_type = detect_query_type(query)
    attrs: Dict[str, Any] = {"gfql.query_type": query_type}
    if isinstance(query, Chain):
        attrs["gfql.chain_len"] = len(query.chain)
        attrs["gfql.has_where"] = bool(query.where)
    elif isinstance(query, list):
        attrs["gfql.chain_len"] = len(query)
        if where:
            attrs["gfql.has_where"] = True
    elif isinstance(query, ASTLet):
        attrs["gfql.binding_count"] = len(query.bindings)
    elif isinstance(query, dict):
        attrs["gfql.binding_count"] = len(query)
        if "chain" in query and isinstance(query["chain"], list):
            attrs["gfql.chain_len"] = len(query["chain"])
    if otel_detail_enabled():
        attrs["gfql.output"] = output is not None
        attrs["gfql.policy"] = policy is not None
        attrs["gfql.engine"] = str(engine)
        if isinstance(query, str):
            attrs["gfql.language"] = language or "cypher"
            attrs["gfql.has_params"] = params is not None
    return attrs


def detect_query_type(query: Any) -> QueryType:
    if isinstance(query, ASTLet):
        return "dag"
    elif isinstance(query, str):
        return "chain"
    elif isinstance(query, (list, Chain)):
        return "chain"
    else:
        return "single"


def _compile_string_query(
    query: str,
    *,
    language: Optional[Literal["cypher", "gremlin"]],
    params: Optional[Mapping[str, Any]],
) -> Any:
    query_language = language or "cypher"
    if query_language != "cypher":
        raise GFQLValidationError(
            ErrorCode.E108,
            f"Unsupported GFQL string language '{query_language}'",
            field="language",
            value=query_language,
            suggestion="Use language='cypher' for now; Gremlin string compilation is not implemented yet.",
            language="gfql",
        )
    return compile_cypher(query, params=params)


@otel_traced("gfql.run", attrs_fn=_gfql_otel_attrs)
def gfql(self: Plottable,
         query: Union[ASTObject, List[ASTObject], ASTLet, Chain, dict, str],
         engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
         output: Optional[str] = None,
         policy: Optional[Dict[str, PolicyFunction]] = None,
         where: Optional[Sequence[WhereComparison]] = None,
         language: Optional[Literal["cypher", "gremlin"]] = None,
         params: Optional[Mapping[str, Any]] = None) -> Plottable:
    """
    Execute a GFQL query - either a chain or a DAG

    Unified entrypoint that automatically detects query type and
    dispatches to the appropriate execution engine.

    :param query: GFQL query - ASTObject, List[ASTObject], Chain, ASTLet, dict, or supported query string
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: For DAGs, name of binding to return (default: last executed)
    :param policy: Optional policy hooks for external control (preload, postload, precall, postcall phases)
    :param where: Optional same-path constraints for list/Chain queries
    :param language: Optional string-query language selector. Defaults to ``"cypher"`` when ``query`` is a string.
    :param params: Optional parameter dictionary for string-query compilation
    :returns: Resulting Plottable
    :rtype: Plottable

    **Policy Hooks**

    The policy parameter enables external control over GFQL query execution
    through hooks at three phases:

    - **preload**: Before data is loaded (can modify query/engine)
    - **postload**: After data is loaded (can inspect data size)
    - **precall**: Before each method call (can deny based on parameters)
    - **postcall**: After each method call (can validate results, timing)

    Policies can accept/deny/modify operations. Modifications are validated
    against a schema and applied immediately. Recursion is prevented at depth 1.

    **Policy Example**

    ::

        from graphistry.compute.gfql.policy import PolicyContext, PolicyException
        from typing import Optional

        def create_tier_policy(max_nodes: int = 10000):
            # State via closure
            state = {"nodes_processed": 0}

            def policy(context: PolicyContext) -> None:
                phase = context['phase']

                if phase == 'preload':
                    # Force CPU for free tier
                    return {'engine': 'cpu'}

                elif phase == 'postload':
                    # Check data size limits
                    stats = context.get('graph_stats', {})
                    nodes = stats.get('nodes', 0)
                    state['nodes_processed'] += nodes

                    if state['nodes_processed'] > max_nodes:
                        raise PolicyException(
                            phase='postload',
                            reason=f'Node limit {max_nodes} exceeded',
                            code=403,
                            data_size={'nodes': state['nodes_processed']}
                        )

                elif phase == 'precall':
                    # Restrict operations
                    op = context.get('call_op', '')
                    if op == 'hypergraph':
                        raise PolicyException(
                            phase='precall',
                            reason='Hypergraph not available in free tier',
                            code=403
                        )

                return None

            return policy

        # Use policy
        policy_func = create_tier_policy(max_nodes=1000)
        result = g.gfql([n()], policy={
            'preload': policy_func,
            'postload': policy_func,
            'precall': policy_func
        })

    **Example: Chain query**

    ::

        from graphistry.compute.ast import n, e

        # As list
        result = g.gfql([n({'type': 'person'}), e(), n()])

        # As Chain object
        from graphistry.compute.chain import Chain
        result = g.gfql(Chain([n({'type': 'person'}), e(), n()]))

        # As list with WHERE
        from graphistry.compute.gfql.same_path_types import col, compare
        result = g.gfql(
            [n(name="a"), e(), n(name="b")],
            where=[compare(col("a", "x"), "==", col("b", "y"))],
        )

    **Example: DAG query**

    ::

        from graphistry.compute.ast import let, ref, n, e

        result = g.gfql(let({
            'people': n({'type': 'person'}),
            'friends': ref('people', [e({'rel': 'knows'}), n()])
        }))

        # Select specific output
        friends = g.gfql(result, output='friends')

    **Example: Transformations (e.g., hypergraph)**

    ::

        from graphistry.compute import hypergraph

        # Simple transformation
        hg = g.gfql(hypergraph(entity_types=['user', 'product']))

        # Or using call()
        from graphistry.compute.ast import call
        hg = g.gfql(call('hypergraph', {'entity_types': ['user', 'product']}))

        # In a DAG with other operations
        result = g.gfql(let({
            'hg': hypergraph(entity_types=['user', 'product']),
            'filtered': ref('hg', [n({'type': 'user'})])
        }))

    **Example: Auto-detection**

    ::

        # List → chain execution
        g.gfql([n(), e(), n()])

        # Single ASTObject → chain execution
        g.gfql(n({'type': 'person'}))

        # Dict → DAG execution (convenience)
        g.gfql({'people': n({'type': 'person'})})

        # Local Cypher-string compilation → GFQL execution
        g.gfql(
            "MATCH (p:Person) RETURN p.name AS person_name ORDER BY person_name ASC LIMIT $top_n",
            params={"top_n": 10},
        )
    """
    context = ExecutionContext()

    if policy and context.policy_depth >= 1:
        logger.debug('Policy disabled due to recursion depth limit (depth=%d)', context.policy_depth)
        policy = None

    policy_depth = context.policy_depth
    if policy:
        context.policy_depth = policy_depth + 1

    expanded_policy: Optional[PolicyDict] = None
    if policy:
        expanded_policy = expand_policy(policy)

    try:
        where_param: Optional[List[WhereComparison]] = None
        if where is not None:
            if isinstance(where, (list, tuple)):
                where_param = normalize_where_entries(where)
            else:
                raise ValueError(f"where must be a list of comparisons, got {type(where).__name__}")

        current_depth = context.execution_depth
        current_path = context.operation_path

        if expanded_policy and 'preload' in expanded_policy:
            policy_context: PolicyContext = {
                'phase': 'preload',
                'hook': 'preload',
                'query': query,
                'current_ast': query,  # For top-level, current == query
                'query_type': detect_query_type(query),
                'execution_depth': current_depth,  # Add execution depth
                'operation_path': current_path,  # Add operation path
                'parent_operation': 'query' if current_depth == 0 else current_path.rsplit('.', 1)[0],
                '_policy_depth': policy_depth
            }

            try:
                expanded_policy['preload'](policy_context)
            except PolicyException as e:
                if e.query_type is None:
                    e.query_type = policy_context.get('query_type')
                raise

        dispatch_self = self
        compiled_query = None

        if where_param and isinstance(query, (dict, ASTLet)):
            raise ValueError("where must be provided inside dict chain under the 'where' key")
        if isinstance(query, str):
            if where_param:
                raise ValueError("where cannot be combined with string queries; embed Cypher predicates in the query itself")
            if language is None and not _looks_like_cypher_query(query):
                raise TypeError("Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict. Got str")
            compiled_query = _compile_string_query(query, language=language, params=params)
            if isinstance(compiled_query, CompiledCypherGraphQuery):
                return _execute_graph_query(self, compiled_query, engine=engine, policy=policy, context=context)
            if isinstance(compiled_query, CompiledCypherQuery):
                if compiled_query.graph_bindings or compiled_query.use_ref:
                    return _execute_query_with_graph_context(self, compiled_query, engine=engine, policy=policy, context=context)
                query = compiled_query.chain
        else:
            if language is not None:
                raise ValueError("language is only supported when query is a string")
            if params is not None:
                raise ValueError("params is only supported when query is a string")

        if isinstance(query, dict) and "chain" in query:
            chain_items: List[ASTObject] = []
            for item in query["chain"]:
                if isinstance(item, dict):
                    from .ast import from_json
                    chain_items.append(from_json(item))
                elif isinstance(item, ASTObject):
                    chain_items.append(item)
                else:
                    raise TypeError(f"Unsupported chain entry type: {type(item)}")
            dict_where = parse_where_json(query.get("where"))
            if not chain_items and dict_where:
                raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")
            query = Chain(chain_items, where=dict_where)
        elif isinstance(query, dict):
            wrapped_dict = {}
            for key, value in query.items():
                if isinstance(value, (ASTNode, ASTEdge)):
                    logger.debug(f'Auto-wrapping {type(value).__name__} in Chain for dict key "{key}"')
                    wrapped_dict[key] = Chain([value])
                else:
                    wrapped_dict[key] = value
            query = ASTLet(wrapped_dict)  # type: ignore

        context.push_depth()

        query_segment = 'dag' if isinstance(query, ASTLet) else 'chain'
        context.push_path(query_segment)

        try:
            if compiled_query is not None and not isinstance(query, Chain):
                logger.debug('GFQL executing compiled string program')
                return _execute_compiled_query(
                    self,
                    compiled_query=compiled_query,
                    engine=engine,
                    policy=expanded_policy,
                    context=context,
                )
            if isinstance(query, ASTLet):
                logger.debug('GFQL executing as DAG')
                return chain_let_impl(dispatch_self, query, engine, output, policy=expanded_policy, context=context)
            elif isinstance(query, Chain):
                logger.debug('GFQL executing as Chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                if where_param:
                    if query.where:
                        raise ValueError("where provided for Chain that already includes where")
                    query = Chain(query.chain, where=where_param)
                if compiled_query is not None:
                    start_nodes = None
                    if compiled_query.start_nodes_query is not None:
                        prefix_result = _execute_compiled_query(
                            self,
                            compiled_query=compiled_query.start_nodes_query,
                            engine=engine,
                            policy=expanded_policy,
                            context=context,
                        )
                        start_nodes = _compiled_query_start_nodes(compiled_query, prefix_result, engine=engine)
                    return _execute_compiled_query(
                        self,
                        compiled_query=compiled_query,
                        engine=engine,
                        policy=expanded_policy,
                        context=context,
                        start_nodes=start_nodes,
                    )
                return _chain_dispatch(dispatch_self, query, engine, expanded_policy, context)
            elif isinstance(query, ASTObject):
                logger.debug('GFQL executing single ASTObject as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                return _chain_dispatch(dispatch_self, Chain([query], where=where_param), engine, expanded_policy, context)
            elif isinstance(query, list):
                logger.debug('GFQL executing list as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')

                if not query and where_param:
                    raise ValueError("where requires at least one named node/edge step; empty chains have no aliases")

                converted_query: List[ASTObject] = []
                for item in query:
                    if isinstance(item, dict):
                        from .ast import from_json
                        converted_query.append(from_json(item))
                    else:
                        converted_query.append(item)

                return _chain_dispatch(
                    dispatch_self,
                    Chain(converted_query, where=where_param),
                    engine,
                    expanded_policy,
                    context,
                )
            else:
                raise TypeError(
                    f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, dict, or string. "
                    f"Got {type(query).__name__}"
                )
        finally:
            context.pop_depth()
            context.pop_path()
    finally:
        if policy:
            context.policy_depth = policy_depth


def _chain_dispatch(
    g: Plottable,
    chain_obj: Chain,
    engine: Union[EngineAbstract, str],
    policy: Optional[PolicyDict],
    context: ExecutionContext,
    start_nodes: Optional[DataFrameT] = None,
) -> Plottable:
    if chain_obj.where:
        if start_nodes is not None:
            raise GFQLValidationError(
                ErrorCode.E108,
                "Cypher MATCH after WITH does not yet support re-entry into MATCH chains with same-path WHERE constraints",
                field="match",
                value="where",
                suggestion="Use a simpler trailing MATCH without additional same-path WHERE constraints.",
                language="cypher",
            )
        first_row_pipeline_idx = next(
            (
                idx
                for idx, op in enumerate(chain_obj.chain)
                if isinstance(op, ASTCall) and is_row_pipeline_call(op.function)
            ),
            None,
        )
        if first_row_pipeline_idx is not None:
            same_path_prefix = chain_obj.chain[:first_row_pipeline_idx]
            row_suffix = chain_obj.chain[first_row_pipeline_idx:]
            validate_chain_schema(g, same_path_prefix, collect_all=False)
            is_cudf = engine == EngineAbstract.CUDF or engine == "cudf"
            engine_enum = Engine.CUDF if is_cudf else Engine.PANDAS
            inputs = build_same_path_inputs(
                g,
                same_path_prefix,
                chain_obj.where,
                engine=engine_enum,
                include_paths=False,
            )
            executor = DFSamePathExecutor(inputs)
            matched = _materialize_split_alias_columns(executor.run(), executor)
            return chain_impl(matched, row_suffix, engine, policy=policy, context=context)
        validate_chain_schema(g, chain_obj.chain, collect_all=False)
        is_cudf = engine == EngineAbstract.CUDF or engine == "cudf"
        engine_enum = Engine.CUDF if is_cudf else Engine.PANDAS
        inputs = build_same_path_inputs(
            g,
            chain_obj.chain,
            chain_obj.where,
            engine=engine_enum,
            include_paths=False,
        )
        return execute_same_path_chain(
            inputs.graph,
            inputs.chain,
            inputs.where,
            inputs.engine,
            inputs.include_paths,
        )
    return chain_impl(g, chain_obj.chain, engine, policy=policy, context=context, start_nodes=start_nodes)
