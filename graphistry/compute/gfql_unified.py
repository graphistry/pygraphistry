"""GFQL unified entrypoint for chains and DAGs"""
# ruff: noqa: E501

from typing import List, Union, Optional, Dict, Any, Sequence
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTNode, ASTEdge
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
from graphistry.compute.gfql.df_executor import (
    build_same_path_inputs,
    execute_same_path_chain,
)
from graphistry.compute.validate.validate_schema import validate_chain_schema

logger = setup_logger(__name__)


def detect_query_type(query: Any) -> QueryType:
    if isinstance(query, ASTLet):
        return "dag"
    elif isinstance(query, (list, Chain)):
        return "chain"
    else:
        return "single"


def gfql(self: Plottable,
         query: Union[ASTObject, List[ASTObject], ASTLet, Chain, dict],
         engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
         output: Optional[str] = None,
         policy: Optional[Dict[str, PolicyFunction]] = None,
         where: Optional[Sequence[WhereComparison]] = None) -> Plottable:
    """
    Execute a GFQL query - either a chain or a DAG

    Unified entrypoint that automatically detects query type and
    dispatches to the appropriate execution engine.

    :param query: GFQL query - ASTObject, List[ASTObject], Chain, ASTLet, or dict
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: For DAGs, name of binding to return (default: last executed)
    :param policy: Optional policy hooks for external control (preload, postload, precall, postcall phases)
    :param where: Optional same-path constraints for list/Chain queries
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

        if where_param and isinstance(query, (dict, ASTLet)):
            raise ValueError("where must be provided inside dict chain under the 'where' key")

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
            if isinstance(query, ASTLet):
                logger.debug('GFQL executing as DAG')
                return chain_let_impl(self, query, engine, output, policy=expanded_policy, context=context)
            elif isinstance(query, Chain):
                logger.debug('GFQL executing as Chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                if where_param:
                    if query.where:
                        raise ValueError("where provided for Chain that already includes where")
                    query = Chain(query.chain, where=where_param)
                return _chain_dispatch(self, query, engine, expanded_policy, context)
            elif isinstance(query, ASTObject):
                logger.debug('GFQL executing single ASTObject as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                return _chain_dispatch(self, Chain([query], where=where_param), engine, expanded_policy, context)
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
                    self,
                    Chain(converted_query, where=where_param),
                    engine,
                    expanded_policy,
                    context,
                )
            else:
                raise TypeError(
                    f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict. "
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
) -> Plottable:
    if chain_obj.where:
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
    return chain_impl(g, chain_obj.chain, engine, policy=policy, context=context)
