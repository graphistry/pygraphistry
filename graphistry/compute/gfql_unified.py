"""GFQL unified entrypoint for chains and DAGs"""

from typing import List, Union, Optional, Dict, Any
from graphistry.Plottable import Plottable
from graphistry.Engine import EngineAbstract
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

logger = setup_logger(__name__)


def detect_query_type(query: Any) -> QueryType:
    """Detect query type for policy context.

    Returns:
        'dag' for ASTLet queries
        'chain' for list/Chain queries
        'single' for single ASTObject queries
    """
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
         policy: Optional[Dict[str, PolicyFunction]] = None) -> Plottable:
    """
    Execute a GFQL query - either a chain or a DAG

    Unified entrypoint that automatically detects query type and
    dispatches to the appropriate execution engine.

    :param query: GFQL query - ASTObject, List[ASTObject], Chain, ASTLet, or dict
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: For DAGs, name of binding to return (default: last executed)
    :param policy: Optional policy hooks for external control (preload, postload, precall, postcall phases)
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
    # Create ExecutionContext at start
    context = ExecutionContext()

    # Recursion prevention - check if we're already in a policy execution
    if policy and context.policy_depth >= 1:
        logger.debug('Policy disabled due to recursion depth limit (depth=%d)', context.policy_depth)
        policy = None  # Disable policy for recursive calls

    # Set depth for this execution
    policy_depth = context.policy_depth
    if policy:
        context.policy_depth = policy_depth + 1

    # Expand policy shortcuts to full hook names (e.g., 'pre' → all pre* hooks)
    expanded_policy: Optional[PolicyDict] = None
    if policy:
        expanded_policy = expand_policy(policy)

    try:
        # Get current execution depth (0 for top-level)
        current_depth = context.execution_depth
        current_path = context.operation_path

        # Preload policy phase - before any processing
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
                # Policy can only accept (None) or deny (exception)
                expanded_policy['preload'](policy_context)

            except PolicyException as e:
                # Enrich exception with context if not already set
                if e.query_type is None:
                    e.query_type = policy_context.get('query_type')
                raise

        # Handle dict convenience first (convert to ASTLet)
        if isinstance(query, dict):
            # Auto-wrap ASTNode and ASTEdge values in Chain for GraphOperation compatibility
            wrapped_dict = {}
            for key, value in query.items():
                if isinstance(value, (ASTNode, ASTEdge)):
                    logger.debug(f'Auto-wrapping {type(value).__name__} in Chain for dict key "{key}"')
                    wrapped_dict[key] = Chain([value])
                else:
                    wrapped_dict[key] = value
            query = ASTLet(wrapped_dict)  # type: ignore

        # Push execution depth and operation path before dispatching
        # This moves us from depth 0 (gfql entry) to depth 1 (chain/let execution)
        context.push_depth()

        # Determine query type segment for operation path
        query_segment = 'dag' if isinstance(query, ASTLet) else 'chain'
        context.push_path(query_segment)

        try:
            # Dispatch based on type - check specific types before generic
            if isinstance(query, ASTLet):
                logger.debug('GFQL executing as DAG')
                return chain_let_impl(self, query, engine, output, policy=expanded_policy, context=context)
            elif isinstance(query, Chain):
                logger.debug('GFQL executing as Chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                return chain_impl(self, query.chain, engine, policy=expanded_policy, context=context)
            elif isinstance(query, ASTObject):
                # Single ASTObject -> execute as single-item chain
                logger.debug('GFQL executing single ASTObject as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')
                return chain_impl(self, [query], engine, policy=expanded_policy, context=context)
            elif isinstance(query, list):
                logger.debug('GFQL executing list as chain')
                if output is not None:
                    logger.warning('output parameter ignored for chain queries')

                # Convert any dictionaries in the list to AST objects
                converted_query: List[ASTObject] = []
                for item in query:
                    if isinstance(item, dict):
                        from .ast import from_json
                        converted_query.append(from_json(item))
                    else:
                        converted_query.append(item)

                return chain_impl(self, converted_query, engine, policy=expanded_policy, context=context)
            else:
                raise TypeError(
                    f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict. "
                    f"Got {type(query).__name__}"
                )
        finally:
            # Pop execution depth and operation path when returning
            context.pop_depth()
            context.pop_path()
    finally:
        # Reset policy depth
        if policy:
            context.policy_depth = policy_depth
