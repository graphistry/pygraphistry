import logging
from typing import Dict, Union, cast, List, Tuple, Sequence, Optional, TYPE_CHECKING
from graphistry.Engine import Engine, EngineAbstract, df_concat, resolve_engine

from graphistry.Plottable import Plottable
from graphistry.compute.ASTSerializable import ASTSerializable
from graphistry.util import setup_logger
from graphistry.utils.json import JSONVal
from .ast import ASTObject, ASTNode, ASTEdge, from_json as ASTObject_from_json
from .typing import DataFrameT
from graphistry.compute.validate.validate_schema import validate_chain_schema

if TYPE_CHECKING:
    from graphistry.compute.exceptions import GFQLSchemaError, GFQLValidationError

logger = setup_logger(__name__)


###############################################################################


class Chain(ASTSerializable):

    def __init__(self, chain: List[ASTObject]) -> None:
        self.chain = chain

    def validate(self, collect_all: bool = False) -> Optional[List['GFQLValidationError']]:
        """Override to collect all chain validation errors."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError, GFQLValidationError
        
        if not collect_all:
            # Use parent's fail-fast implementation
            return super().validate(collect_all=False)
        
        # Collect all errors mode
        errors: List[GFQLValidationError] = []
        
        # Check if chain is a list
        if not isinstance(self.chain, list):
            errors.append(GFQLTypeError(
                ErrorCode.E101,
                f"Chain must be a list, but got {type(self.chain).__name__}. Wrap your operations in a list []."
            ))
            return errors  # Can't continue if not a list
        
        # Check each operation
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                errors.append(GFQLTypeError(
                    ErrorCode.E101,
                    f"Chain operation at index {i} is not a valid GFQL operation. Got {type(op).__name__} instead of an ASTObject.",
                    operation_index=i,
                    actual_type=type(op).__name__,
                    suggestion="Use n() for nodes, e() for edges, or other GFQL operations"
                ))
        
        # Validate child AST nodes
        for child in self._get_child_validators():
            child_errors = child.validate(collect_all=True)
            if child_errors:
                errors.extend(child_errors)
        
        return errors
    
    def _validate_fields(self) -> None:
        """Validate Chain fields."""
        from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
        
        if not isinstance(self.chain, list):
            raise GFQLTypeError(
                ErrorCode.E101,
                f"Chain must be a list, but got {type(self.chain).__name__}. Wrap your operations in a list []."
            )
        
        for i, op in enumerate(self.chain):
            if not isinstance(op, ASTObject):
                raise GFQLTypeError(
                    ErrorCode.E101,
                    f"Chain operation at index {i} is not a valid GFQL operation. Got {type(op).__name__} instead of an ASTObject.",
                    operation_index=i,
                    actual_type=type(op).__name__,
                    suggestion="Use n() for nodes, e() for edges, or other GFQL operations"
                )
    
    def _get_child_validators(self) -> List[ASTSerializable]:
        """Return child AST nodes that need validation."""
        # Only return valid ASTObject instances
        return cast(List[ASTSerializable], [op for op in self.chain if isinstance(op, ASTObject)])

    @classmethod
    def from_json(cls, d: Dict[str, JSONVal], validate: bool = True) -> 'Chain':
        """
        Convert a JSON AST into a list of ASTObjects
        """
        from graphistry.compute.exceptions import ErrorCode, GFQLSyntaxError
        
        if not isinstance(d, dict):
            raise GFQLSyntaxError(
                ErrorCode.E101,
                f"Chain JSON must be a dictionary, got {type(d).__name__}"
            )
        
        if 'chain' not in d:
            raise GFQLSyntaxError(
                ErrorCode.E105,
                "Chain JSON missing required 'chain' field"
            )
        
        if not isinstance(d['chain'], list):
            raise GFQLSyntaxError(
                ErrorCode.E101,
                f"Chain field must be a list, got {type(d['chain']).__name__}"
            )
        
        out = cls([ASTObject_from_json(op, validate=validate) for op in d['chain']])
        if validate:
            out.validate()
        return out

    def to_json(self, validate=True) -> Dict[str, JSONVal]:
        """
        Convert a list of ASTObjects into a JSON AST
        """
        if validate:
            self.validate()
        return {
            'type': self.__class__.__name__,
            'chain': [op.to_json() for op in self.chain]
        }

    def validate_schema(self, g: Plottable, collect_all: bool = False) -> Optional[List['GFQLSchemaError']]:
        """Validate this chain against a graph's schema without executing.

        Args:
            g: Graph to validate against
            collect_all: If True, collect all errors. If False, raise on first.

        Returns:
            If collect_all=True: List of errors (empty if valid)  
            If collect_all=False: None if valid

        Raises:
            GFQLSchemaError: If collect_all=False and validation fails
        """
        return validate_chain_schema(g, self, collect_all)


###############################################################################


def combine_steps(g: Plottable, kind: str, steps: List[Tuple[ASTObject,Plottable]], engine: Engine) -> DataFrameT:
    """
    Collect nodes and edges, taking care to deduplicate and tag any names
    """

    id = getattr(g, '_node' if kind == 'nodes' else '_edge')
    df_fld = '_nodes' if kind == 'nodes' else '_edges'
    op_type = ASTNode if kind == 'nodes' else ASTEdge

    if id is None:
        raise ValueError(f'Cannot combine steps with empty id for kind {kind}')

    logger.debug('combine_steps ops pre: %s', [op for (op, _) in steps])
    if kind == 'edges':
        logger.debug('EDGES << recompute forwards given reduced set')
        steps = [
            (
                op,  # forward op
                op(
                    g=g.edges(g_step._edges),  # transition via any found edge
                    prev_node_wavefront=g_step._nodes,  # start from where backwards step says is reachable

                    #target_wave_front=steps[i+1][1]._nodes  # end at where next backwards step says is reachable
                    target_wave_front=None,  # ^^^ optimization: valid transitions already limit to known-good ones
                    engine=engine
                )
            )
            for (op, g_step) in steps
        ]

    concat = df_concat(engine)

    logger.debug('-----------[ combine %s ---------------]', kind)

    # df[[id]] - with defensive checks for column existence
    dfs_to_concat = []
    for (op, g_step) in steps:
        step_df = getattr(g_step, df_fld)
        if id not in step_df.columns:
            step_id = getattr(g_step, '_node' if kind == 'nodes' else '_edge')
            raise ValueError(f"Column '{id}' not found in {kind} step DataFrame. "
                           f"Step has id='{step_id}', available columns: {list(step_df.columns)}. "
                           f"Operation: {op}")
        dfs_to_concat.append(step_df[[id]])
    
    out_df = concat(dfs_to_concat).drop_duplicates(subset=[id])
    if logger.isEnabledFor(logging.DEBUG):
        for (op, g_step) in steps:
            if kind == 'edges':
                logger.debug('adding edges to concat: %s', g_step._edges[[g_step._source, g_step._destination]])
            else:
                logger.debug('adding nodes to concat: %s', g_step._nodes[[g_step._node]])

    # df[[id, op_name1, ...]]
    logger.debug('combine_steps ops: %s', [op for (op, _) in steps])
    for (op, g_step) in steps:
        if op._name is not None and isinstance(op, op_type):
            logger.debug('tagging kind [%s] name %s', op_type, op._name)
            out_df = out_df.merge(
                getattr(g_step, df_fld)[[id, op._name]],
                on=id,
                how='left'
            )
            s = out_df[op._name]
            out_df[op._name] = s.where(s.notna(), False).astype('bool')
    out_df = out_df.merge(getattr(g, df_fld), on=id, how='left')

    logger.debug('COMBINED[%s] >>\n%s', kind, out_df)

    return out_df


###############################################################################
#
#  Implementation: The algorithm performs three phases -
#
#     1. Forward wavefront (slowed)
#
#     Each step is processed, yielding the nodes it matches based on the nodes reached by the previous step
#
#     Full node/edge table merges are happening, so any pre-filtering would help
#
#     2. Reverse pruning pass  (fastish)
#
#     Some paths traversed during Step 1 are deadends that must be pruned
#    
#     To only pick nodes on full paths, we then run in a reverse pass on a graph subsetted to nodes along full/partial paths.
#
#     - Every node encountered on the reverse pass is guaranteed to be on a full path
#    
#     - Every 'good' node will be encountered
#    
#     - No 'bad' deadend nodes will be included
#
#     3. Forward output pass
#
#     This pass is likely fusable into Step 2: collect and label outputs
#
###############################################################################

def chain(self: Plottable, ops: Union[List[ASTObject], Chain], engine: Union[EngineAbstract, str] = EngineAbstract.AUTO, validate_schema: bool = True, policy=None, context=None) -> Plottable:
    """
    Chain a list of ASTObject (node/edge) traversal operations

    Return subgraph of matches according to the list of node & edge matchers
    If any matchers are named, add a correspondingly named boolean-valued column to the output

    For direct calls, exposes convenience `List[ASTObject]`. Internal operational should prefer `Chain`.

    Use `engine='cudf'` to force automatic GPU acceleration mode

    :param ops: List[ASTObject] Various node and edge matchers
    :param validate_schema: Whether to validate the chain against the graph schema before executing
    :param policy: Optional policy dict for hooks
    :param context: Optional ExecutionContext for tracking execution state

    :returns: Plotter
    :rtype: Plotter
    """
    # Create context if not provided
    if context is None:
        from .execution_context import ExecutionContext
        context = ExecutionContext()

    # If policy provided, set it in thread-local for ASTCall operations
    if policy:
        from graphistry.compute.gfql.call_executor import _thread_local as call_thread_local
        old_policy = getattr(call_thread_local, 'policy', None)
        try:
            call_thread_local.policy = policy
            return _chain_impl(self, ops, engine, validate_schema, policy, context)
        finally:
            call_thread_local.policy = old_policy
    else:
        return _chain_impl(self, ops, engine, validate_schema, policy, context)


def _chain_impl(self: Plottable, ops: Union[List[ASTObject], Chain], engine: Union[EngineAbstract, str], validate_schema: bool, policy, context) -> Plottable:
    """
    Internal implementation of chain without policy wrapper indentation.

    **Example: Find nodes of some type**

    ::

            from graphistry.ast import n

            people_nodes_df = g.chain([ n({"type": "person"}) ])._nodes
            
    **Example: Find 2-hop edge sequences with some attribute**

    ::

            from graphistry.ast import e_forward

            g_2_hops = g.chain([ e_forward({"interesting": True}, hops=2) ])
            g_2_hops.plot()

    **Example: Find any node 1-2 hops out from another node, and label each hop**

    ::

            from graphistry.ast import n, e_undirected

            g_2_hops = g.chain([ n({g._node: "a"}), e_undirected(name="hop1"), e_undirected(name="hop2") ])
            print('# first-hop edges:', len(g_2_hops._edges[ g_2_hops._edges.hop1 == True ]))

    **Example: Transaction nodes between two kinds of risky nodes**

    ::

            from graphistry.ast import n, e_forward, e_reverse

            g_risky = g.chain([
                n({"risk1": True}),
                e_forward(to_fixed=True),
                n({"type": "transaction"}, name="hit"),
                e_reverse(to_fixed=True),
                n({"risk2": True})
            ])
            print('# hits:', len(g_risky._nodes[ g_risky._nodes.hit ]))

    **Example: Filter by multiple node types at each step using is_in**

    ::

            from graphistry.ast import n, e_forward, e_reverse, is_in

            g_risky = g.chain([
                n({"type": is_in(["person", "company"])}),
                e_forward({"e_type": is_in(["owns", "reviews"])}, to_fixed=True),
                n({"type": is_in(["transaction", "account"])}, name="hit"),
                e_reverse(to_fixed=True),
                n({"risk2": True})
            ])
            print('# hits:', len(g_risky._nodes[ g_risky._nodes.hit ]))
    
    **Example: Run with automatic GPU acceleration**

    ::

            import cudf
            import graphistry

            e_gdf = cudf.from_pandas(df)
            g1 = graphistry.edges(e_gdf, 's', 'd')
            g2 = g1.chain([ ... ])

    **Example: Run with automatic GPU acceleration, and force GPU mode**

    ::

            import cudf
            import graphistry

            e_gdf = cudf.from_pandas(df)
            g1 = graphistry.edges(e_gdf, 's', 'd')
            g2 = g1.chain([ ... ], engine='cudf')

    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    if isinstance(ops, Chain):
        ops = ops.chain

    # Recursive dispatch for schema-changing operations (UMAP, hypergraph, etc.)
    # These operations create entirely new graph structures, so we split the chain
    # and execute segments sequentially: before → schema_changer → rest
    from graphistry.compute.ast import ASTCall

    # Extensible list of schema-changing operations
    schema_changers = ['umap', 'hypergraph']

    # Find first schema-changer in ops
    schema_changer_idx = None
    for i, op in enumerate(ops):
        if isinstance(op, ASTCall) and op.function in schema_changers:
            schema_changer_idx = i
            break

    if schema_changer_idx is not None:
        if len(ops) == 1:
            # Singleton schema-changer - execute directly without going through chain machinery
            from graphistry.compute.gfql.call_executor import execute_call
            from graphistry.compute.exceptions import GFQLTypeError, ErrorCode

            engine_concrete = resolve_engine(engine, self)
            schema_changer = ops[0]

            # Type narrowing: we know it's ASTCall from the isinstance check above
            if not isinstance(schema_changer, ASTCall):
                raise GFQLTypeError(
                    code=ErrorCode.E201,
                    message="Schema-changer operation must be ASTCall",
                    field="operation",
                    value=type(schema_changer).__name__,
                    suggestion="Use call('umap', {...}) or call('hypergraph', {...})"
                )

            # Validate schema if requested (even though ASTCall doesn't check columns, respect the flag)
            if validate_schema:
                validate_chain_schema(self, ops, collect_all=False)

            return execute_call(self, schema_changer.function, schema_changer.params, engine_concrete, policy=policy, context=context)
        else:
            # Multiple ops with schema-changer - split and recurse
            before = ops[:schema_changer_idx]
            schema_changer = ops[schema_changer_idx]
            rest = ops[schema_changer_idx + 1:]

            # Execute segments: before → schema_changer → rest
            # Recursion handles multiple schema-changers automatically
            g_temp = self.chain(before, engine=engine, validate_schema=validate_schema, policy=policy, context=context) if before else self  # type: ignore[call-arg]
            g_temp2 = g_temp.chain([schema_changer], engine=engine, validate_schema=validate_schema, policy=policy, context=context)  # type: ignore[call-arg]
            return g_temp2.chain(rest, engine=engine, validate_schema=validate_schema, policy=policy, context=context) if rest else g_temp2  # type: ignore[call-arg]

    if validate_schema:
        validate_chain_schema(self, ops, collect_all=False)

    if len(ops) == 0:
        return self

    logger.debug('orig chain >> %s', ops)

    engine_concrete = resolve_engine(engine, self)
    logger.debug('chain engine: %s => %s', engine, engine_concrete)

    if isinstance(ops[0], ASTEdge):
        logger.debug('adding initial node to ensure initial link has needed reversals')
        ops = cast(List[ASTObject], [ ASTNode() ]) + ops

    if isinstance(ops[-1], ASTEdge):
        logger.debug('adding final node to ensure final link has needed reversals')
        ops = ops + cast(List[ASTObject], [ ASTNode() ])

    logger.debug('final chain >> %s', ops)

    # Initialize variables for finally block
    g_out = None
    error = None
    success = False

    try:
        g = self.materialize_nodes(engine=EngineAbstract(engine_concrete.value))

        # Store original edge binding to restore it if we add temporary index
        original_edge = g._edge

        # Handle node-only graphs (e.g., for hypergraph transformation)
        if g._edges is None:
            added_edge_index = False
        elif g._edge is None:
            if 'index' in g._edges.columns:
                raise ValueError('Edges cannot have column "index", please remove or set as g._edge via bind() or edges()')
            added_edge_index = True
            indexed_edges_df = g._edges.reset_index()
            g = g.edges(indexed_edges_df, edge='index')
        else:
            added_edge_index = False

        # Prechain hook - fires BEFORE chain operations execute
        if policy and 'prechain' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            stats = extract_graph_stats(g)
            current_path = context.operation_path

            prechain_context: PolicyContext = {
                'phase': 'prechain',
                'hook': 'prechain',
                'query': ops,
                'current_ast': ops,
                'query_type': 'chain',
                'plottable': g,
                'graph_stats': stats,
                'execution_depth': context.execution_depth,
                'operation_path': current_path,
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            try:
                policy['prechain'](prechain_context)
            except PolicyException:
                raise

        logger.debug('======================== FORWARDS ========================')

        # Forwards
        # This computes valid path *prefixes*, where each g nodes/edges is the path wavefront:
        #  g_step._nodes: The nodes reached in this step
        #  g_step._edges: The edges used to reach those nodes
        # At the paths are prefixes, wavefront nodes may invalid wrt subsequent steps (e.g., halt early)
        g_stack : List[Plottable] = []
        for op in ops:
            prev_step_nodes = (  # start from only prev step's wavefront node
                None  # first uses full graph
                if len(g_stack) == 0
                else g_stack[-1]._nodes
            )
            g_step = (
                op(
                    g=g,  # transition via any original edge
                    prev_node_wavefront=prev_step_nodes,
                    target_wave_front=None,  # implicit any
                    engine=engine_concrete
                )
            )
            g_stack.append(g_step)

        import logging
        if logger.isEnabledFor(logging.DEBUG):
            for (i, g_step) in enumerate(g_stack):
                logger.debug('~' * 10 + '\nstep %s', i)
                logger.debug('nodes: %s', g_step._nodes)
                logger.debug('edges: %s', g_step._edges)

        logger.debug('======================== BACKWARDS ========================')

        # Backwards
        # Compute reverse and thus complete paths. Dropped nodes/edges are thus the incomplete path prefixes.
        # Each g node/edge represents a valid wavefront entry for that step.
        g_stack_reverse : List[Plottable] = []
        for (op, g_step) in zip(reversed(ops), reversed(g_stack)):
            prev_loop_step = g_stack[-1] if len(g_stack_reverse) == 0 else g_stack_reverse[-1]
            if len(g_stack_reverse) == len(g_stack) - 1:
                prev_orig_step = None
            else:
                prev_orig_step = g_stack[-(len(g_stack_reverse) + 2)]
            assert prev_loop_step._nodes is not None
            g_step_reverse = (
                (op.reverse())(

                    # Edges: edges used in step (subset matching prev_node_wavefront will be returned)
                    # Nodes: nodes reached in step (subset matching prev_node_wavefront will be returned)
                    g=g_step,

                    # check for hits against fully valid targets
                    # ast will replace g.node() with this as its starting points
                    prev_node_wavefront=prev_loop_step._nodes,

                    # only allow transitions to these nodes (vs prev_node_wavefront)
                    target_wave_front=prev_orig_step._nodes if prev_orig_step is not None else None,

                    engine=engine_concrete
                )
            )
            g_stack_reverse.append(g_step_reverse)

        import logging
        if logger.isEnabledFor(logging.DEBUG):
            for (i, g_step) in enumerate(g_stack_reverse):
                logger.debug('~' * 10 + '\nstep %s', i)
                logger.debug('nodes: %s', g_step._nodes)
                logger.debug('edges: %s', g_step._edges)

        logger.debug('============ COMBINE NODES ============')
        final_nodes_df = combine_steps(g, 'nodes', list(zip(ops, reversed(g_stack_reverse))), engine_concrete)

        logger.debug('============ COMBINE EDGES ============')
        final_edges_df = combine_steps(g, 'edges', list(zip(ops, reversed(g_stack_reverse))), engine_concrete)
        if added_edge_index:
            final_edges_df = final_edges_df.drop(columns=['index'])
            # Fix: Restore original edge binding instead of using modified 'index' binding
            g_out = self.nodes(final_nodes_df).edges(final_edges_df, edge=original_edge)
        else:
            g_out = g.nodes(final_nodes_df).edges(final_edges_df)

        # Mark as successful
        success = True

    except Exception as e:
        # Capture error for postload hook
        error = e
        # Don't re-raise yet - let finally block run first

    finally:
        # Postchain hook - fires AFTER chain operations complete (even on error)
        postchain_policy_error = None
        if policy and 'postchain' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, g_out is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, g_out) if success else self
            stats = extract_graph_stats(graph_for_stats)
            current_path = context.operation_path

            postchain_context: PolicyContext = {
                'phase': 'postchain',
                'hook': 'postchain',
                'query': ops,
                'current_ast': ops,
                'query_type': 'chain',
                'plottable': graph_for_stats,
                'graph_stats': stats,
                'success': success,
                'execution_depth': context.execution_depth,
                'operation_path': current_path,
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            # Add error information if execution failed
            if error is not None:
                postchain_context['error'] = str(error)  # type: ignore
                postchain_context['error_type'] = type(error).__name__  # type: ignore

            try:
                policy['postchain'](postchain_context)
            except PolicyException as e:
                # Capture policy error instead of raising immediately
                postchain_policy_error = e

        # Postload policy phase - ALWAYS fires (even on error)
        policy_error = None
        if policy and 'postload' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, g_out is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, g_out) if success else self
            stats = extract_graph_stats(graph_for_stats)

            policy_context: PolicyContext = {
                'phase': 'postload',
                'hook': 'postload',
                'query': ops,
                'current_ast': ops,  # For chain, current == ops
                'query_type': 'chain',
                'plottable': graph_for_stats,  # RESULT or INPUT
                'graph_stats': stats,
                'success': success,  # True if successful, False if error
                'execution_depth': context.execution_depth,  # Add execution depth
                '_policy_depth': getattr(ops, '_policy_depth', 0) if hasattr(ops, '_policy_depth') else 0
            }

            # Add error information if execution failed
            if error is not None:
                policy_context['error'] = str(error)  # type: ignore
                policy_context['error_type'] = type(error).__name__  # type: ignore

            try:
                # Policy can only accept (None) or deny (exception)
                policy['postload'](policy_context)

            except PolicyException as e:
                # Enrich exception with context if not already set
                if e.query_type is None:
                    e.query_type = 'chain'
                if e.data_size is None:
                    e.data_size = stats
                # Capture policy error instead of raising immediately
                policy_error = e

    # After finally block, decide which error to raise
    # Priority: postchain PolicyException > postload PolicyException > operation error
    if postchain_policy_error is not None:
        # postchain policy error takes highest priority
        if error is not None:
            raise postchain_policy_error from error
        else:
            raise postchain_policy_error
    elif policy_error is not None:
        # postload policy error is second priority
        if error is not None:
            raise policy_error from error
        else:
            raise policy_error
    elif error is not None:
        raise error

    # Cast: At this point, all error paths have been handled, so g_out is guaranteed to be a Plottable
    return cast(Plottable, g_out)
