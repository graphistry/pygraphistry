from typing import Dict, Set, List, Optional, Tuple, Union, cast, TYPE_CHECKING
from typing_extensions import Literal
import pandas as pd
from graphistry.Engine import Engine, EngineAbstract, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTRef, ASTRemoteGraph, ASTNode, ASTEdge, ASTCall
from .execution_context import ExecutionContext
from .engine_coercion import ensure_engine_match

if TYPE_CHECKING:
    from graphistry.compute.chain import Chain

logger = setup_logger(__name__)


def extract_dependencies(ast_obj: Union[ASTObject, 'Chain', 'Plottable']) -> Set[str]:
    """Recursively find all ASTRef references in an AST object or GraphOperation
    
    :param ast_obj: AST object or GraphOperation to analyze
    :returns: Set of referenced binding names
    :rtype: Set[str]
    """
    from graphistry.compute.chain import Chain
    from graphistry.Plottable import Plottable
    
    deps = set()
    
    if isinstance(ast_obj, ASTRef):
        deps.add(ast_obj.ref)
        # Also check chain operations
        for op in ast_obj.chain:
            deps.update(extract_dependencies(op))
    
    elif isinstance(ast_obj, ASTLet):
        # Nested let bindings
        for binding in ast_obj.bindings.values():
            deps.update(extract_dependencies(binding))
    
    elif isinstance(ast_obj, Chain):
        # Chain may contain ASTRef operations
        for op in ast_obj.chain:
            if isinstance(op, ASTObject):
                deps.update(extract_dependencies(op))
    
    elif isinstance(ast_obj, Plottable):
        # Plottable instances have no dependencies
        pass
    
    # Other AST types (ASTCall, ASTRemoteGraph) have no dependencies
    return deps


def build_dependency_graph(bindings: Dict[str, Union[ASTObject, 'Chain', 'Plottable']]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build dependency and dependent mappings from bindings
    
    :param bindings: Dictionary of name -> GraphOperation bindings
    :returns: Tuple of (dependencies dict, dependents dict)
    :rtype: Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]
    """
    dependencies: Dict[str, Set[str]] = {}
    dependents: Dict[str, Set[str]] = {}
    
    for name, ast_obj in bindings.items():
        deps = extract_dependencies(ast_obj)
        dependencies[name] = deps
        
        # Build reverse mapping
        for dep in deps:
            if dep not in dependents:
                dependents[dep] = set()
            dependents[dep].add(name)
    
    return dependencies, dependents


def validate_dependencies(bindings: Dict[str, Union[ASTObject, 'Chain', 'Plottable']], 
                        dependencies: Dict[str, Set[str]]) -> None:
    """Check for missing references and self-cycles
    
    :param bindings: Dictionary of available GraphOperation bindings
    :param dependencies: Dictionary of dependencies per binding
    :raises ValueError: If missing references or self-cycles found
    """
    all_names = set(bindings.keys())
    
    for name, deps in dependencies.items():
        # Check self-reference
        if name in deps:
            raise ValueError(f"Self-reference cycle detected: '{name}' depends on itself")
        
        # Check missing references
        missing = deps - all_names
        if missing:
            raise ValueError(
                f"Node '{name}' references undefined nodes: {sorted(missing)}. "
                f"Available nodes: {sorted(all_names)}"
            )


def detect_cycles(dependencies: Dict[str, Set[str]]) -> Optional[List[str]]:
    """Use DFS to detect cycles and return the cycle path if found
    
    :param dependencies: Dictionary mapping nodes to their dependencies
    :returns: List representing cycle path if found, None otherwise
    :rtype: Optional[List[str]]
    """
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {node: WHITE for node in dependencies}
    
    def dfs(node: str, path: List[str]) -> Optional[List[str]]:
        color[node] = GRAY
        path.append(node)
        
        for neighbor in dependencies.get(node, set()):
            if color.get(neighbor, WHITE) == GRAY:
                # Found cycle - build cycle path
                cycle_start = path.index(neighbor)
                return path[cycle_start:] + [neighbor]
            
            if color.get(neighbor, WHITE) == WHITE:
                cycle = dfs(neighbor, path[:])
                if cycle:
                    return cycle
        
        color[node] = BLACK
        return None
    
    for node in dependencies:
        if color[node] == WHITE:
            cycle = dfs(node, [])
            if cycle:
                return cycle
    
    return None


def topological_sort(bindings: Dict[str, Union[ASTObject, 'Chain', 'Plottable']],
                    dependencies: Dict[str, Set[str]],
                    dependents: Dict[str, Set[str]]) -> List[str]:
    """Kahn's algorithm for topological sort"""
    # Calculate in-degrees
    in_degree = {name: len(dependencies.get(name, set())) for name in bindings}
    
    # Start with nodes that have no dependencies
    queue = [name for name, degree in in_degree.items() if degree == 0]
    result = []
    
    while queue:
        # Process node with no remaining dependencies
        current = queue.pop(0)
        result.append(current)
        
        # Update dependents
        for dependent in dependents.get(current, set()):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)
    
    if len(result) != len(bindings):
        # Cycle detected - use DFS to find it for better error
        cycle = detect_cycles(dependencies)
        if cycle:
            raise ValueError(
                f"Circular dependency detected: {' -> '.join(cycle)}. "
                "Please restructure your DAG to remove cycles."
            )
        else:
            # Should not happen, but be defensive
            raise ValueError("Failed to determine execution order (possible circular dependency)")
    
    return result


def determine_execution_order(bindings: Dict[str, Union[ASTObject, 'Chain', 'Plottable']]) -> List[str]:
    """Determine topological execution order for DAG bindings
    
    Validates dependencies and computes execution order that respects
    all dependencies. Detects cycles and missing references.
    
    :param bindings: Dictionary of name -> GraphOperation bindings
    :returns: List of binding names in execution order
    :rtype: List[str]
    :raises ValueError: If cycles detected or references missing
    """
    # Handle trivial cases
    if not bindings:
        return []
    if len(bindings) == 1:
        return list(bindings.keys())
    
    # Build dependency graph
    dependencies, dependents = build_dependency_graph(bindings)
    
    # Validate all references exist
    validate_dependencies(bindings, dependencies)
    
    # Check for cycles with detailed error
    cycle = detect_cycles(dependencies)
    if cycle:
        raise ValueError(
            f"Circular dependency detected: {' -> '.join(cycle)}. "
            "Please restructure your DAG to remove cycles."
        )
    
    # Compute topological sort
    return topological_sort(bindings, dependencies, dependents)


def execute_node(name: str, ast_obj: Union[ASTObject, 'Chain', 'Plottable'], g: Plottable,
                context: ExecutionContext, engine: Engine, policy=None, global_query=None) -> Plottable:
    """Execute a single node in the DAG

    Handles different GraphOperation types:
    - ASTLet: Recursive let execution
    - ASTRef: Reference resolution and chain execution
    - ASTCall: Method calls on graphs
    - ASTRemoteGraph: Remote graph loading
    - Chain: Chain operations on graphs
    - Plottable: Direct graph instances

    :param name: Binding name for this node
    :param ast_obj: GraphOperation to execute
    :param g: Input graph
    :param context: Execution context for storing/retrieving results
    :param engine: Engine to use (pandas/cudf)
    :param policy: Optional policy dictionary with preload/postload/precall/postcall hooks
    :param global_query: The global query AST for policy context
    :returns: Resulting Plottable
    :rtype: Plottable
    :raises ValueError: If reference not found in context
    :raises NotImplementedError: For unsupported types
    """
    logger.debug("Executing node '%s' of type %s", name, type(ast_obj).__name__)

    # Handle different AST object types
    if isinstance(ast_obj, ASTLet):
        # Nested let execution
        result = chain_let_impl(g, ast_obj, EngineAbstract(engine.value), policy=policy, context=context)
    elif isinstance(ast_obj, ASTRef):
        # Resolve reference from context
        try:
            referenced_result = context.get_binding(ast_obj.ref)
        except KeyError as e:
            available = sorted(context.get_all_bindings().keys())
            raise ValueError(
                f"Node '{name}' references '{ast_obj.ref}' which has not been executed yet. "
                f"Available bindings: {available}"
            ) from e

        # Execute the chain on the referenced result
        if ast_obj.chain:
            # Import chain function to execute the operations
            from .chain import chain as chain_impl
            chain_result = chain_impl(referenced_result, ast_obj.chain, EngineAbstract(engine.value), policy=policy, context=context)
            # ASTRef with chain should return the filtered result directly
            result = chain_result
        else:
            # Empty chain - just return the referenced result
            result = referenced_result
    elif isinstance(ast_obj, ASTNode):
        # ASTNode operates on the original graph (unless accessed via ASTRef)
        original_g = context.get_binding('__original_graph__') if context.has_binding('__original_graph__') else g
        from .chain import chain as chain_impl
        result = chain_impl(original_g, [ast_obj], EngineAbstract(engine.value), policy=policy, context=context)
    elif isinstance(ast_obj, ASTEdge):
        # ASTEdge operates on the original graph (unless accessed via ASTRef)
        original_g = context.get_binding('__original_graph__') if context.has_binding('__original_graph__') else g
        from .chain import chain as chain_impl
        result = chain_impl(original_g, [ast_obj], EngineAbstract(engine.value), policy=policy, context=context)
    elif isinstance(ast_obj, ASTRemoteGraph):
        # Create a new plottable bound to the remote dataset_id
        # This doesn't fetch the data immediately - it just creates a reference
        result = g.bind(dataset_id=ast_obj.dataset_id)

        # Policy is passed as parameter to execute_node

        # Preload policy phase for remote data loading
        if policy and 'preload' in policy:
            from .gfql.policy import PolicyContext, PolicyException

            preload_context: PolicyContext = {
                'phase': 'preload',
                'hook': 'preload',
                'query': global_query if global_query else ast_obj,  # Global query if available
                'current_ast': ast_obj,
                'query_type': 'dag' if global_query else 'single',
                'is_remote': True,
                'engine': engine.value,
                'execution_depth': context.execution_depth,  # Add execution depth
                '_policy_depth': 0
            }

            try:
                # Policy can only accept (None) or deny (exception)
                policy['preload'](preload_context)
            except PolicyException:
                # Re-raise without modification
                raise

        # If we need to actually fetch the data, we would use chain_remote
        # For now, we'll fetch it immediately to ensure we have the data
        from .chain_remote import chain_remote as chain_remote_impl

        # Fetch the remote dataset with an empty chain (no filtering)
        # Convert engine to the expected type for chain_remote
        chain_engine: Optional[Literal["pandas", "cudf"]] = None
        if engine.value == "pandas":
            chain_engine = "pandas"
        elif engine.value == "cudf":
            chain_engine = "cudf"

        result = chain_remote_impl(
            result,
            [],  # Empty chain - just fetch the entire dataset
            api_token=ast_obj.token,
            dataset_id=ast_obj.dataset_id,
            output_type="all",  # Get full graph (nodes and edges)
            engine=chain_engine
        )

        # Postload policy phase for remote data
        if policy and 'postload' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            stats = extract_graph_stats(result)

            postload_context: PolicyContext = {
                'phase': 'postload',
                'hook': 'postload',
                'query': global_query if global_query else ast_obj,
                'current_ast': ast_obj,
                'query_type': 'dag' if global_query else 'single',
                'is_remote': True,
                'engine': engine.value,
                'graph_stats': stats,
                'execution_depth': context.execution_depth,  # Add execution depth
                '_policy_depth': 0
            }

            try:
                # Policy can only accept (None) or deny (exception)
                policy['postload'](postload_context)
            except PolicyException:
                # Re-raise without modification
                raise
    elif isinstance(ast_obj, ASTCall):
        # Execute method call with validation
        from .gfql.call_executor import execute_call
        result = execute_call(g, ast_obj.function, ast_obj.params, engine, policy=policy, context=context)
    else:
        # Check if it's a Chain or Plottable
        from graphistry.compute.chain import Chain
        if isinstance(ast_obj, Chain):
            # Execute the chain operations
            # For DAG context: Chain should filter from the original graph independently
            # Get the original graph from the context (stored at initialization)
            from .chain import chain as chain_impl
            original_g = context.get_binding('__original_graph__') if context.has_binding('__original_graph__') else g
            result = chain_impl(original_g, ast_obj.chain, EngineAbstract(engine.value), policy=policy, context=context)
        elif isinstance(ast_obj, Plottable):
            # Direct Plottable instance - just return it
            result = ast_obj
        else:
            # Other AST object types not yet implemented
            raise NotImplementedError(f"Execution of {type(ast_obj).__name__} not yet implemented")

    # Store result in context
    context.set_binding(name, result)

    return result


def chain_let_impl(g: Plottable, dag: ASTLet,
                  engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
                  output: Optional[str] = None,
                  policy=None,
                  context: Optional[ExecutionContext] = None) -> Plottable:
    """Internal implementation of chain_let execution

    Validates DAG, determines execution order, and executes nodes
    in topological order.

    :param g: Input graph
    :param dag: Let specification with named bindings
    :param engine: Engine selection (auto/pandas/cudf)
    :param output: Name of binding to return (default: last executed)
    :param policy: Optional policy dictionary
    :param context: Optional ExecutionContext for tracking execution state
    :returns: Result from specified or last executed node
    :rtype: Plottable
    :raises TypeError: If dag is not an ASTLet
    :raises RuntimeError: If node execution fails
    :raises ValueError: If output binding not found
    """
    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    # Validate the let parameter
    if not isinstance(dag, ASTLet):
        raise TypeError(f"dag must be an ASTLet, got {type(dag).__name__}")

    # Validate the let bindings
    dag.validate()

    # Resolve engine
    engine_concrete = resolve_engine(engine, g)
    logger.debug('chain_let engine: %s => %s', engine, engine_concrete)

    # Materialize nodes if needed (following chain.py pattern)
    g = g.materialize_nodes(engine=EngineAbstract(engine_concrete.value))

    # Use provided context or create new one for bindings
    if context is None:
        context = ExecutionContext()

    # Store original graph for independent Chain filtering
    context.set_binding('__original_graph__', g)

    # Handle empty let bindings
    if not dag.bindings:
        return g
    
    # Determine execution order
    order = determine_execution_order(dag.bindings)
    logger.debug("DAG execution order: %s", order)

    # Build dependency graph for binding hooks
    dependencies, dependents = build_dependency_graph(dag.bindings)

    # Initialize variables for finally block
    result = None
    error = None
    success = False
    last_result = None

    try:
        # Prelet hook - fires BEFORE any bindings execute
        if policy and 'prelet' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            stats = extract_graph_stats(g)
            current_path = context.operation_path

            prelet_context: PolicyContext = {
                'phase': 'prelet',
                'hook': 'prelet',
                'query': dag,
                'current_ast': dag,
                'query_type': 'dag',
                'plottable': g,
                'graph_stats': stats,
                'execution_depth': context.execution_depth,
                'operation_path': current_path,
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            try:
                policy['prelet'](prelet_context)
            except PolicyException:
                raise

        # Execute nodes in topological order
        # Start with the original graph and accumulate all binding columns
        accumulated_result = g

        for binding_index, node_name in enumerate(order):
            ast_obj = dag.bindings[node_name]
            logger.debug("Executing node '%s' in DAG", node_name)

            # Preletbinding hook - fires BEFORE binding execution
            if policy and 'preletbinding' in policy:
                from .gfql.policy import PolicyContext, PolicyException

                current_path = context.operation_path
                # Build path that includes this binding (even though we haven't pushed yet)
                binding_path = f"{current_path}.binding:{node_name}"

                preletbinding_context: PolicyContext = {
                    'phase': 'preletbinding',
                    'hook': 'preletbinding',
                    'query': dag,
                    'current_ast': dag,
                    'query_type': 'dag',
                    'binding_name': node_name,
                    'binding_index': binding_index,
                    'total_bindings': len(order),
                    'binding_dependencies': list(dependencies.get(node_name, set())),
                    'binding_ast': ast_obj,
                    'execution_depth': context.execution_depth,  # Add execution depth
                    'operation_path': binding_path,  # Include binding in path
                    'parent_operation': current_path,  # Parent is the DAG level
                    '_policy_depth': 0
                }

                try:
                    policy['preletbinding'](preletbinding_context)
                except PolicyException:
                    raise

            # Execute the node with postletbinding in finally block
            binding_result = None
            binding_error = None
            binding_success = False

            # Push execution depth and operation path for binding execution
            # This moves from depth 1 (let) to depth 2 (binding)
            context.push_depth()
            context.push_path(f"binding:{node_name}")

            try:
                # Execute node - this adds the binding name as a column
                binding_result = execute_node(node_name, ast_obj, accumulated_result, context, engine_concrete, policy, dag)
                binding_success = True

                # Accumulate the new column(s) onto our result
                accumulated_result = binding_result
                result = binding_result

            except Exception as e:
                # Capture binding error
                binding_error = e
                # Don't re-raise yet - let postletbinding fire

            finally:
                # Pop execution depth and operation path before firing postletbinding hook
                context.pop_depth()
                context.pop_path()

                # Postletbinding hook - fires AFTER binding execution (even on error)
                policy_error = None
                if policy and 'postletbinding' in policy:
                    from .gfql.policy import PolicyContext, PolicyException
                    from .gfql.policy.stats import extract_graph_stats

                    # Extract stats from binding result (if success) or current graph (if error)
                    # Cast: if binding_success=True, binding_result is guaranteed to be a Plottable
                    graph_for_stats = cast(Plottable, binding_result) if binding_success else accumulated_result
                    stats = extract_graph_stats(graph_for_stats)

                    current_path = context.operation_path
                    postletbinding_context: PolicyContext = {
                        'phase': 'postletbinding',
                        'hook': 'postletbinding',
                        'query': dag,
                        'current_ast': dag,
                        'query_type': 'dag',
                        'plottable': graph_for_stats,
                        'graph_stats': stats,
                        'binding_name': node_name,
                        'binding_index': binding_index,
                        'total_bindings': len(order),
                        'binding_dependencies': list(dependencies.get(node_name, set())),
                        'binding_ast': ast_obj,
                        'success': binding_success,
                        'execution_depth': context.execution_depth,  # Add execution depth
                        'operation_path': current_path,  # Add operation path
                        'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                        '_policy_depth': 0
                    }

                    # Add error information if binding failed
                    if binding_error is not None:
                        postletbinding_context['error'] = str(binding_error)  # type: ignore
                        postletbinding_context['error_type'] = type(binding_error).__name__  # type: ignore

                    try:
                        policy['postletbinding'](postletbinding_context)
                    except PolicyException as e:
                        # Capture policy error
                        policy_error = e

            # After finally, handle binding errors
            # Priority: PolicyException > binding error
            if policy_error is not None:
                if binding_error is not None:
                    raise policy_error from binding_error
                else:
                    raise policy_error
            elif binding_error is not None:
                # Wrap in RuntimeError with context
                raise RuntimeError(
                    f"Failed to execute node '{node_name}' in DAG. "
                    f"Error: {type(binding_error).__name__}: {str(binding_error)}"
                ) from binding_error

        last_result = accumulated_result

        # Return requested output or last executed result
        if output is not None:
            if output not in context.get_all_bindings():
                # Filter out internal bindings from the error message
                available = sorted([
                    k for k in context.get_all_bindings().keys()
                    if not k.startswith('__')
                ])
                raise ValueError(
                    f"Output binding '{output}' not found. "
                    f"Available bindings: {available}"
                )
            result = context.get_binding(output)
        else:
            result = last_result

        # Mark as successful
        success = True

    except Exception as e:
        # Capture error for postload hook
        error = e
        # Don't re-raise yet - let finally block run first

    finally:
        # Postlet hook - fires AFTER all bindings complete (even on error)
        postlet_policy_error = None
        if policy and 'postlet' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, result is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, result) if success else g
            stats = extract_graph_stats(graph_for_stats)
            current_path = context.operation_path

            postlet_context: PolicyContext = {
                'phase': 'postlet',
                'hook': 'postlet',
                'query': dag,
                'current_ast': dag,
                'query_type': 'dag',
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
                postlet_context['error'] = str(error)  # type: ignore
                postlet_context['error_type'] = type(error).__name__  # type: ignore

            try:
                policy['postlet'](postlet_context)
            except PolicyException as e:
                # Capture policy error instead of raising immediately
                postlet_policy_error = e

        # Postload policy phase - ALWAYS fires (even on error)
        policy_error = None
        if policy and 'postload' in policy:
            from .gfql.policy import PolicyContext, PolicyException
            from .gfql.policy.stats import extract_graph_stats

            # Extract stats from result (if success) or input graph (if error)
            # Cast: if success=True, result is guaranteed to be a Plottable
            graph_for_stats = cast(Plottable, result) if success else g
            stats = extract_graph_stats(graph_for_stats)

            context_dict: PolicyContext = {
                'phase': 'postload',
                'hook': 'postload',
                'query': dag,
                'current_ast': dag,  # For DAG postload, current == dag
                'query_type': 'dag',
                'plottable': graph_for_stats,  # RESULT graph (if success) or INPUT graph (if error)
                'graph_stats': stats,
                'success': success,  # True if successful, False if error
                'execution_depth': context.execution_depth,  # Add execution depth
                '_policy_depth': 0  # Will be handled by thread-local in gfql_unified
            }

            # Add error information if execution failed
            if error is not None:
                context_dict['error'] = str(error)  # type: ignore
                context_dict['error_type'] = type(error).__name__  # type: ignore

            try:
                # Policy can only accept (None) or deny (exception)
                policy['postload'](context_dict)

            except PolicyException as e:
                # Enrich exception with context if not already set
                if e.query_type is None:
                    e.query_type = 'dag'
                if e.data_size is None:
                    e.data_size = stats
                # Capture policy error instead of raising immediately
                policy_error = e

    # After finally block, decide which error to raise
    # Priority: postlet PolicyException > postload PolicyException > operation error
    if postlet_policy_error is not None:
        # postlet policy error takes highest priority
        if error is not None:
            raise postlet_policy_error from error
        else:
            raise postlet_policy_error
    elif policy_error is not None:
        # postload policy error is second priority
        if error is not None:
            raise policy_error from error
        else:
            raise policy_error
    elif error is not None:
        raise error

    # Ensure output matches requested engine (defensive coercion)
    # Schema-changing operations (UMAP, hypergraph) may alter DataFrame types
    if result is not None:
        result = ensure_engine_match(result, engine_concrete)

    # Cast: At this point, all error paths have been handled, so result is guaranteed to be a Plottable
    return cast(Plottable, result)


def chain_let(self: Plottable, dag: ASTLet,
             engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
             output: Optional[str] = None,
             policy=None,
             context: Optional[ExecutionContext] = None) -> Plottable:
    """
    Execute a DAG of named graph operations with dependency resolution
    
    Chain operations can reference results from other operations by name,
    enabling parallel branches and complex data flows.
    
    :param dag: ASTLet containing named bindings of operations
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: Name of binding to return (default: last executed)
    :returns: Plottable result from the specified or last operation
    :rtype: Plottable
    
    **Example: Single operation (no dependencies)**
    
    ::
    
        from graphistry.compute.ast import ASTLet, n
        
        dag = ASTLet({
            'people': n({'type': 'person'})
        })
        result = g.chain_let(dag)
    
    **Example: Linear dependencies**
    
    ::
    
        from graphistry.compute.ast import ASTLet, ASTRef, n, e
        
        dag = ASTLet({
            'start': n({'type': 'person'}),
            'friends': ASTRef('start', [e(), n()])
        })
        result = g.chain_let(dag)
    
    **Example: Diamond pattern**
    
    ::
    
        dag = ASTLet({
            'people': n({'type': 'person'}),
            'transactions': n({'type': 'transaction'}),
            'branch1': ASTRef('people', [e()]),
            'branch2': ASTRef('transactions', [e()]), 
            'merged': g.union(ASTRef('branch1'), ASTRef('branch2'))
        })
        result = g.chain_let(dag)  # Returns last executed
        
        # Or select specific output
        people_result = g.chain_let(dag, output='people')
    """
    return chain_let_impl(self, dag, engine, output, policy, context)
