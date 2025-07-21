from typing import Dict, Set, List, Optional, Tuple, Union, cast
from typing_extensions import Literal
from graphistry.Engine import Engine, EngineAbstract, resolve_engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTChainRef, ASTRemoteGraph, ASTNode, ASTEdge, ASTCall
from .execution_context import ExecutionContext

logger = setup_logger(__name__)


def extract_dependencies(ast_obj: ASTObject) -> Set[str]:
    """Recursively find all ASTChainRef references in an AST object
    
    :param ast_obj: AST object to analyze
    :returns: Set of referenced binding names
    :rtype: Set[str]
    """
    deps = set()
    
    if isinstance(ast_obj, ASTChainRef):
        deps.add(ast_obj.ref)
        # Also check chain operations
        for op in ast_obj.chain:
            deps.update(extract_dependencies(op))
    
    elif isinstance(ast_obj, ASTLet):
        # Nested let bindings
        for binding in ast_obj.bindings.values():
            deps.update(extract_dependencies(binding))
    
    # Other AST types (ASTNode, ASTEdge, ASTRemoteGraph) have no dependencies
    return deps


def build_dependency_graph(bindings: Dict[str, ASTObject]) -> Tuple[Dict[str, Set[str]], Dict[str, Set[str]]]:
    """Build dependency and dependent mappings from bindings
    
    :param bindings: Dictionary of name -> AST object bindings
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


def validate_dependencies(bindings: Dict[str, ASTObject], 
                        dependencies: Dict[str, Set[str]]) -> None:
    """Check for missing references and self-cycles
    
    :param bindings: Dictionary of available bindings
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


def topological_sort(bindings: Dict[str, ASTObject],
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


def determine_execution_order(bindings: Dict[str, ASTObject]) -> List[str]:
    """Determine topological execution order for DAG bindings
    
    Validates dependencies and computes execution order that respects
    all dependencies. Detects cycles and missing references.
    
    :param bindings: Dictionary of name -> AST object bindings
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


def execute_node(name: str, ast_obj: ASTObject, g: Plottable, 
                context: ExecutionContext, engine: Engine) -> Plottable:
    """Execute a single node in the DAG
    
    Handles different AST object types:
    - ASTLet: Recursive let execution
    - ASTChainRef: Reference resolution and chain execution
    - ASTNode: Node filtering operations
    - ASTEdge: Edge traversal operations
    - Others: NotImplementedError
    
    :param name: Binding name for this node
    :param ast_obj: AST object to execute
    :param g: Input graph
    :param context: Execution context for storing/retrieving results
    :param engine: Engine to use (pandas/cudf)
    :returns: Resulting Plottable
    :rtype: Plottable
    :raises ValueError: If reference not found in context
    :raises NotImplementedError: For unsupported AST types
    """
    logger.debug("Executing node '%s' of type %s", name, type(ast_obj).__name__)
    
    # Handle different AST object types
    if isinstance(ast_obj, ASTLet):
        # Nested let execution
        result = chain_dag_impl(g, ast_obj, engine.value)
    elif isinstance(ast_obj, ASTChainRef):
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
            result = chain_impl(referenced_result, ast_obj.chain, engine.value)
        else:
            # Empty chain - just return the referenced result
            result = referenced_result
    elif isinstance(ast_obj, ASTNode):
        # For chain_dag, we execute nodes in a simpler way than chain()
        # No wavefront propagation - just filter the graph's nodes
        node_obj = cast(ASTNode, ast_obj)
        if node_obj.filter_dict or node_obj.query:
            filtered_g = g
            if node_obj.filter_dict:
                filtered_g = filtered_g.filter_nodes_by_dict(node_obj.filter_dict)
            if node_obj.query:
                filtered_g = filtered_g.nodes(lambda g: g._nodes.query(node_obj.query))
            result = filtered_g
        else:
            # Empty filter - return original graph
            result = g
        
        # Add name column if specified
        if node_obj._name:
            result = result.nodes(result._nodes.assign(**{node_obj._name: True}))
    elif isinstance(ast_obj, ASTEdge):
        # For chain_dag, execute edge operations using hop()
        # This is simpler than the full chain() wavefront approach
        result = g.hop(
            nodes=None,  # Start from all nodes
            hops=ast_obj.hops,
            to_fixed_point=ast_obj.to_fixed_point,
            direction=ast_obj.direction,
            source_node_match=ast_obj.source_node_match,
            edge_match=ast_obj.edge_match,
            destination_node_match=ast_obj.destination_node_match,
            source_node_query=ast_obj.source_node_query,
            edge_query=ast_obj.edge_query,
            destination_node_query=ast_obj.destination_node_query,
            return_as_wave_front=False  # Return full graph
        )
        
        # Add name column to edges if specified
        if ast_obj._name:
            result = result.edges(result._edges.assign(**{ast_obj._name: True}))
    elif isinstance(ast_obj, ASTRemoteGraph):
        # Create a new plottable bound to the remote dataset_id
        # This doesn't fetch the data immediately - it just creates a reference
        result = g.bind(dataset_id=ast_obj.dataset_id)
        
        # If we need to actually fetch the data, we would use chain_remote
        # For now, we'll fetch it immediately to ensure we have the data
        from .chain_remote import chain_remote as chain_remote_impl
        
        # Fetch the remote dataset with an empty chain (no filtering)
        result = chain_remote_impl(
            result,
            [],  # Empty chain - just fetch the entire dataset
            api_token=ast_obj.token,
            dataset_id=ast_obj.dataset_id,
            output_type="all",  # Get full graph (nodes and edges)
            engine=cast(Literal["pandas", "cudf"], engine.value)
        )
    elif isinstance(ast_obj, ASTCall):
        # Execute method call with validation
        from .gfql.call_executor import execute_call
        result = execute_call(g, ast_obj.function, ast_obj.params, engine)
    else:
        # Other AST object types not yet implemented
        raise NotImplementedError(f"Execution of {type(ast_obj).__name__} not yet implemented")
    
    # Store result in context
    context.set_binding(name, result)
    
    return result


def chain_dag_impl(g: Plottable, dag: ASTLet, 
                  engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
                  output: Optional[str] = None) -> Plottable:
    """Internal implementation of chain_dag execution
    
    Validates DAG, determines execution order, and executes nodes
    in topological order.
    
    :param g: Input graph
    :param dag: Let specification with named bindings
    :param engine: Engine selection (auto/pandas/cudf)
    :param output: Name of binding to return (default: last executed)
    :returns: Result from specified or last executed node
    :rtype: Plottable
    :raises TypeError: If dag is not an ASTQueryDAG
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
    logger.debug('chain_dag engine: %s => %s', engine, engine_concrete)
    
    # Materialize nodes if needed (following chain.py pattern)
    g = g.materialize_nodes(engine=EngineAbstract(engine_concrete.value))
    
    # Create execution context
    context = ExecutionContext()
    
    # Handle empty let bindings
    if not dag.bindings:
        return g
    
    # Determine execution order
    order = determine_execution_order(dag.bindings)
    logger.debug("DAG execution order: %s", order)
    
    # Execute nodes in topological order
    last_result = g
    for node_name in order:
        ast_obj = dag.bindings[node_name]
        logger.debug("Executing node '%s' in DAG", node_name)
        
        # Execute the node and store result in context
        try:
            result = execute_node(node_name, ast_obj, g, context, engine_concrete)
            last_result = result
        except Exception as e:
            # Add context to error
            raise RuntimeError(
                f"Failed to execute node '{node_name}' in DAG. "
                f"Error: {type(e).__name__}: {str(e)}"
            ) from e
    
    # Return requested output or last executed result
    if output is not None:
        if output not in context.get_all_bindings():
            available = sorted(context.get_all_bindings().keys())
            raise ValueError(
                f"Output binding '{output}' not found. "
                f"Available bindings: {available}"
            )
        return context.get_binding(output)
    else:
        return last_result


def chain_dag(self: Plottable, dag: ASTLet,
             engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
             output: Optional[str] = None) -> Plottable:
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
        result = g.chain_dag(dag)
    
    **Example: Linear dependencies**
    
    ::
    
        from graphistry.compute.ast import ASTLet, ASTChainRef, n, e
        
        dag = ASTLet({
            'start': n({'type': 'person'}),
            'friends': ASTChainRef('start', [e(), n()])
        })
        result = g.chain_dag(dag)
    
    **Example: Diamond pattern**
    
    ::
    
        dag = ASTLet({
            'people': n({'type': 'person'}),
            'transactions': n({'type': 'transaction'}),
            'branch1': ASTChainRef('people', [e()]),
            'branch2': ASTChainRef('transactions', [e()]), 
            'merged': g.union(ASTChainRef('branch1'), ASTChainRef('branch2'))
        })
        result = g.chain_dag(dag)  # Returns last executed
        
        # Or select specific output
        people_result = g.chain_dag(dag, output='people')
    """
    return chain_dag_impl(self, dag, engine, output)
