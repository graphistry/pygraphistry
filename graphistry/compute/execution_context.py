"""Execution context for DAG operations and policy tracking."""
from typing import Any, Dict


class ExecutionContext:
    """Manages variable bindings and execution state during GFQL execution.

    Provides a namespace for:
    - Named graph result bindings during ASTLet DAG execution
    - Execution depth tracking for policy hooks (0=query, 1=let/chain, 2=binding, 3=call)
    - Operation path tracking for OpenTelemetry-style tracing
    - Policy recursion prevention

    **Example: DAG bindings**::

        context = ExecutionContext()
        context.set_binding('persons', person_graph)
        friends = context.get_binding('persons')

    **Example: Execution tracking**::

        context = ExecutionContext()
        context.push_depth()  # 0 → 1
        context.push_path('dag')  # 'query' → 'query.dag'
        depth = context.execution_depth  # 1
        path = context.operation_path  # 'query.dag'
        context.pop_path()  # 'query.dag' → 'query'
        context.pop_depth()  # 1 → 0
    """

    def __init__(self) -> None:
        """Initialize an empty execution context."""
        self._bindings: Dict[str, Any] = {}
        self.execution_depth: int = 0
        self.operation_path: str = 'query'
        self.policy_depth: int = 0
    
    def set_binding(self, name: str, value: Any) -> None:
        """Store a named result in the context.
        
        :param name: Name for the binding
        :type name: str
        :param value: Value to bind (typically a Plottable)
        :type value: Any
        :raises TypeError: If name is not a string
        """
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        self._bindings[name] = value
    
    def get_binding(self, name: str) -> Any:
        """Retrieve a named result from the context.
        
        :param name: Name of the binding to retrieve
        :type name: str
        :returns: The bound value
        :rtype: Any
        :raises TypeError: If name is not a string
        :raises KeyError: If no binding exists for the given name
        """
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        if name not in self._bindings:
            raise KeyError(f"No binding found for '{name}'")
        return self._bindings[name]
    
    def has_binding(self, name: str) -> bool:
        """Check if a binding exists in the context.
        
        :param name: Name to check
        :type name: str
        :returns: True if binding exists, False otherwise
        :rtype: bool
        :raises TypeError: If name is not a string
        """
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        return name in self._bindings
    
    def clear(self) -> None:
        """Clear all bindings from the context."""
        self._bindings.clear()
    
    def get_all_bindings(self) -> Dict[str, Any]:
        """Get a copy of all bindings in the context.

        :returns: Dictionary of all current bindings
        :rtype: Dict[str, Any]
        """
        return self._bindings.copy()

    # Execution depth tracking methods

    def push_depth(self) -> int:
        """Increment execution depth and return new depth.

        :returns: New execution depth after increment
        :rtype: int
        """
        self.execution_depth += 1
        return self.execution_depth

    def pop_depth(self) -> int:
        """Decrement execution depth and return new depth.

        :returns: New execution depth after decrement
        :rtype: int
        """
        self.execution_depth = max(0, self.execution_depth - 1)
        return self.execution_depth

    # Operation path tracking methods

    def push_path(self, segment: str) -> str:
        """Append a segment to the operation path and return new path.

        :param segment: Path segment to append (e.g., "dag", "binding:people", "call:hypergraph")
        :type segment: str
        :returns: New operation path after appending segment
        :rtype: str
        """
        self.operation_path = f"{self.operation_path}.{segment}"
        return self.operation_path

    def pop_path(self) -> str:
        """Remove last segment from operation path and return new path.

        :returns: New operation path after removing last segment
        :rtype: str
        """
        if '.' in self.operation_path:
            self.operation_path = self.operation_path.rsplit('.', 1)[0]
        else:
            self.operation_path = 'query'  # Reset to root
        return self.operation_path

    def get_parent_operation(self) -> str:
        """Get the parent operation path for the current operation.

        :returns: Parent operation path
        :rtype: str
        """
        if '.' in self.operation_path:
            return self.operation_path.rsplit('.', 1)[0]
        else:
            return 'query'
