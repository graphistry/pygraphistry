"""Execution context for DAG operations."""
from typing import Any, Dict


class ExecutionContext:
    """Manages variable bindings during DAG execution.
    
    Provides a namespace for storing and retrieving named graph results
    during the execution of ASTLet DAGs. Each binding maps a string name
    to a Plottable graph instance.
    
    **Example::**
    
        context = ExecutionContext()
        context.set_binding('persons', person_graph)
        friends = context.get_binding('persons')
    """
    
    def __init__(self) -> None:
        """Initialize an empty execution context."""
        self._bindings: Dict[str, Any] = {}
    
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
