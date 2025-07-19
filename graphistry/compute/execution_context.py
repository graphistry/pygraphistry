"""Execution context for DAG operations"""
from typing import Any, Dict


class ExecutionContext:
    """Manages variable bindings during DAG execution"""
    
    def __init__(self):
        self._bindings: Dict[str, Any] = {}
    
    def set_binding(self, name: str, value: Any) -> None:
        """Store a named result"""
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        self._bindings[name] = value
    
    def get_binding(self, name: str) -> Any:
        """Retrieve a named result"""
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        if name not in self._bindings:
            raise KeyError(f"No binding found for '{name}'")
        return self._bindings[name]
    
    def has_binding(self, name: str) -> bool:
        """Check if binding exists"""
        if not isinstance(name, str):
            raise TypeError(f"Binding name must be string, got {type(name)}")
        return name in self._bindings
    
    def clear(self) -> None:
        """Clear all bindings"""
        self._bindings.clear()
    
    def get_all_bindings(self) -> Dict[str, Any]:
        """Get a copy of all bindings"""
        return self._bindings.copy()