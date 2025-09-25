"""GraphOperation type definition for let() bindings.

GraphOperation represents types that can be bound in let() statements -
operations that produce or reference Plottable objects.
"""
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.chain import Chain
    from graphistry.compute.ast import (
        ASTRef, ASTCall, ASTRemoteGraph, ASTLet
    )

# GraphOperation represents values that can be bound in let()
# These are operations that produce Plottable objects
GraphOperation = Union[
    'Plottable',        # Direct graph instances
    'Chain',            # Chain operations
    'ASTRef',           # References to other bindings
    'ASTCall',          # Method calls on graphs
    'ASTRemoteGraph',   # Remote graph references
    'ASTLet',           # Nested let bindings
]
