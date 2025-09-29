"""Type definitions for GFQL policy system."""

from typing import TypedDict, Optional, Dict, Any, Literal, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.gfql.policy.stats import GraphStats

# Phase literal type
Phase = Literal["preload", "postload", "call"]

# Query type literal
QueryType = Literal["chain", "dag", "single"]


class PolicyContext(TypedDict, total=False):
    """Strongly typed context passed to policy functions.

    Attributes:
        phase: Current execution phase (preload, postload, call)
        hook: Hook name (same as phase, useful for shared handlers)
        query: Original/global query object
        current_ast: Current AST object being executed (if applicable)
        query_type: Type of query (chain, dag, single)
        plottable: Plottable instance (postload/call phases)
        call_op: Call operation name (call phase only)
        call_params: Call parameters (call phase only)
        graph_stats: Graph statistics (nodes, edges, memory)
        is_remote: True for remote/network operations
        engine: Engine being used (pandas, cudf, etc.)
        _policy_depth: Internal recursion prevention counter
    """

    phase: Phase
    hook: Phase  # Same as phase, for shared handler convenience
    query: Any  # Global/original query
    current_ast: Optional[Any]  # Current sub-AST being executed
    query_type: QueryType
    plottable: Optional['Plottable']
    call_op: Optional[str]
    call_params: Optional[Dict[str, Any]]
    graph_stats: Optional['GraphStats']
    is_remote: Optional[bool]
    engine: Optional[str]
    _policy_depth: int


# Type alias for policy functions
# Policies can only accept (return None) or deny (raise PolicyException)
PolicyFunction = Callable[[PolicyContext], None]

# Type alias for policy dictionary
PolicyDict = Dict[Phase, PolicyFunction]
