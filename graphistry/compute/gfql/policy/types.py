"""Type definitions for GFQL policy system."""

from typing import List, TypedDict, Optional, Dict, Any, Literal, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from graphistry.Plottable import Plottable
    from graphistry.compute.gfql.policy.stats import GraphStats

# Phase literal type (runtime hooks plus opt-in compiler hooks)
Phase = Literal["preload", "postload", "prelet", "postlet", "prechain", "postchain", "precall", "postcall", "preletbinding", "postletbinding", "postcompile"]

# Shortcut key literal type (includes general, scope, and specific keys)
GeneralShortcut = Literal["pre", "post"]
ScopeShortcut = Literal["load", "let", "chain", "binding", "call"]
ShortcutKey = Literal["pre", "post", "load", "let", "chain", "binding", "call", "preload", "postload", "prelet", "postlet", "prechain", "postchain", "precall", "postcall", "preletbinding", "postletbinding", "postcompile"]

# Query type literal
QueryType = Literal["chain", "dag", "single"]


class CompilerAliasSummary(TypedDict, total=False):
    """Stable alias-binding summary emitted by compiler policy hooks."""

    name: str
    kind: Literal["node", "edge", "scalar"]
    nullable: bool


class CompilerProjectionSummary(TypedDict, total=False):
    """Stable projection item summary emitted by compiler policy hooks."""

    clause: Literal["with", "return"]
    output: str
    expr: str
    expr_kind: Literal["alias", "property", "aggregate", "expr", "wildcard"]
    source: Optional[str]
    property: Optional[str]
    entity_kind: Optional[Literal["node", "edge", "scalar"]]


class CompilerAggregateSummary(TypedDict, total=False):
    """Stable aggregate summary emitted by compiler policy hooks."""

    clause: Literal["with", "return"]
    output: str
    fn: str
    input: str
    distinct: bool


class CompilerPolicySummary(TypedDict, total=False):
    """Stable Cypher compiler metadata for opt-in policy hooks."""

    phase: Literal["postcompile"]
    language: Literal["cypher"]
    query_type: QueryType
    query_hash: str
    aliases: List[CompilerAliasSummary]
    projections: List[CompilerProjectionSummary]
    group_keys: List[str]
    aggregates: List[CompilerAggregateSummary]
    param_keys: List[str]


class PolicyContext(TypedDict, total=False):
    """Strongly typed context passed to policy functions.

    Attributes:
        phase: Current execution phase (preload, postload, prelet, postlet, prechain, postchain, precall, postcall, preletbinding, postletbinding)
        hook: Hook name (same as phase, useful for shared handlers)
        query: Original/global query object
        current_ast: Current AST object being executed (if applicable)
        query_type: Type of query (chain, dag, single)
        plottable: Plottable instance (postload/precall/postcall phases)
                  - precall: INPUT graph
                  - postcall: RESULT graph (if success) or INPUT graph (if error)
        call_op: Call operation name (precall/postcall phases only)
        call_params: Call parameters (precall/postcall phases only)
        graph_stats: Graph statistics (nodes, edges, memory)
                    - precall: INPUT graph stats
                    - postcall: RESULT graph stats (if success) or INPUT stats (if error)
        is_remote: True for remote/network operations
        engine: Engine being used (pandas, cudf, etc.)
        execution_time: Method execution duration (postcall phase only)
        success: Execution success flag (postcall/postload/postletbinding phases)
        error: Error message string (post* phases, when success=False)
        error_type: Error type name (post* phases, when success=False)

        # Hierarchy/tracing fields (for OpenTelemetry and telemetry systems)
        execution_depth: Nesting depth (0=query, 1=let/chain, 2=binding/op, ...)
        operation_path: Unique path like "query.let.binding:hg.call:hypergraph"
        parent_operation: Parent operation path (for span parent relationships)

        # Binding-specific fields (preletbinding/postletbinding phases only)
        binding_name: Name of the current binding being executed
        binding_index: Execution order of this binding (0-indexed)
        total_bindings: Total number of bindings in the let expression
        binding_dependencies: List of binding names this binding depends on
        binding_ast: The AST object being bound (the value in let({name: ast}))

        _policy_depth: Internal recursion prevention counter

        # Compiler-specific fields (postcompile phase only)
        language: Query language name
        compiler_summary: Stable semantic compiler summary
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
    execution_time: Optional[float]  # Method execution duration (postcall only)
    success: Optional[bool]          # Execution success flag (post* phases)
    error: Optional[str]             # Error message (post* phases, when success=False)
    error_type: Optional[str]        # Error type name (post* phases, when success=False)

    # Hierarchy/tracing fields (for OpenTelemetry and telemetry systems)
    execution_depth: Optional[int]       # Nesting depth (0=query, 1=let/chain, 2=binding, ...)
    operation_path: Optional[str]        # Unique path like "query.let.binding:hg.call:hypergraph"
    parent_operation: Optional[str]      # Parent operation path (for span parent relationships)

    # Binding-specific fields (preletbinding/postletbinding phases only)
    binding_name: Optional[str]          # Name of the current binding
    binding_index: Optional[int]         # Execution order (0-indexed)
    total_bindings: Optional[int]        # Total bindings in let expression
    binding_dependencies: Optional[list]  # List of binding names this depends on
    binding_ast: Optional[Any]           # The AST object being bound

    # Compiler-specific fields (postcompile phase only)
    language: Optional[str]
    compiler_summary: Optional[CompilerPolicySummary]

    _policy_depth: int


# Type alias for policy functions
# Policies can only accept (return None) or deny (raise PolicyException)
PolicyFunction = Callable[[PolicyContext], None]

# Type alias for policy dictionary
PolicyDict = Dict[Phase, PolicyFunction]
