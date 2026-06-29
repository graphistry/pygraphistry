"""Execute validated method calls on Plottable objects.

This module handles the actual execution of safelisted methods
after parameter validation.
"""

import threading
import time
from typing import Dict, Any, cast, Optional, TYPE_CHECKING, Tuple
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.gfql.call.validation import validate_call_params
from graphistry.compute.gfql.row.pipeline import (
    execute_row_pipeline_call,
    is_row_pipeline_call,
)
from graphistry.compute.exceptions import ErrorCode, GFQLTypeError
from graphistry.compute.engine_coercion import ensure_engine_match
from graphistry.compute.gfql.policy import PolicyContext, PolicyException
from graphistry.compute.gfql.policy.stats import extract_graph_stats
from graphistry.compute.gfql.schema_effects import apply_call_schema_effect

if TYPE_CHECKING:
    from graphistry.compute.execution_context import ExecutionContext
    from graphistry.compute.gfql.policy.stats import GraphStats

# Thread-local storage for policy context
_thread_local = threading.local()


def _is_plottable_like(obj: object) -> bool:
    """Runtime-safe structural check for Plottable-like objects.

    Avoids isinstance(..., Plottable) Protocol checks, which may trigger expensive
    or fragile attribute probing on third-party objects (e.g., pandas DataFrame.style
    optional dependency paths on some Python versions).
    """
    return all(hasattr(obj, attr) for attr in ("_nodes", "_edges", "bind"))


def _run_policy_hook(handler: Any, policy_context: 'PolicyContext', data_size: Optional['GraphStats']) -> Optional[PolicyException]:
    try:
        handler(policy_context)
    except PolicyException as e:
        if e.query_type is None:
            e.query_type = 'call'
        if e.data_size is None:
            e.data_size = data_size
        return e
    return None


def _active_frames_are_polars(g: Plottable) -> bool:
    """True when the graph's active frames are polars (the polars-engine path).

    Mirrors ``execute_row_pipeline_call``'s frame-type probe: under engine='polars'/
    'polars-gpu' the frames are polars; pandas/cuDF inputs were coerced upstream.
    """
    nodes = getattr(g, "_nodes", None)
    if nodes is not None:
        return "polars" in type(nodes).__module__
    edges = getattr(g, "_edges", None)
    return edges is not None and "polars" in type(edges).__module__


def _execute_validated_call(g: Plottable, function: str, validated_params: Dict[str, Any]) -> Any:
    if is_row_pipeline_call(function):
        return execute_row_pipeline_call(g, function, validated_params)

    # NATIVE polars get_degrees: pure groupby/count over edge endpoints — NO pandas
    # bridge (see NO-CHEATING). Reached by the let()/ref() DAG surface (and the
    # schema-changer chain path); the native chain surface routes the same op through
    # engine_polars.chain._try_native_row_op. The result is polars, so it passes the
    # no-bridge guard in execute_call (and ensure_engine_match is then a no-op).
    # Other Plottable-method calls have no native polars impl and stay declined by
    # that guard.
    if function == "get_degrees" and _active_frames_are_polars(g):
        from graphistry.compute.gfql.engine_polars.chain import get_degrees_polars
        return get_degrees_polars(g, **validated_params)

    if not hasattr(g, function):
        raise AttributeError(
            f"Plottable has no method '{function}'. "
            f"This should not happen if safelist is properly configured."
        )

    method = getattr(g, function)
    return method(**validated_params)


def _postcall_graph_and_stats(
    g: Plottable,
    result: Any,
    success: bool
) -> Tuple[Plottable, Optional['GraphStats']]:
    if success and _is_plottable_like(result):
        graph_for_stats = cast(Plottable, result)
        return graph_for_stats, extract_graph_stats(graph_for_stats)
    if success:
        return g, None
    return g, extract_graph_stats(g)


def execute_call(g: Plottable, function: str, params: Dict[str, Any], engine: Engine, policy=None, context: Optional['ExecutionContext'] = None) -> Plottable:
    """Execute a validated method call on a Plottable."""
    if context is None:
        from graphistry.compute.execution_context import ExecutionContext
        context = ExecutionContext()

    if policy is None:
        policy = getattr(_thread_local, 'policy', None)

    final_params = params

    if policy and 'precall' in policy:
        stats = extract_graph_stats(g)

        current_path = context.operation_path
        # Build path that includes this call (even though we haven't pushed yet)
        call_path = f"{current_path}.call:{function}"

        policy_context: 'PolicyContext' = {
            'phase': 'precall',
            'hook': 'precall',
            'query': None,  # Not available in call context
            'current_ast': None,  # Calls don't have AST object
            'call_op': function,
            'call_params': params,
            'plottable': g,  # INPUT graph
            'graph_stats': stats,  # INPUT stats
            'execution_depth': context.execution_depth,  # Add execution depth
            'operation_path': call_path,  # Include call in path
            'parent_operation': current_path,  # Parent is the current level
            '_policy_depth': 0
        }

        policy_error = _run_policy_hook(policy['precall'], policy_context, stats)
        if policy_error is not None:
            raise policy_error

    result = None
    error = None
    success = False
    execution_time = 0.0
    start_time = time.perf_counter()
    validated_params = None
    hypergraph_returns_dataframe = False

    context.push_depth()
    context.push_path(f"call:{function}")

    try:
        validated_params = validate_call_params(function, final_params)
        if function == 'hypergraph':
            hypergraph_returns_dataframe = validated_params.get('return_as', 'graph') != 'graph'

        result = _execute_validated_call(g, function, validated_params)
        execution_time = time.perf_counter() - start_time

        # Ensure result is a Plottable (most methods return self or new Plottable)
        # Exception: hypergraph can return DataFrame when return_as != 'graph'
        if not hypergraph_returns_dataframe and not _is_plottable_like(result):
            raise GFQLTypeError(
                ErrorCode.E201,
                f"Method '{function}' returned non-Plottable result",
                field="function",
                value=f"{type(result).__name__}",
                suggestion="Only methods that return Plottable objects are allowed"
            )

        # Ensure result matches requested engine (defensive coercion)
        # Schema-changing operations (UMAP, hypergraph) may alter DataFrame types
        if _is_plottable_like(result):
            # NO-CHEATING: a Plottable-method call (get_degrees / hypergraph / umap / igraph /
            # cugraph / nx algos) has no native polars implementation — it runs on pandas/cuDF.
            # Coercing that result back to polars here would be a SILENT polars-engine bridge, and
            # is inconsistent with the chain surface, which honestly raises NotImplementedError for
            # the identical op. So under a polars engine, if the result frames are not already
            # polars, decline instead of bridging. (Native-polars row-pipeline calls — select etc. —
            # produce polars frames and pass through unchanged; pandas/cuDF engines are unaffected.)
            from graphistry.Engine import Engine as _Eng
            if engine in (_Eng.POLARS, _Eng.POLARS_GPU):
                _res_nodes = getattr(result, '_nodes', None)
                _res_edges = getattr(result, '_edges', None)
                _probe = _res_nodes if _res_nodes is not None else _res_edges
                if _probe is not None and 'polars' not in type(_probe).__module__:
                    raise NotImplementedError(
                        f"GFQL engine='{engine.value}' does not natively support call "
                        f"'{function}'; it runs on pandas/cuDF and coercing the result back to "
                        "polars would be a silent bridge. Use engine='pandas'."
                    )
            result = ensure_engine_match(cast(Plottable, result), engine)
            result = apply_call_schema_effect(g, cast(Plottable, result), function, validated_params)

        success = True

    except Exception as e:
        execution_time = time.perf_counter() - start_time
        error = e

    finally:
        context.pop_depth()
        context.pop_path()

        # Postcall policy phase - ALWAYS fires (even on error)
        policy_error = None
        if policy and 'postcall' in policy:
            graph_for_stats, result_stats = _postcall_graph_and_stats(g, result, success)

            current_path = context.operation_path
            postcall_context: 'PolicyContext' = {
                'phase': 'postcall',
                'hook': 'postcall',
                'query': None,  # Not available in call context
                'current_ast': None,  # Calls don't have AST object
                'call_op': function,
                'call_params': params,  # Original parameters for reference
                'plottable': graph_for_stats,  # RESULT graph (if success) or INPUT graph (if error)
                'graph_stats': result_stats,
                'execution_time': execution_time,
                'success': success,  # True if successful, False if error
                'execution_depth': context.execution_depth,  # Add execution depth
                'operation_path': current_path,  # Add operation path
                'parent_operation': current_path.rsplit('.', 1)[0] if '.' in current_path else 'query',
                '_policy_depth': 0
            }

            # Add error information if execution failed
            if error is not None:
                postcall_context['error'] = str(error)  # type: ignore
                postcall_context['error_type'] = type(error).__name__  # type: ignore

            policy_error = _run_policy_hook(policy['postcall'], postcall_context, result_stats)

    if policy_error is not None:
        if error is not None:
            raise policy_error from error
        raise policy_error
    if isinstance(error, TypeError):
        raise GFQLTypeError(
            ErrorCode.E201,
            f"Parameter error calling '{function}': {str(error)}",
            field="params",
            value=validated_params if validated_params is not None else params,
            suggestion="Check parameter names and types"
        ) from error
    if isinstance(error, GFQLTypeError):
        raise error
    if isinstance(error, NotImplementedError):
        # Honest engine-capability decline (e.g. the polars no-silent-bridge guard above) —
        # propagate as-is so the DAG surface matches the chain surface's NotImplementedError.
        raise error
    if error is not None:
        raise GFQLTypeError(
            ErrorCode.E303,
            f"Error executing '{function}': {str(error)}",
            field="function",
            value=function
        ) from error

    # Cast: At this point, all error paths have been handled, so result is guaranteed to be a Plottable
    return cast(Plottable, result)
