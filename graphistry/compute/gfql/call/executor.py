"""Execute validated method calls on Plottable objects.

This module handles the actual execution of safelisted methods
after parameter validation.
"""

import threading
import time
import warnings
from typing import Dict, Any, cast, Optional, TYPE_CHECKING, Set, Tuple
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine
from graphistry.compute.gfql.lazy import call_mode
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


from graphistry.Engine import active_frames_are_polars as _active_frames_are_polars

# Off-engine call() modality bridge (PHASE 12). A GFQL call() that runs a Plottable-method
# ANALYTIC (umap / hypergraph / compute_cugraph / compute_igraph / layout_* / collapse / ...)
# has NO native polars impl and never will — it runs eagerly on pandas/cuDF. Under a polars
# engine, call_mode()='auto' (default) BRIDGES: run the analytic off-engine on pandas (polars)
# / cuDF (polars-gpu), then coerce the result back to polars losslessly (Arrow). call_mode()=
# 'strict' DECLINES (parity-or-NIE) for benchmark integrity / a hard memory ceiling. This is
# DELIBERATELY narrower than CHAIN traversal/filter/row ops, which stay parity-or-NIE (a bridge
# there hides a missing impl + cheats a benchmark). The distinction is MECHANICAL, not a curated
# list: row-pipeline calls (is_row_pipeline_call) and native-polars degree calls are dispatched
# BEFORE the analytic path in _execute_validated_call, so everything reaching the bridge is by
# construction a non-native eager analytic. Mirrors the GRAPHISTRY_CUDF_SAME_PATH_MODE auto/strict
# precedent. (Follow-ups tracked in plan PHASE 12: G3 otel attribution, G4 queryable flag, G5 size guard.)
_OFFENGINE_BRIDGE_WARNED: Set[str] = set()


def _compute_engine_for_offengine_call(engine: Engine, function: str) -> Engine:
    """Modality an off-engine analytic runs on under a polars engine.

    ``polars`` -> pandas; ``polars-gpu`` -> cuDF (on-device). polars-gpu is GPU-or-error:
    if cuDF is unavailable we do NOT silently drop a GPU analytic to host pandas (which
    would break the engine contract and risk a unified-memory OOM) — we decline.
    """
    if engine == Engine.POLARS_GPU:
        try:
            import cudf  # type: ignore  # noqa: F401
        except Exception as e:  # cuDF/GPU stack missing
            raise NotImplementedError(
                f"GFQL engine='polars-gpu' call '{function}' runs on cuDF (on device), but "
                f"cuDF is unavailable ({type(e).__name__}). polars-gpu does not silently fall "
                "back to host pandas. Install the cuDF/GPU stack, or use engine='pandas'."
            )
        return Engine.CUDF
    return Engine.PANDAS


def _bridge_graph_for_offengine_call(g: Plottable, function: str, engine: Engine) -> Plottable:
    """Bridge ``g`` to the compute modality for an off-engine analytic, or decline (strict).

    ``call_mode()='strict'`` raises ``NotImplementedError`` (honest decline). ``'auto'`` (default)
    converts ``g``'s frames polars->pandas (or polars->cuDF for polars-gpu) so the analytic gets
    the frame type it expects, warns ONCE per (process, function), and returns the bridged graph.
    The caller coerces the analytic's result back to polars (``ensure_engine_match``, lossless).
    """
    if call_mode() == "strict":
        raise NotImplementedError(
            f"GFQL engine='{engine.value}' does not natively support call '{function}' "
            "(it runs on pandas/cuDF); call_mode='strict' declines the off-engine bridge. "
            "Use engine='pandas', or set call_mode='auto' (the default) to run it off-engine."
        )
    compute_engine = _compute_engine_for_offengine_call(engine, function)
    # Convert the frames EXPLICITLY via df_to_engine — not ensure_engine_match, whose
    # resolve_engine(AUTO, ...) detection classifies a polars frame as PANDAS (polars isn't a
    # resolve_engine target), so it would treat the polars input as "already pandas" and no-op.
    # df_to_engine is a genuine no-op when the frame is already compute_engine's type.
    from graphistry.Engine import df_to_engine
    bridged = g
    if g._nodes is not None:
        bridged = bridged.nodes(df_to_engine(g._nodes, compute_engine), g._node)
    if g._edges is not None:
        bridged = bridged.edges(
            df_to_engine(g._edges, compute_engine), g._source, g._destination, edge=g._edge
        )
    if function not in _OFFENGINE_BRIDGE_WARNED:
        _OFFENGINE_BRIDGE_WARNED.add(function)
        warnings.warn(
            f"GFQL call '{function}' has no native polars implementation; running it off-engine "
            f"on {compute_engine.value} and coercing the result back to '{engine.value}' "
            "(lossless via Arrow). Set call_mode='strict' to decline instead.",
            RuntimeWarning,
            stacklevel=3,
        )
    return bridged


def _execute_validated_call(g: Plottable, function: str, validated_params: Dict[str, Any], engine: Engine) -> Any:
    if is_row_pipeline_call(function):
        return execute_row_pipeline_call(g, function, validated_params)

    # NATIVE polars degree calls (get_degrees / get_indegrees / get_outdegrees): pure
    # groupby/count over edge endpoints — NO pandas bridge (see NO-CHEATING). Reached by
    # the let()/ref() DAG surface (and the schema-changer chain path); the native chain
    # surface routes the same ops through polars.chain._try_native_row_op. The result is
    # polars, so it passes the no-bridge guard in execute_call (and ensure_engine_match is
    # then a no-op). Other Plottable-method calls have no native polars impl and stay
    # declined by that guard.
    if _active_frames_are_polars(g):
        if function == "get_degrees":
            from graphistry.compute.gfql.lazy.engine.polars.degrees import get_degrees_polars
            return get_degrees_polars(g, **validated_params)
        if function == "get_indegrees":
            from graphistry.compute.gfql.lazy.engine.polars.degrees import get_indegrees_polars
            return get_indegrees_polars(g, **validated_params)
        if function == "get_outdegrees":
            from graphistry.compute.gfql.lazy.engine.polars.degrees import get_outdegrees_polars
            return get_outdegrees_polars(g, **validated_params)

    if not hasattr(g, function):
        raise AttributeError(
            f"Plottable has no method '{function}'. "
            f"This should not happen if safelist is properly configured."
        )

    # Off-engine analytic (no native polars impl — the row-pipeline + native-degree paths
    # above already returned for the native ops). Under a polars engine, bridge to pandas/cuDF
    # (call_mode='auto', no-op if frames already match) or decline (call_mode='strict'). Gated on
    # engine, not frame type, so strict declines every off-engine analytic consistently.
    if engine in (Engine.POLARS, Engine.POLARS_GPU):
        g = _bridge_graph_for_offengine_call(g, function, engine)

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

        result = _execute_validated_call(g, function, validated_params, engine)
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

        # Ensure result matches requested engine (defensive coercion). Schema-changing ops
        # (UMAP, hypergraph) may alter DataFrame types. Under a polars engine, an off-engine
        # analytic ran on pandas/cuDF via the call_mode='auto' bridge (see _execute_validated_call);
        # coerce its result back to polars losslessly (Arrow). call_mode='strict' already declined
        # upstream with NotImplementedError (propagated cleanly below).
        if _is_plottable_like(result):
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
    if isinstance(error, NotImplementedError) and engine in (Engine.POLARS, Engine.POLARS_GPU):
        # Honest engine-capability decline — propagate as-is so the DAG surface matches the chain
        # surface's NotImplementedError. Sources: call_mode='strict' off-engine decline, and the
        # polars-gpu GPU-or-error when cuDF is unavailable (_bridge_graph_for_offengine_call).
        # Gated to the polars engines: a pandas/cudf NIE (e.g. fa2_layout requiring a GPU) must
        # still fall through to the GFQLTypeError(E303) wrapper below, not leak as a bare NIE.
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
