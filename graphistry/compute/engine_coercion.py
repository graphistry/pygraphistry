"""Helper utilities for engine coercion across GFQL operations.

This module provides defensive DataFrame type coercion to ensure
all GFQL operations honor the user's explicit engine parameter.
"""

from typing import Optional
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract, resolve_engine, df_to_engine
from graphistry.util import setup_logger

logger = setup_logger(__name__)


def ensure_engine_match(g: Plottable, requested_engine: Engine) -> Plottable:
    """Ensure Plottable's DataFrames match the requested engine.

    If there's a mismatch between the actual DataFrame type and the
    requested engine, converts DataFrames to match the user's request.
    This is a defensive pattern for handling schema-changing operations
    (UMAP, hypergraph) that may alter DataFrame types mid-execution.

    The conversion is a no-op when types already match, making it safe
    to call at multiple layers (chain, let, call) without performance penalty.

    :param g: Plottable to check and potentially convert
    :type g: Plottable
    :param requested_engine: Engine type requested by user (pandas/cudf)
    :type requested_engine: Engine
    :returns: Plottable with DataFrames matching requested engine
    :rtype: Plottable
    :raises: Never raises - returns original graph on error (graceful degradation)

    **Example::**

        # After a schema-changing operation that might have changed types
        result = execute_umap(g)  # May return cuDF even if g was pandas
        result = ensure_engine_match(result, Engine.PANDAS)  # Converts back

    **Performance::**

        - No-op path (types match): ~1 microsecond
        - Conversion path (types differ): ~10-100ms depending on data size
    """
    try:
        # Detect actual engine from nodes DataFrame
        actual_engine = resolve_engine(EngineAbstract.AUTO, g._nodes)

        # If types already match, return as-is (no-op optimization)
        if actual_engine == requested_engine:
            return g

        # Log conversion for debugging
        logger.info(
            "Engine mismatch detected: requested %s but data is %s. Converting to match request.",
            requested_engine.value,
            actual_engine.value
        )

        # Convert both nodes and edges to match requested engine
        new_nodes = df_to_engine(g._nodes, requested_engine)
        new_edges = df_to_engine(g._edges, requested_engine) if g._edges is not None else None

        # Return new Plottable with converted DataFrames
        return g.nodes(new_nodes).edges(new_edges)

    except Exception as e:
        # Graceful degradation: log error but return original graph
        # Better to return "wrong" type than crash user's workflow
        logger.warning(
            "Engine coercion failed: %s. Returning original graph with actual type %s instead of requested %s.",
            str(e),
            actual_engine.value if 'actual_engine' in locals() else 'unknown',
            requested_engine.value,
            exc_info=True
        )
        return g
