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
        # Check if conversion is needed by detecting types of both nodes and edges
        # Schema-changing operations (UMAP hypergraph) may create edges/nodes in different types
        nodes_engine = resolve_engine(EngineAbstract.AUTO, g._nodes) if g._nodes is not None else None
        edges_engine = resolve_engine(EngineAbstract.AUTO, g._edges) if g._edges is not None else None

        # Check if either DataFrame needs conversion
        nodes_need_conversion = (nodes_engine is not None and nodes_engine != requested_engine)
        edges_need_conversion = (edges_engine is not None and edges_engine != requested_engine)

        # If both types already match, return as-is (no-op optimization)
        if not nodes_need_conversion and not edges_need_conversion:
            return g

        # Log conversion for debugging
        if nodes_need_conversion and nodes_engine is not None:
            logger.info(
                "Engine mismatch in nodes: requested %s but data is %s. Converting.",
                requested_engine.value,
                nodes_engine.value
            )
        if edges_need_conversion and edges_engine is not None:
            logger.info(
                "Engine mismatch in edges: requested %s but data is %s. Converting.",
                requested_engine.value,
                edges_engine.value
            )

        # Convert DataFrames that need conversion
        new_nodes = df_to_engine(g._nodes, requested_engine) if nodes_need_conversion else g._nodes
        new_edges = df_to_engine(g._edges, requested_engine) if edges_need_conversion else g._edges

        # Return new Plottable with converted DataFrames
        return g.nodes(new_nodes).edges(new_edges)

    except Exception as e:
        # Graceful degradation: log error but return original graph
        # Better to return "wrong" type than crash user's workflow
        nodes_type = nodes_engine.value if 'nodes_engine' in locals() and nodes_engine else 'unknown'
        edges_type = edges_engine.value if 'edges_engine' in locals() and edges_engine else 'unknown'
        logger.warning(
            "Engine coercion failed: %s. Returning original graph (nodes=%s, edges=%s) instead of requested %s.",
            str(e),
            nodes_type,
            edges_type,
            requested_engine.value,
            exc_info=True
        )
        return g
