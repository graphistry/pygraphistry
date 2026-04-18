"""Helper utilities for engine coercion across GFQL operations.

This module provides defensive DataFrame type coercion to ensure
all GFQL operations honor the user's explicit engine parameter.
"""

from typing import Any, Optional
import numbers
import pandas as pd
from graphistry.Plottable import Plottable
from graphistry.Engine import Engine, EngineAbstract, resolve_engine, df_to_engine
from graphistry.util import setup_logger

logger = setup_logger(__name__)


def _looks_like_graph_table(df: Any) -> bool:
    cols = {str(col) for col in getattr(df, "columns", [])}
    has_node_shape = "id" in cols or any(col.startswith("label__") for col in cols)
    has_edge_shape = ({"s", "d"} <= cols) or ({"src", "dst"} <= cols) or ("edge_id" in cols)
    return has_node_shape or has_edge_shape


def _pandas_row_table_has_lossy_numeric_nulls(df: Any) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    for col in df.columns:
        series = df[col]
        if not getattr(series, "isna", None) or not bool(series.isna().any()):
            continue
        if pd.api.types.is_numeric_dtype(series.dtype):
            return True
        non_null = series.dropna()
        values = list(non_null.tolist()) if len(non_null) > 0 else []
        if values and all(isinstance(value, numbers.Real) and not isinstance(value, bool) for value in values):
            return True
    return False


def _pandas_row_table_has_lossy_object_values(df: Any) -> bool:
    if not isinstance(df, pd.DataFrame):
        return False
    for col in df.columns:
        series = df[col]
        if not pd.api.types.is_object_dtype(series.dtype):
            continue
        non_null = series.dropna()
        values = list(non_null.tolist()) if len(non_null) > 0 else []
        if any(isinstance(value, (list, tuple, dict)) for value in values):
            return True
    return False


def _attach_to_pandas_shim(df: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(df, "to_pandas"):
        df.to_pandas = lambda df=df: df  # type: ignore[attr-defined]
    return df


def _is_non_dask_cudf_df(df: Any) -> bool:
    type_module = type(df).__module__
    return "cudf" in type_module and "dask" not in type_module


def ensure_local_engine_match(g: Plottable, requested_engine: Engine) -> Plottable:
    """Coerce local pandas/cudf DataFrames on a Plottable to requested engine.

    This helper keeps legacy behavior used by `materialize_nodes()`:
    - pandas -> cudf conversion when CUDF requested
    - non-dask-cudf -> pandas conversion when PANDAS requested
    - dask-backed cudf objects are intentionally left unchanged
    """
    if requested_engine == Engine.CUDF:
        if g._nodes is not None and isinstance(g._nodes, pd.DataFrame):
            g = g.nodes(df_to_engine(g._nodes, Engine.CUDF), g._node)
        if g._edges is not None and isinstance(g._edges, pd.DataFrame):
            g = g.edges(
                df_to_engine(g._edges, Engine.CUDF),
                g._source,
                g._destination,
                edge=g._edge,
            )
    elif requested_engine == Engine.PANDAS:
        if g._nodes is not None and _is_non_dask_cudf_df(g._nodes):
            g = g.nodes(df_to_engine(g._nodes, Engine.PANDAS), g._node)
        if g._edges is not None and _is_non_dask_cudf_df(g._edges):
            g = g.edges(
                df_to_engine(g._edges, Engine.PANDAS),
                g._source,
                g._destination,
                edge=g._edge,
            )
    return g


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
        preserve_pandas_row_table = (
            requested_engine == Engine.CUDF
            and nodes_need_conversion
            and isinstance(g._nodes, pd.DataFrame)
            and not _looks_like_graph_table(g._nodes)
            and (
                _pandas_row_table_has_lossy_numeric_nulls(g._nodes)
                or _pandas_row_table_has_lossy_object_values(g._nodes)
            )
        )

        if preserve_pandas_row_table:
            logger.info(
                "Keeping pandas row-table nodes during requested cudf execution to preserve numeric-null semantics."
            )
            new_nodes = _attach_to_pandas_shim(g._nodes.copy())
        else:
            new_nodes = df_to_engine(g._nodes, requested_engine) if nodes_need_conversion else g._nodes
        new_edges = df_to_engine(g._edges, requested_engine) if edges_need_conversion else g._edges

        if preserve_pandas_row_table:
            out = g.bind()
            out._nodes = new_nodes
            out._edges = new_edges
            return out

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


def ensure_pandas(df: Any) -> pd.DataFrame:
    """Convert to pandas if not already (e.g. cuDF). No-op for pandas.

    Uses nullable=True when available (cuDF >= 22.02) to preserve nullable
    integer dtypes through the round-trip, avoiding silent Int64 to float64
    conversion when nulls are present.
    """
    if isinstance(df, pd.DataFrame):
        return df
    try:
        return df.to_pandas(nullable=True)
    except TypeError:
        return df.to_pandas()


def restore_engine(g: Plottable, original_nodes: Any, original_edges: Any) -> Plottable:
    """Convert result DataFrames back to the original engine if needed.

    Detects the engine from the original input frames and converts back
    via :func:`ensure_local_engine_match`. No-op when types already match.
    Failures are logged and the pandas result is returned as-is.
    """
    ref = original_nodes if original_nodes is not None else original_edges
    try:
        engine = resolve_engine(EngineAbstract.AUTO, ref)
        return ensure_local_engine_match(g, engine)
    except Exception:
        logger.warning("Failed to restore engine for graph", exc_info=True)
        return g
