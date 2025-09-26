"""Typed builders for GFQL call operations.

These provide type-safe, IDE-friendly wrappers around the dynamic call() system.
"""

from typing import Optional, List, Dict, Any, Literal
from .ast import ASTCall, call


def hypergraph(
    entity_types: Optional[List[str]] = None,
    opts: Optional[Dict[str, Any]] = None,
    drop_na: bool = True,
    drop_edge_attrs: bool = True,
    verbose: bool = False,
    direct: bool = True,
    engine: Literal['pandas', 'cudf', 'dask', 'auto'] = 'auto',
    npartitions: Optional[int] = None,
    chunksize: Optional[int] = None
) -> ASTCall:
    """Create a hypergraph transformation for GFQL.

    Transforms event data into entity relationships by connecting entities
    that appear together in events.

    Args:
        entity_types: Column names to use as entity types. If None, uses all columns.
        opts: Additional options to pass to the hypergraph engine.
        drop_na: Whether to drop rows with NA values in entity columns.
        drop_edge_attrs: Whether to drop non-entity attributes from edges.
        verbose: Whether to print verbose output during transformation.
        direct: If True, creates direct entity-to-entity edges.
                If False, keeps hypernodes to show event connections.
        engine: Processing engine - 'pandas', 'cudf' (GPU), 'dask' (streaming), or 'auto'.
        npartitions: Number of partitions for Dask processing.
        chunksize: Chunk size for streaming processing.

    Returns:
        ASTCall object for use in gfql() or gfql_remote().

    Example:
        >>> from graphistry.compute import hypergraph
        >>> import pandas as pd
        >>>
        >>> events_df = pd.DataFrame({
        ...     'user': ['alice', 'bob', 'alice'],
        ...     'product': ['laptop', 'phone', 'tablet']
        ... })
        >>> g = graphistry.nodes(events_df)
        >>>
        >>> # Simple transformation
        >>> hg = g.gfql(hypergraph(entity_types=['user', 'product']))
        >>>
        >>> # With options
        >>> hg = g.gfql(hypergraph(
        ...     entity_types=['user', 'product'],
        ...     direct=False,  # Keep hypernodes
        ...     engine='cudf'   # Use GPU
        ... ))
        >>>
        >>> # In a DAG
        >>> from graphistry.compute import let, ref, n
        >>> result = g.gfql(
        ...     let({
        ...         'hg': hypergraph(entity_types=['user', 'product']),
        ...         'filtered': ref('hg', [n({'type': 'user'})])
        ...     })
        ... )
    """
    # Build params dict, excluding None values
    params: Dict[str, Any] = {}

    if entity_types is not None:
        params['entity_types'] = entity_types
    if opts is not None:
        params['opts'] = opts
    if not drop_na:  # Only include if False (not default True)
        params['drop_na'] = drop_na
    if not drop_edge_attrs:  # Only include if False (not default True)
        params['drop_edge_attrs'] = drop_edge_attrs
    if verbose:  # Only include if True (not default False)
        params['verbose'] = verbose
    if not direct:  # Only include if False (not default True)
        params['direct'] = direct
    if engine != 'auto':  # Only include if not default
        params['engine'] = engine
    if npartitions is not None:
        params['npartitions'] = npartitions
    if chunksize is not None:
        params['chunksize'] = chunksize

    return call('hypergraph', params)
