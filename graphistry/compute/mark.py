"""
Mark mode: Annotate nodes/edges with boolean columns based on GFQL patterns.

Unlike filtering operations, marking preserves ALL entities and adds a boolean
column indicating which entities matched the pattern.
"""
from typing import TYPE_CHECKING, List, Optional, Union
import pandas as pd

from graphistry.Engine import EngineAbstract, resolve_engine, df_to_engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .ast import ASTObject, ASTNode, ASTEdge
from .chain import Chain
from .exceptions import ErrorCode, GFQLTypeError, GFQLSyntaxError, GFQLSchemaError

if TYPE_CHECKING:
    from typing import Tuple


logger = setup_logger(__name__)


def _validate_mark_params(
    gfql: Chain,
    name: str,
    target_df: pd.DataFrame
) -> None:
    """Validate mark() parameters.

    Args:
        gfql: GFQL pattern as Chain (already converted from list/JSON)
        name: Name for the boolean marker column
        target_df: Target DataFrame (nodes or edges)

    Raises:
        GFQLTypeError: If parameters have wrong type
        GFQLSyntaxError: If GFQL pattern is invalid
        GFQLSchemaError: If column name conflicts
    """
    # Validate gfql type (should already be Chain at this point)
    if not isinstance(gfql, Chain):
        raise GFQLTypeError(
            ErrorCode.E201,
            "gfql must be Chain (internal error)",
            field="gfql",
            value=type(gfql).__name__
        )

    # Validate gfql not empty
    if len(gfql.chain) == 0:
        raise GFQLSyntaxError(
            ErrorCode.E105,
            "gfql cannot be empty",
            field="gfql",
            suggestion="Provide at least one node or edge matcher"
        )

    # Validate final operation is node or edge matcher
    final_op = gfql.chain[-1]
    if not isinstance(final_op, (ASTNode, ASTEdge)):
        raise GFQLSyntaxError(
            ErrorCode.E104,
            "mark() requires GFQL ending with node or edge matcher",
            field="gfql",
            value=type(final_op).__name__,
            suggestion=f"Remove {type(final_op).__name__} from end of pattern"
        )

    # Validate name type
    if not isinstance(name, str):
        raise GFQLTypeError(
            ErrorCode.E201,
            "name must be a string",
            field="name",
            value=type(name).__name__
        )

    # Validate name not empty
    if len(name) == 0:
        raise GFQLTypeError(
            ErrorCode.E106,
            "name cannot be empty",
            field="name",
            value=name
        )

    # Validate name doesn't conflict with internal columns
    from graphistry.compute.gfql.identifiers import validate_column_name
    try:
        validate_column_name(name, "mark() name parameter")
    except ValueError as e:
        # Convert ValueError to GFQLSchemaError
        raise GFQLSchemaError(
            ErrorCode.E304,
            str(e),
            field="name",
            value=name,
            suggestion="Use user-facing column name without '__gfql_' prefix"
        ) from e

    # Validate name doesn't already exist
    if name in target_df.columns:
        raise GFQLSchemaError(
            ErrorCode.E301,
            f"Column '{name}' already exists",
            field="name",
            value=name,
            suggestion="Choose different name or drop existing column first"
        )


def mark(
    self: Plottable,
    gfql: Union[Chain, List[ASTObject]],
    name: Optional[str] = None,
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """Mark nodes or edges matching GFQL pattern with boolean column.

    Executes GFQL pattern and adds boolean column indicating matches.
    Unlike filtering, ALL entities are preserved.

    Args:
        gfql: GFQL pattern to match (Chain or list of AST objects)
        name: Name for the boolean marker column (defaults to 'is_matched_node' or 'is_matched_edge')
        engine: Execution engine (pandas/cudf/dask)

    Returns:
        New Plottable with marker column added to nodes or edges

    Raises:
        GFQLTypeError: If gfql is not Chain or List[ASTObject]
        GFQLSyntaxError: If gfql pattern is invalid
        GFQLSchemaError: If name conflicts with internal columns

    Example:
        # Mark VIP customers
        g2 = g.mark(
            gfql=[n({'customer_type': 'VIP'})],
            name='is_vip'
        )

        # Multiple marks accumulate
        g3 = g2.mark(
            gfql=[n({'region': 'EMEA'})],
            name='is_emea'
        )
        # g3._nodes has both 'is_vip' and 'is_emea' columns

    See Also:
        - :meth:`gfql`: General GFQL execution
        - :meth:`filter_nodes_by_dict`: Filter nodes (removes non-matches)
        - :meth:`call`: Use mark in call operations: call('mark', {...})
    """
    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    g = self

    # Convert gfql to Chain if needed
    if isinstance(gfql, list):
        # List of ASTObject instances
        gfql_chain = Chain(gfql)
    elif isinstance(gfql, dict):
        # JSON representation (from remote execution)
        # Import here to avoid circular dependency
        from graphistry.compute.chain import Chain as ChainClass
        gfql_chain = ChainClass.from_json(gfql, validate=True)
    elif isinstance(gfql, Chain):
        # Already a Chain
        gfql_chain = gfql
    else:
        raise GFQLTypeError(
            ErrorCode.E201,
            "gfql must be Chain, List[ASTObject], or JSON dict",
            field="gfql",
            value=type(gfql).__name__,
            suggestion="Use [n()] or chain([n()])"
        )

    # Validate gfql is not empty (do this before accessing chain[-1])
    if len(gfql_chain.chain) == 0:
        raise GFQLSyntaxError(
            ErrorCode.E105,
            "gfql cannot be empty",
            field="gfql",
            suggestion="Provide at least one node or edge matcher"
        )

    # Determine target (nodes or edges) from final operation
    final_op = gfql_chain.chain[-1]
    is_node_mark = isinstance(final_op, ASTNode)

    # Generate default name if not provided
    if name is None:
        name = 'is_matched_node' if is_node_mark else 'is_matched_edge'
        logger.debug(f"Using default mark name: '{name}'")

    # Get target DataFrame
    id_col: Union[str, List[str]]
    if is_node_mark:
        if g._nodes is None:
            g = g.materialize_nodes(engine=engine)
        target_df = g._nodes
        id_col = g._node if g._node is not None else 'node'
    else:
        if g._edges is None:
            raise ValueError("Cannot mark edges when graph has no edges")
        target_df = g._edges
        # For edges, we need both source and destination
        src = g._source if g._source is not None else 'src'
        dst = g._destination if g._destination is not None else 'dst'
        id_col = [src, dst]

    # Validate parameters
    _validate_mark_params(gfql_chain, name, target_df)

    # Execute GFQL to get matches
    logger.debug(f"Executing GFQL for mark '{name}'")
    # gfql is added to Plottable via ComputeMixin at runtime
    matched_g = g.gfql(gfql_chain, engine=engine)  # type: ignore[attr-defined]

    # Get matched DataFrame
    if is_node_mark:
        matched_df = matched_g._nodes
    else:
        matched_df = matched_g._edges

    # Create boolean mask
    if is_node_mark:
        # Simple node ID matching
        if len(matched_df) == 0:
            # No matches - all False
            mask = pd.Series([False] * len(target_df), index=target_df.index)
        else:
            # Type narrowing: id_col is str in node case
            assert isinstance(id_col, str), "id_col must be str for node marking"
            matched_ids = set(matched_df[id_col].values)
            mask = target_df[id_col].isin(matched_ids)
    else:
        # Edge matching on (source, destination) pairs
        if len(matched_df) == 0:
            # No matches - all False
            mask = pd.Series([False] * len(target_df), index=target_df.index)
        else:
            # Type narrowing: id_col is List[str] in edge case
            assert isinstance(id_col, list) and len(id_col) == 2, "id_col must be list of 2 strings for edge marking"
            src_col, dst_col = id_col[0], id_col[1]
            # Create set of (source, dest) tuples for efficient lookup
            matched_pairs = set(
                zip(matched_df[src_col].values, matched_df[dst_col].values)
            )
            # Check if each edge is in matched set
            mask = target_df.apply(
                lambda row: (row[src_col], row[dst_col]) in matched_pairs,
                axis=1
            )

    # Add boolean column to target DataFrame
    engine_concrete = resolve_engine(engine, target_df)
    target_df = df_to_engine(target_df, engine_concrete)
    enriched_df = target_df.assign(**{name: mask})

    logger.debug(f"Marked {mask.sum()}/{len(mask)} {'nodes' if is_node_mark else 'edges'} as '{name}'")

    # Return new graph with enriched DataFrame
    if is_node_mark:
        return g.nodes(enriched_df)
    else:
        return g.edges(enriched_df)
