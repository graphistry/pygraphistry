import logging
from typing import List, Optional, Tuple, TYPE_CHECKING, Union

from graphistry.Engine import (
    EngineAbstract, df_concat, df_cons, df_to_engine, resolve_engine
)
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .filter_by_dict import filter_by_dict
from .typing import DataFrameT
from .util import generate_safe_column_name


logger = setup_logger(__name__)


def prepare_merge_dataframe(
    edges_indexed: 'DataFrameT', 
    column_conflict: bool, 
    source_col: str, 
    dest_col: str, 
    edge_id_col: str, 
    node_col: str, 
    temp_col: str, 
    is_reverse: bool = False
) -> 'DataFrameT':
    """
    Prepare a merge DataFrame handling column name conflicts for hop operations.
    Centralizes the conflict resolution logic for both forward and reverse directions.
    
    Parameters:
    -----------
    edges_indexed : DataFrame
        The indexed edges DataFrame
    column_conflict : bool
        Whether there's a column name conflict
    source_col : str
        The source column name
    dest_col : str
        The destination column name
    edge_id_col : str
        The edge ID column name
    node_col : str
        The node column name
    temp_col : str
        The temporary column name to use in case of conflict
    is_reverse : bool, default=False
        Whether to prepare for reverse direction hop
        
    Returns:
    --------
    DataFrame
        A merge DataFrame prepared for hop operation
    """
    # For reverse direction, swap source and destination
    if is_reverse:
        src, dst = dest_col, source_col
    else:
        src, dst = source_col, dest_col
    
    # Select columns based on direction
    required_cols = [src, dst, edge_id_col]
    
    if column_conflict:
        # Handle column conflict by creating temporary column
        merge_df = edges_indexed[required_cols].assign(
            **{temp_col: edges_indexed[src]}
        )
        # Assign node using the temp column
        merge_df = merge_df.assign(**{node_col: merge_df[temp_col]})
    else:
        # No conflict, proceed normally
        merge_df = edges_indexed[required_cols]
        merge_df = merge_df.assign(**{node_col: merge_df[src]})
    
    return merge_df


def query_if_not_none(query: Optional[str], df: DataFrameT) -> DataFrameT:
    if query is None:
        return df
    return df.query(query)


def process_hop_direction(
    direction_name: str,
    wave_front_iter: 'DataFrameT',
    edges_indexed: 'DataFrameT',
    column_conflict: bool,
    source_col: str,
    dest_col: str,
    edge_id_col: str,
    node_col: str,
    temp_col: str,
    intermediate_target_wave_front: Optional['DataFrameT'],
    base_target_nodes: 'DataFrameT',
    target_col: str,
    node_match_query: Optional[str],
    node_match_dict: Optional[dict],
    is_reverse: bool,
    debugging: bool
) -> Tuple['DataFrameT', 'DataFrameT']:
    """
    Process a single hop direction (forward or reverse)
    
    Parameters:
    -----------
    direction_name : str
        Name of the direction for debug logging ('forward' or 'reverse')
    wave_front_iter : DataFrame
        Current wave front of nodes to expand from
    edges_indexed : DataFrame
        The indexed edges DataFrame
    column_conflict : bool
        Whether there's a name conflict between node and edge columns
    source_col : str
        The source column name
    dest_col : str
        The destination column name
    edge_id_col : str
        The edge ID column name
    node_col : str
        The node column name
    temp_col : str
        The temporary column name for conflict resolution
    intermediate_target_wave_front : DataFrame or None
        Pre-calculated target wave front for filtering
    base_target_nodes : DataFrame
        The base target nodes for destination filtering
    target_col : str
        The target column for merging (destination or source depending on direction)
    node_match_query : str or None
        Optional query for node filtering
    node_match_dict : dict or None
        Optional dictionary for node filtering
    is_reverse : bool
        Whether this is the reverse direction
    debugging : bool
        Whether debug logging is enabled
        
    Returns:
    --------
    Tuple[DataFrame, DataFrame]
        The processed hop edges and node IDs
    """
    
    # Prepare edges for merging using centralized function
    merge_df = prepare_merge_dataframe(
        edges_indexed=edges_indexed,
        column_conflict=column_conflict,
        source_col=source_col,
        dest_col=dest_col,
        edge_id_col=edge_id_col,
        node_col=node_col,
        temp_col=temp_col,
        is_reverse=is_reverse
    )
    
    # Select the appropriate columns based on direction
    if is_reverse:
        # For reverse direction: dst, src, id
        ordered_cols = [dest_col, source_col, edge_id_col]
    else:
        # For forward direction: src, dst, id
        ordered_cols = [source_col, dest_col, edge_id_col]
    
    # Merge with wavefront to follow links
    hop_edges = (
        wave_front_iter.merge(
            merge_df,
            how='inner',
            on=node_col)
        [ordered_cols]
    )
    
    if debugging:
        logger.debug('--- direction %s ---', direction_name)
        logger.debug('hop_edges basic:\n%s', hop_edges)
    
    # Apply target wave front filtering if provided
    if intermediate_target_wave_front is not None:
        hop_edges = hop_edges.merge(
            intermediate_target_wave_front.rename(columns={node_col: target_col}),
            how='inner',
            on=target_col
        )
        if debugging:
            logger.debug('hop_edges filtered by target_wave_front:\n%s', hop_edges)
    
    # Extract node IDs from results - use the appropriate column based on direction
    result_col = source_col if is_reverse else dest_col
    new_node_ids = hop_edges[[result_col]].rename(columns={result_col: node_col}).drop_duplicates()
    
    # Apply node filtering if needed
    if node_match_query is not None or node_match_dict is not None:
        if debugging:
            logger.debug('--- node filtering ---')
            logger.debug('node_match_query: %s', node_match_query)
            logger.debug('node_match_dict: %s', node_match_dict)
            logger.debug('base_target_nodes:\n%s', base_target_nodes)
            logger.debug('new_node_ids:\n%s', new_node_ids)
            logger.debug('enriched nodes for filtering:\n%s', 
                        base_target_nodes.merge(new_node_ids, on=node_col, how='inner'))
            
        new_node_ids = query_if_not_none(
            node_match_query,
            filter_by_dict(
                base_target_nodes.merge(new_node_ids, on=node_col, how='inner'),
                node_match_dict
        ))[[node_col]]
        
        hop_edges = hop_edges.merge(
            new_node_ids.rename(columns={node_col: target_col}),
            how='inner',
            on=target_col
        )
        
        if debugging:
            logger.debug('new_node_ids after filtering:\n%s', new_node_ids)
            logger.debug('hop_edges filtered by node predicates:\n%s', hop_edges)
    
    if debugging:
        logger.debug('hop_edges final:\n%s', hop_edges)
        logger.debug('new_node_ids final:\n%s', new_node_ids)
        
    return hop_edges, new_node_ids


def hop(self: Plottable,
    nodes: Optional[DataFrameT] = None,  # chain: incoming wavefront
    hops: Optional[int] = 1,
    to_fixed_point: bool = False,
    direction: str = 'forward',
    edge_match: Optional[dict] = None,
    source_node_match: Optional[dict] = None,
    destination_node_match: Optional[dict] = None,
    source_node_query: Optional[str] = None,
    destination_node_query: Optional[str] = None,
    edge_query: Optional[str] = None,
    return_as_wave_front = False,
    target_wave_front: Optional[DataFrameT] = None,  # chain: limit hits to these for reverse pass
    engine: Union[EngineAbstract, str] = EngineAbstract.AUTO
) -> Plottable:
    """
    Given a graph and some source nodes, return subgraph of all paths within k-hops from the sources

    This can be faster than the equivalent chain([...]) call that wraps it with additional steps

    See chain() examples for examples of many of the parameters

    g: Plotter
    nodes: dataframe with id column matching g._node. None signifies all nodes (default).
    hops: consider paths of length 1 to 'hops' steps, if any (default 1).
    to_fixed_point: keep hopping until no new nodes are found (ignores hops)
    direction: 'forward', 'reverse', 'undirected'
    edge_match: dict of kv-pairs to exact match (see also: filter_edges_by_dict)
    source_node_match: dict of kv-pairs to match nodes before hopping (including intermediate)
    destination_node_match: dict of kv-pairs to match nodes after hopping (including intermediate)
    source_node_query: dataframe query to match nodes before hopping (including intermediate)
    destination_node_query: dataframe query to match nodes after hopping (including intermediate)
    edge_query: dataframe query to match edges before hopping (including intermediate)
    return_as_wave_front: Exclude starting node(s) in return, returning only encountered nodes
    target_wave_front: Only consider these nodes + self._nodes for reachability
    engine: 'auto', 'pandas', 'cudf' (GPU)
    """

    """
    When called by chain() during reverse phase:
    - return_as_wave_front: True
    - this hop will be `op.reverse()`
    - nodes will be the wavefront of the next step
    
    """

    if isinstance(engine, str):
        engine = EngineAbstract(engine)

    engine_concrete = resolve_engine(engine, self)
    if not TYPE_CHECKING:
        DataFrameT = df_cons(engine_concrete)
    concat = df_concat(engine_concrete)
    
    nodes = df_to_engine(nodes, engine_concrete) if nodes is not None else None
    target_wave_front = df_to_engine(target_wave_front, engine_concrete) if target_wave_front is not None else None

    #TODO target_wave_front code also includes nodes for handling intermediate hops
    # ... better to make an explicit param of allowed intermediates? (vs recording each intermediate hop)

    debugging_hop = False

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('=======================')
        logger.debug('======== HOP ==========')
        logger.debug('nodes:\n%s', nodes)
        logger.debug('self._nodes:\n%s', self._nodes)
        logger.debug('self._edges:\n%s', self._edges)
        logger.debug('hops: %s', hops)
        logger.debug('to_fixed_point: %s', to_fixed_point)
        logger.debug('direction: %s', direction)
        logger.debug('edge_match: %s', edge_match)
        logger.debug('source_node_match: %s', source_node_match)
        logger.debug('destination_node_match: %s', destination_node_match)
        logger.debug('source_node_query: %s', source_node_query)
        logger.debug('destination_node_query: %s', destination_node_query)
        logger.debug('edge_query: %s', edge_query)
        logger.debug('return_as_wave_front: %s', return_as_wave_front)
        logger.debug('target_wave_front:\n%s', target_wave_front)
        logger.debug('engine: %s', engine)
        logger.debug('engine_concrete: %s', engine_concrete)
        logger.debug('---------------------')

    if not to_fixed_point and not isinstance(hops, int):
        raise ValueError(f'Must provide hops int when to_fixed_point is False, received: {hops}')

    if direction not in ['forward', 'reverse', 'undirected']:
        raise ValueError(f'Invalid direction: "{direction}", must be one of: "forward" (default), "reverse", "undirected"')
    
    if target_wave_front is not None and nodes is None:
        raise ValueError('target_wave_front requires nodes to target against (for intermediate hops)')

    if destination_node_match == {}:
        destination_node_match = None

    g2 = self.materialize_nodes(engine=EngineAbstract(engine_concrete.value))
    logger.debug('materialized node/eddge types: %s, %s', type(g2._nodes), type(g2._edges))

    # Early validation: ensure bindings are not None
    if g2._node is None:
        raise ValueError('Node binding cannot be None, please set g._node via bind() or nodes()')

    if g2._source is None or g2._destination is None:
        raise ValueError('Source and destination binding cannot be None, please set g._source and g._destination via bind() or edges()')

    # Type narrowing assertions for mypy - these are guaranteed by the checks above
    assert g2._source is not None, "Source binding checked above"
    assert g2._destination is not None, "Destination binding checked above"

    # Check for column name conflicts
    node_src_conflict = g2._node == g2._source
    node_dst_conflict = g2._node == g2._destination

    # Only generate temp names if there's a conflict
    TEMP_SRC_COL = str(g2._source)
    TEMP_DST_COL = str(g2._destination)

    if node_src_conflict:
        TEMP_SRC_COL = generate_safe_column_name(g2._source, g2._edges)
        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Node column conflicts with source column, using temp name: %s', TEMP_SRC_COL)

    if node_dst_conflict:
        TEMP_DST_COL = generate_safe_column_name(g2._destination, g2._edges)
        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('Node column conflicts with destination column, using temp name: %s', TEMP_DST_COL)

    starting_nodes = nodes if nodes is not None else g2._nodes

    if g2._edge is None:
        # Get the pre-filtered edges
        pre_indexed_edges = query_if_not_none(edge_query, g2.filter_edges_by_dict(edge_match)._edges)

        # Generate a guaranteed unique internal column name to avoid conflicts with user data
        GFQL_EDGE_INDEX = generate_safe_column_name('edge_index', pre_indexed_edges, prefix='__gfql_', suffix='__')

        # reset_index() adds the index as a column, creating 'index' if there's no name, or 'level_0', etc. if there is
        edges_indexed = pre_indexed_edges.reset_index(drop=False)
        # Find the index column (it will be the first column that wasn't in original columns)
        # reset_index() always adds the new column at position 0, so we can use next() with a generator for early exit
        pre_indexed_cols = set(pre_indexed_edges.columns)
        index_col_name = next(col for col in edges_indexed.columns if col not in pre_indexed_cols)
        edges_indexed = edges_indexed.rename(columns={index_col_name: GFQL_EDGE_INDEX})
        EDGE_ID = GFQL_EDGE_INDEX
    else:
        edges_indexed = query_if_not_none(edge_query, g2.filter_edges_by_dict(edge_match)._edges)
        EDGE_ID = g2._edge
        # Defensive check: ensure edge binding column exists
        if EDGE_ID not in edges_indexed.columns:
            raise ValueError(f"Edge binding column '{EDGE_ID}' (from g._edge='{g2._edge}') not found in edges. Available columns: {list(edges_indexed.columns)}")

    hops_remaining = hops

    wave_front = starting_nodes[[g2._node]][:0]

    matches_nodes = None
    matches_edges = edges_indexed[[EDGE_ID]][:0]

    #richly-attributed subset for dest matching & return-enriching
    if target_wave_front is None:
        base_target_nodes = g2._nodes
    else:
        base_target_nodes = concat([target_wave_front, g2._nodes], ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])
    #TODO precompute src/dst match subset if multihop?

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('~~~~~~~~~~ LOOP PRE ~~~~~~~~~~~')
        logger.debug('starting_nodes:\n%s', starting_nodes)
        logger.debug('g2._nodes:\n%s', g2._nodes)
        logger.debug('g2._edges:\n%s', g2._edges)
        logger.debug('edges_indexed:\n%s', edges_indexed)
        logger.debug('=====================')

    first_iter = True
    combined_node_ids = None
    while True:

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP BEGIN ~~~~~~~~~~~')
            logger.debug('hops_remaining: %s', hops_remaining)
            logger.debug('wave_front:\n%s', wave_front)
            logger.debug('matches_nodes:\n%s', matches_nodes)
            logger.debug('matches_edges:\n%s', matches_edges)
            logger.debug('first_iter: %s', first_iter)
            logger.debug('source_node_match: %s', source_node_match)
            logger.debug('starting_nodes:\n%s', starting_nodes)
            logger.debug('self._nodes:\n%s', self._nodes)
            logger.debug('wave_front:\n%s', wave_front)
            logger.debug('wave_front_base:\n%s',
                starting_nodes
                if first_iter else
                wave_front.merge(self._nodes, on=g2._node, how='left'),
            )

        if not to_fixed_point and hops_remaining is not None:
            if hops_remaining < 1:
                break
            hops_remaining = hops_remaining - 1
        
        assert len(wave_front.columns) == 1, "just indexes"
        wave_front_iter : DataFrameT = query_if_not_none(
            source_node_query,
            filter_by_dict(
                starting_nodes
                if first_iter else
                wave_front.merge(self._nodes, on=g2._node, how='left'),
                source_node_match
            )
        )[[ g2._node ]]
        first_iter = False

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP CONTINUE ~~~~~~~~~~~')
            logger.debug('wave_front_iter:\n%s', wave_front_iter)
            
        # Pre-calculate intermediate_target_wave_front once for this iteration
        # This will be used for both forward and reverse directions if needed
        intermediate_target_wave_front = None
        if target_wave_front is not None:
            # Calculate this once for both directions
            if hops_remaining:
                intermediate_target_wave_front = concat([
                    target_wave_front[[g2._node]],
                    self._nodes[[g2._node]]
                    ], sort=False, ignore_index=True
                ).drop_duplicates()
            else:
                intermediate_target_wave_front = target_wave_front[[g2._node]]

        # Initialize hop edges and node IDs for both directions
        hop_edges_forward = None
        new_node_ids_forward = None
        hop_edges_reverse = None
        new_node_ids_reverse = None
        
        # Process the forward direction if needed
        if direction in ['forward', 'undirected']:
            hop_edges_forward, new_node_ids_forward = process_hop_direction(
                direction_name='forward',
                wave_front_iter=wave_front_iter,
                edges_indexed=edges_indexed,
                column_conflict=node_src_conflict,
                source_col=g2._source,
                dest_col=g2._destination,
                edge_id_col=EDGE_ID,
                node_col=g2._node,
                temp_col=TEMP_SRC_COL,
                intermediate_target_wave_front=intermediate_target_wave_front,
                base_target_nodes=base_target_nodes,
                target_col=g2._destination,
                node_match_query=destination_node_query,
                node_match_dict=destination_node_match,
                is_reverse=False,
                debugging=debugging_hop and logger.isEnabledFor(logging.DEBUG)
            )

        # Process the reverse direction if needed
        if direction in ['reverse', 'undirected']:
            hop_edges_reverse, new_node_ids_reverse = process_hop_direction(
                direction_name='reverse',
                wave_front_iter=wave_front_iter,
                edges_indexed=edges_indexed,
                column_conflict=node_dst_conflict,
                source_col=g2._source,
                dest_col=g2._destination,
                edge_id_col=EDGE_ID,
                node_col=g2._node,
                temp_col=TEMP_DST_COL,
                intermediate_target_wave_front=intermediate_target_wave_front,
                base_target_nodes=base_target_nodes,
                target_col=g2._source,
                node_match_query=destination_node_query,
                node_match_dict=destination_node_match,
                is_reverse=True,
                debugging=debugging_hop and logger.isEnabledFor(logging.DEBUG)
            )

        mt : List[DataFrameT] = []  # help mypy

        matches_edges = concat(
            [ matches_edges ]
            + ([ hop_edges_forward[[ EDGE_ID ]] ] if hop_edges_forward is not None else mt)  # noqa: W503
            + ([ hop_edges_reverse[[ EDGE_ID ]] ] if hop_edges_reverse is not None else mt),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates(subset=[EDGE_ID])

        new_node_ids = concat(
            mt
                + ( [ new_node_ids_forward ] if new_node_ids_forward is not None else mt )  # noqa: W503
                + ( [ new_node_ids_reverse] if new_node_ids_reverse is not None else mt ),  # noqa: W503
            ignore_index=True, sort=False).drop_duplicates()

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP MERGES 1 ~~~~~~~~~~~')
            logger.debug('matches_edges:\n%s', matches_edges)
            logger.debug('matches_nodes:\n%s', matches_nodes)
            logger.debug('new_node_ids:\n%s', new_node_ids)
            logger.debug('hop_edges_forward:\n%s', hop_edges_forward)
            logger.debug('hop_edges_reverse:\n%s', hop_edges_reverse)

        # When !return_as_wave_front, include starting nodes in returned matching node set
        # (When return_as_wave_front, skip starting nodes, just include newly reached)
        # Only need to do this in the first loop step
        if matches_nodes is None:  # first iteration
            if return_as_wave_front:
                matches_nodes = new_node_ids[:0]
            else:
                matches_nodes = concat(
                    mt
                        + ( [hop_edges_forward[[g2._source]].rename(columns={g2._source: g2._node}).drop_duplicates()]  # noqa: W503
                            if hop_edges_forward is not None
                            else mt)
                        + ( [hop_edges_reverse[[g2._destination]].rename(columns={g2._destination: g2._node}).drop_duplicates()]  # noqa: W503
                            if hop_edges_reverse is not None
                            else mt),
                    ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])

            if debugging_hop and logger.isEnabledFor(logging.DEBUG):
                logger.debug('~~~~~~~~~~ LOOP STEP MERGES 2 ~~~~~~~~~~~')
                logger.debug('matches_edges:\n%s', matches_edges)

        if len(matches_nodes) > 0:
            combined_node_ids = concat([matches_nodes, new_node_ids], ignore_index=True, sort=False).drop_duplicates()
        else:
            combined_node_ids = new_node_ids

        if len(combined_node_ids) == len(matches_nodes):
            #fixedpoint, exit early: future will come to same spot!
            break
    
        wave_front = new_node_ids
        matches_nodes = combined_node_ids

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP POST ~~~~~~~~~~~')
            logger.debug('matches_nodes:\n%s', matches_nodes)
            logger.debug('combined_node_ids:\n%s', combined_node_ids)
            logger.debug('wave_front:\n%s', wave_front)
            logger.debug('matches_nodes:\n%s', matches_nodes)

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('~~~~~~~~~~ LOOP END POST ~~~~~~~~~~~')
        logger.debug('matches_nodes:\n%s', matches_nodes)
        logger.debug('matches_edges:\n%s', matches_edges)
        logger.debug('combined_node_ids:\n%s', combined_node_ids)
        logger.debug('nodes (self):\n%s', self._nodes)
        logger.debug('nodes (init):\n%s', nodes)
        logger.debug('target_wave_front:\n%s', target_wave_front)

    #hydrate edges
    final_edges = edges_indexed.merge(matches_edges, on=EDGE_ID, how='inner')
    if EDGE_ID not in self._edges:
        final_edges = final_edges.drop(columns=[EDGE_ID])
    g_out = g2.edges(final_edges)

    #hydrate nodes
    if self._nodes is not None:
        logger.debug('~~~~~~~~~~ NODES HYDRATION ~~~~~~~~~~~')
        #FIXME what was this for? Removed for shortest-path reverse pass fixes
        #if target_wave_front is not None:
        #    rich_nodes = target_wave_front
        #else:
        #    rich_nodes = self._nodes
        rich_nodes = self._nodes
        if target_wave_front is not None:
            rich_nodes = concat([rich_nodes, target_wave_front], ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])
        logger.debug('rich_nodes available for inner merge:\n%s', rich_nodes[[self._node]])
        logger.debug('target_wave_front:\n%s', target_wave_front)
        logger.debug('matches_nodes:\n%s', matches_nodes)
        logger.debug('wave_front:\n%s', wave_front)
        logger.debug('self._nodes:\n%s', self._nodes)
        final_nodes = rich_nodes.merge(
            matches_nodes if matches_nodes is not None else wave_front[:0],
            on=self._node,
            how='inner')
        g_out = g_out.nodes(final_nodes)

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('~~~~~~~~~~ HOP OUTPUT ~~~~~~~~~~~')
        logger.debug('nodes:\n%s', g_out._nodes)
        logger.debug('edges:\n%s', g_out._edges)
        logger.debug('======== /HOP =============')
        logger.debug('==========================')

    return g_out
