"""
Graph hop/traversal operations for PyGraphistry

NOTE: Excluded from pyre (.pyre_configuration) - hop() complexity causes hang. Use mypy.
"""
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import pandas as pd

from graphistry.Engine import (
    EngineAbstract, df_concat, df_cons, df_to_engine, resolve_engine
)
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .filter_by_dict import filter_by_dict
from graphistry.Engine import safe_merge
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
        safe_merge(
            wave_front_iter,
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
        hop_edges = safe_merge(
            hop_edges,
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
                        safe_merge(base_target_nodes, new_node_ids, on=node_col, how='inner'))

        new_node_ids = query_if_not_none(
            node_match_query,
            filter_by_dict(
                safe_merge(base_target_nodes, new_node_ids, on=node_col, how='inner'),
                node_match_dict
        ))[[node_col]]
        
        hop_edges = safe_merge(
            hop_edges,
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
    *,
    min_hops: Optional[int] = None,
    max_hops: Optional[int] = None,
    output_min_hops: Optional[int] = None,
    output_max_hops: Optional[int] = None,
    label_node_hops: Optional[str] = None,
    label_edge_hops: Optional[str] = None,
    label_seeds: bool = False,
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
    hops: consider paths of length 1 to 'hops' steps, if any (default 1). Shorthand for max_hops.
    min_hops/max_hops: inclusive traversal bounds; defaults preserve legacy behavior (min=1 unless max=0; max defaults to hops).
    output_min_hops/output_max_hops: optional output slice applied after traversal; defaults keep all traversed hops up to max_hops. Useful for showing a subrange (e.g., min/max = 2..4 but display only hops 3..4).
    label_node_hops/label_edge_hops: optional column names for hop numbers (omit or None to skip). Nodes record the first hop step they are reached (1 = first expansion); edges record the hop step that traversed them.
    label_seeds: when True and labeling, also write hop 0 for seed nodes in the node label column.
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

    def _combine_first_no_warn(target, fill):
        """Avoid pandas concat warning when combine_first sees empty inputs."""
        if target is None or len(target) == 0:
            return target
        return target.where(target.notna(), fill)

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

    if direction not in ['forward', 'reverse', 'undirected']:
        raise ValueError(f'Invalid direction: "{direction}", must be one of: "forward" (default), "reverse", "undirected"')
    
    if target_wave_front is not None and nodes is None:
        raise ValueError('target_wave_front requires nodes to target against (for intermediate hops)')

    # Resolve hop bounds with legacy compatibility
    resolved_max_hops = max_hops if max_hops is not None else hops
    resolved_min_hops = min_hops

    if not to_fixed_point:
        if resolved_max_hops is not None and not isinstance(resolved_max_hops, int):
            raise ValueError(f'Must provide integer hops when to_fixed_point is False, received: {resolved_max_hops}')
    else:
        resolved_max_hops = None

    if resolved_min_hops is None:
        resolved_min_hops = 0 if resolved_max_hops == 0 else 1

    if resolved_min_hops < 0:
        raise ValueError(f'min_hops must be >= 0, received: {resolved_min_hops}')

    if resolved_max_hops is not None and resolved_max_hops < 0:
        raise ValueError(f'max_hops must be >= 0, received: {resolved_max_hops}')

    if resolved_max_hops is not None and resolved_min_hops > resolved_max_hops:
        raise ValueError(f'min_hops ({resolved_min_hops}) cannot exceed max_hops ({resolved_max_hops})')

    resolved_output_min = output_min_hops
    resolved_output_max = output_max_hops

    if resolved_output_min is not None and resolved_output_min < 0:
        raise ValueError(f'output_min_hops must be >= 0, received: {resolved_output_min}')
    if resolved_output_max is not None and resolved_output_max < 0:
        raise ValueError(f'output_max_hops must be >= 0, received: {resolved_output_max}')
    if resolved_output_min is not None and resolved_output_max is not None and resolved_output_min > resolved_output_max:
        raise ValueError(f'output_min_hops ({resolved_output_min}) cannot exceed output_max_hops ({resolved_output_max})')

    # Default output slice: include all traversed hops unless explicitly post-filtered
    if resolved_output_max is None:
        resolved_output_max = resolved_max_hops

    # Keep output slice within traversal range if both known
    if resolved_output_min is not None and resolved_max_hops is not None and resolved_output_min > resolved_max_hops:
        raise ValueError(f'output_min_hops ({resolved_output_min}) cannot exceed max_hops traversal bound ({resolved_max_hops})')
    if resolved_output_max is not None and resolved_min_hops is not None and resolved_output_max < resolved_min_hops:
        raise ValueError(f'output_max_hops ({resolved_output_max}) cannot be below min_hops traversal bound ({resolved_min_hops})')

    final_output_min = resolved_output_min
    final_output_max = resolved_output_max

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

    seeds_provided = nodes is not None
    starting_nodes = nodes if seeds_provided else g2._nodes
    if starting_nodes is None:
        raise ValueError('hop requires a node DataFrame; starting_nodes is None')

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

    def resolve_label_col(requested: Optional[str], df, default_base: str) -> Optional[str]:
        if requested is None:
            return generate_safe_column_name(default_base, df, prefix='__gfqlhop_', suffix='__')
        if requested not in df.columns:
            return requested
        counter = 1
        candidate = f"{requested}_{counter}"
        while candidate in df.columns:
            counter = counter + 1
            candidate = f"{requested}_{counter}"
        return candidate

    # Track hops when needed for labels, output slices, or min_hops pruning
    needs_min_hop_pruning = resolved_min_hops is not None and resolved_min_hops > 1
    track_hops = bool(
        label_node_hops
        or label_edge_hops
        or label_seeds
        or output_min_hops is not None
        or output_max_hops is not None
        or needs_min_hop_pruning
    )
    track_node_hops = track_hops or bool(label_node_hops or label_seeds)
    track_edge_hops = track_hops or label_edge_hops is not None

    edge_hop_col = None
    node_hop_col = None
    if track_edge_hops:
        edge_hop_col = resolve_label_col(label_edge_hops, edges_indexed, '_hop')
        seen_edge_marker_col = generate_safe_column_name('__gfql_edge_seen__', edges_indexed, prefix='__seen_', suffix='__')
    if track_node_hops:
        node_hop_col = resolve_label_col(label_node_hops, g2._nodes, '_hop')
        seen_node_marker_col = generate_safe_column_name('__gfql_node_seen__', g2._nodes, prefix='__seen_', suffix='__')

    wave_front = starting_nodes[[g2._node]][:0]

    matches_nodes = None
    matches_edges = edges_indexed[[EDGE_ID]][:0]

    #richly-attributed subset for dest matching & return-enriching
    if target_wave_front is None:
        base_target_nodes = g2._nodes
    else:
        base_target_nodes = concat([target_wave_front, g2._nodes], ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])
    #TODO precompute src/dst match subset if multihop?

    node_hop_records = None
    edge_hop_records = None

    if track_node_hops and label_seeds and node_hop_col is not None:
        seed_nodes = starting_nodes[[g2._node]].drop_duplicates()
        node_hop_records = seed_nodes.assign(**{node_hop_col: 0})

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('~~~~~~~~~~ LOOP PRE ~~~~~~~~~~~')
        logger.debug('starting_nodes:\n%s', starting_nodes)
        logger.debug('g2._nodes:\n%s', g2._nodes)
        logger.debug('g2._edges:\n%s', g2._edges)
        logger.debug('edges_indexed:\n%s', edges_indexed)
        logger.debug('=====================')

    first_iter = True
    combined_node_ids = None
    current_hop = 0
    max_reached_hop = 0
    while True:

        if not to_fixed_point and resolved_max_hops is not None and current_hop >= resolved_max_hops:
            break

        current_hop += 1

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP BEGIN ~~~~~~~~~~~')
            logger.debug('current_hop: %s', current_hop)
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
                safe_merge(wave_front, self._nodes, on=g2._node, how='left'),
            )

        assert len(wave_front.columns) == 1, "just indexes"
        wave_front_iter : DataFrameT = query_if_not_none(
            source_node_query,
            filter_by_dict(
                starting_nodes
                if first_iter else
                safe_merge(wave_front, self._nodes, on=g2._node, how='left'),
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
            has_more_hops_planned = to_fixed_point or resolved_max_hops is None or current_hop < resolved_max_hops
            if has_more_hops_planned:
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

        if len(new_node_ids) > 0:
            max_reached_hop = current_hop

        if track_edge_hops and edge_hop_col is not None:
            assert seen_edge_marker_col is not None
            edge_label_candidates : List[DataFrameT] = []
            if hop_edges_forward is not None:
                edge_label_candidates.append(hop_edges_forward[[EDGE_ID]])
            if hop_edges_reverse is not None:
                edge_label_candidates.append(hop_edges_reverse[[EDGE_ID]])

            for edge_df_iter in edge_label_candidates:
                if len(edge_df_iter) == 0:
                    continue
                labeled_edges = edge_df_iter.assign(**{edge_hop_col: current_hop})
                if edge_hop_records is None:
                    edge_hop_records = labeled_edges
                else:
                    edge_seen = edge_hop_records[[EDGE_ID]].assign(**{seen_edge_marker_col: 1})
                    merged_edge_labels = safe_merge(
                        labeled_edges,
                        edge_seen,
                        on=EDGE_ID,
                        how='left',
                        engine=engine_concrete
                    )
                    new_edge_labels = merged_edge_labels[merged_edge_labels[seen_edge_marker_col].isna()].drop(columns=[seen_edge_marker_col])
                    if len(new_edge_labels) > 0:
                        edge_hop_records = concat(
                            [edge_hop_records, new_edge_labels],
                            ignore_index=True,
                            sort=False
                        ).drop_duplicates(subset=[EDGE_ID])

        if track_node_hops and node_hop_col is not None:
            assert seen_node_marker_col is not None
            if node_hop_records is None:
                node_hop_records = new_node_ids.assign(**{node_hop_col: current_hop})
            else:
                node_seen = node_hop_records[[g2._node]].assign(**{seen_node_marker_col: 1})
                merged_node_labels = safe_merge(
                    new_node_ids,
                    node_seen,
                    on=g2._node,
                    how='left',
                    engine=engine_concrete
                )
                new_node_labels = merged_node_labels[merged_node_labels[seen_node_marker_col].isna()].drop(columns=[seen_node_marker_col])
                if len(new_node_labels) > 0:
                    node_hop_records = concat(
                        [node_hop_records, new_node_labels.assign(**{node_hop_col: current_hop})],
                        ignore_index=True,
                        sort=False
                    ).drop_duplicates(subset=[g2._node])

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

    if resolved_min_hops is not None and max_reached_hop < resolved_min_hops:
        matches_nodes = starting_nodes[[g2._node]][:0]
        matches_edges = edges_indexed[[EDGE_ID]][:0]
        if node_hop_records is not None:
            node_hop_records = node_hop_records[:0]
        if edge_hop_records is not None:
            edge_hop_records = edge_hop_records[:0]

    # Prune dead-end branches that don't reach min_hops
    # When min_hops > 1, only keep edges/nodes on paths that reach at least min_hops
    if (
        resolved_min_hops is not None
        and resolved_min_hops > 1
        and node_hop_records is not None
        and edge_hop_records is not None
        and node_hop_col is not None
        and edge_hop_col is not None
        and max_reached_hop >= resolved_min_hops
    ):
        # Yannakakis: use edge endpoints, not node_hop_records (lossy min-hop-per-node)
        # A node reachable at hop 1 AND hop 2 only records hop 1 in node_hop_records,
        # but IS a valid goal if reached via a longer path at hop >= min_hops.
        valid_endpoint_edges = edge_hop_records[edge_hop_records[edge_hop_col] >= resolved_min_hops]
        valid_endpoint_edges_with_nodes = safe_merge(
            valid_endpoint_edges,
            edges_indexed[[EDGE_ID, g2._source, g2._destination]],
            on=EDGE_ID,
            how='inner'
        )
        # Use Series instead of set() to avoid GPU->CPU transfers for cudf
        if direction == 'forward':
            goal_node_series = valid_endpoint_edges_with_nodes[g2._destination].drop_duplicates()
        elif direction == 'reverse':
            goal_node_series = valid_endpoint_edges_with_nodes[g2._source].drop_duplicates()
        else:
            # Undirected: either endpoint could be a goal
            goal_node_series = concat([
                valid_endpoint_edges_with_nodes[g2._source],
                valid_endpoint_edges_with_nodes[g2._destination]
            ], ignore_index=True, sort=False).drop_duplicates()

        if len(goal_node_series) > 0:
            # Backtrack from goal nodes to find all edges/nodes on valid paths
            # We need to traverse backwards through the edge records to find which edges lead to goals
            edge_records_with_endpoints = safe_merge(
                edge_hop_records,
                edges_indexed[[EDGE_ID, g2._source, g2._destination]],
                on=EDGE_ID,
                how='inner'
            )

            # Build Series of valid nodes and edges by backtracking from goal nodes
            # Using Series + concat avoids GPU->CPU transfers for cudf
            valid_node_series = goal_node_series
            valid_edge_list = []  # Collect edge Series to concat at end

            # Start with edges that lead TO goal nodes
            current_targets = goal_node_series

            # Backtrack through hops from max edge hop down to 1
            # Use actual max edge hop, not max_reached_hop which may include extra traversal steps
            max_edge_hop = int(edge_hop_records[edge_hop_col].max()) if len(edge_hop_records) > 0 else max_reached_hop
            for hop_level in range(max_edge_hop, 0, -1):
                # Find edges at this hop level that reach current targets
                hop_edges = edge_records_with_endpoints[
                    edge_records_with_endpoints[edge_hop_col] == hop_level
                ]

                if direction == 'forward':
                    # Forward: edges go src->dst, so dst should be in targets
                    reaching_edges = hop_edges[hop_edges[g2._destination].isin(current_targets)]
                    new_source_series = reaching_edges[g2._source]
                elif direction == 'reverse':
                    # Reverse: edges go dst->src conceptually, so src should be in targets
                    reaching_edges = hop_edges[hop_edges[g2._source].isin(current_targets)]
                    new_source_series = reaching_edges[g2._destination]
                else:
                    # Undirected: either endpoint could be in targets
                    reaching_fwd = hop_edges[hop_edges[g2._destination].isin(current_targets)]
                    reaching_rev = hop_edges[hop_edges[g2._source].isin(current_targets)]
                    reaching_edges = concat([reaching_fwd, reaching_rev], ignore_index=True, sort=False).drop_duplicates(subset=[EDGE_ID])
                    new_source_series = concat([
                        reaching_fwd[g2._source],
                        reaching_rev[g2._destination]
                    ], ignore_index=True, sort=False)

                valid_edge_list.append(reaching_edges[EDGE_ID])
                valid_node_series = concat([valid_node_series, new_source_series], ignore_index=True, sort=False)
                current_targets = new_source_series.drop_duplicates()

            # Deduplicate collected nodes and edges
            valid_node_series = valid_node_series.drop_duplicates()
            valid_edge_series = concat(valid_edge_list, ignore_index=True, sort=False).drop_duplicates() if valid_edge_list else goal_node_series[:0]

            # Filter records to only valid paths
            edge_hop_records = edge_hop_records[edge_hop_records[EDGE_ID].isin(valid_edge_series)]
            node_hop_records = node_hop_records[node_hop_records[g2._node].isin(valid_node_series)]
            matches_edges = matches_edges[matches_edges[EDGE_ID].isin(valid_edge_series)]
            if matches_nodes is not None:
                matches_nodes = matches_nodes[matches_nodes[g2._node].isin(valid_node_series)]

    #hydrate edges
    if track_edge_hops and edge_hop_col is not None:
        edge_labels_source = edge_hop_records
        if edge_labels_source is None:
            edge_labels_source = edges_indexed[[EDGE_ID]][:0].assign(**{edge_hop_col: []})

        edge_mask = None
        if final_output_min is not None:
            edge_mask = edge_labels_source[edge_hop_col] >= final_output_min
        if final_output_max is not None:
            max_mask = edge_labels_source[edge_hop_col] <= final_output_max
            edge_mask = max_mask if edge_mask is None else edge_mask & max_mask

        if edge_mask is not None:
            edge_labels_source = edge_labels_source[edge_mask]

        final_edges = safe_merge(edges_indexed, edge_labels_source, on=EDGE_ID, how='inner')
        if label_edge_hops is None and edge_hop_col in final_edges:
            # Preserve hop labels when output slicing is requested so callers can filter
            if output_min_hops is None and output_max_hops is None:
                final_edges = final_edges.drop(columns=[edge_hop_col])
    else:
        final_edges = safe_merge(edges_indexed, matches_edges, on=EDGE_ID, how='inner')

    if EDGE_ID not in self._edges:
        final_edges = final_edges.drop(columns=[EDGE_ID])
    g_out = g2.edges(final_edges)

    #hydrate nodes
    if self._nodes is not None:
        logger.debug('~~~~~~~~~~ NODES HYDRATION ~~~~~~~~~~~')
        rich_nodes = self._nodes
        if target_wave_front is not None:
            rich_nodes = concat([rich_nodes, target_wave_front], ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])
        logger.debug('rich_nodes available for inner merge:\n%s', rich_nodes[[self._node]])
        logger.debug('target_wave_front:\n%s', target_wave_front)
        logger.debug('matches_nodes:\n%s', matches_nodes)
        logger.debug('wave_front:\n%s', wave_front)
        logger.debug('self._nodes:\n%s', self._nodes)

        base_nodes = matches_nodes if matches_nodes is not None else wave_front[:0]

        if track_node_hops and node_hop_col is not None:
            node_labels_source = node_hop_records
            if node_labels_source is None:
                node_labels_source = base_nodes.assign(**{node_hop_col: []})

            node_labels_source = node_labels_source.copy()
            unfiltered_node_labels_source = node_labels_source.copy()
            node_mask = None
            if final_output_min is not None:
                node_mask = node_labels_source[node_hop_col] >= final_output_min
            if final_output_max is not None:
                max_node_mask = node_labels_source[node_hop_col] <= final_output_max
                node_mask = max_node_mask if node_mask is None else node_mask & max_node_mask

            if node_mask is not None:
                node_labels_source.loc[~node_mask, node_hop_col] = pd.NA

            if label_seeds:
                if node_hop_records is not None:
                    seed_rows = node_hop_records[node_hop_col] == 0
                    if seed_rows.any():
                        seeds_for_output = node_hop_records[seed_rows]
                        node_labels_source = concat(
                            [node_labels_source, seeds_for_output],
                            ignore_index=True,
                            sort=False
                        ).drop_duplicates(subset=[g2._node])
                elif starting_nodes is not None and g2._node in starting_nodes.columns:
                    seed_nodes = starting_nodes[[g2._node]].drop_duplicates()
                    node_labels_source = concat(
                        [node_labels_source, seed_nodes.assign(**{node_hop_col: 0})],
                        ignore_index=True,
                        sort=False
                    ).drop_duplicates(subset=[g2._node])

            filtered_nodes = safe_merge(
                base_nodes,
                node_labels_source[[g2._node]],
                on=g2._node,
                how='inner')

            final_nodes = safe_merge(
                rich_nodes,
                filtered_nodes,
                on=self._node,
                how='inner')

            final_nodes = safe_merge(
                final_nodes,
                node_labels_source,
                on=g2._node,
                how='left')

            if node_hop_col in final_nodes and unfiltered_node_labels_source is not None:
                fallback_map = (
                    unfiltered_node_labels_source[[g2._node, node_hop_col]]
                    .drop_duplicates(subset=[g2._node])
                    .set_index(g2._node)[node_hop_col]
                )
                try:
                    final_nodes[node_hop_col] = _combine_first_no_warn(
                        final_nodes[node_hop_col],
                        final_nodes[g2._node].map(fallback_map)
                    )
                except Exception:
                    pass

                try:
                    if final_nodes[node_hop_col].notna().all():
                        final_nodes[node_hop_col] = final_nodes[node_hop_col].astype('int64')
                except Exception:
                    pass

            if label_node_hops is None and node_hop_col in final_nodes:
                final_nodes = final_nodes.drop(columns=[node_hop_col])
        else:
            final_nodes = safe_merge(
                rich_nodes,
                base_nodes,
                on=self._node,
                how='inner')

        g_out = g_out.nodes(final_nodes)

    # Ensure all edge endpoints are present in nodes
    if g_out._edges is not None and len(g_out._edges) > 0 and g_out._nodes is not None:
        endpoints = concat(
            [
                g_out._edges[[g_out._source]].rename(columns={g_out._source: g_out._node}),
                g_out._edges[[g_out._destination]].rename(columns={g_out._destination: g_out._node}),
            ],
            ignore_index=True,
            sort=False,
        ).drop_duplicates(subset=[g_out._node])
        if track_node_hops and node_hop_records is not None and node_hop_col is not None:
            endpoints = safe_merge(
                endpoints,
                node_hop_records[[g_out._node, node_hop_col]].drop_duplicates(subset=[g_out._node]),
                on=g_out._node,
                how='left'
            )
        # Align engine types
        if resolve_engine(EngineAbstract.AUTO, endpoints) != resolve_engine(EngineAbstract.AUTO, g_out._nodes):
            endpoints = df_to_engine(endpoints, resolve_engine(EngineAbstract.AUTO, g_out._nodes))
        g_out = g_out.nodes(
            concat([g_out._nodes, endpoints], ignore_index=True, sort=False).drop_duplicates(subset=[g_out._node])
        )

    if track_node_hops and node_hop_records is not None and node_hop_col is not None and g_out._nodes is not None:
        hop_map = (
            node_hop_records[[g_out._node, node_hop_col]]
            .drop_duplicates(subset=[g_out._node])
            .set_index(g_out._node)[node_hop_col]
        )
        if g_out._node in g_out._nodes.columns and node_hop_col in g_out._nodes.columns:
            try:
                mapped = g_out._nodes[g_out._node].map(hop_map)
                g_out._nodes[node_hop_col] = g_out._nodes[node_hop_col].where(
                    g_out._nodes[node_hop_col].notna(),
                    mapped
                )
            except Exception:
                pass
            seeds_mask = None
            if seeds_provided and not label_seeds and starting_nodes is not None and g_out._node in starting_nodes.columns:
                seed_ids = starting_nodes[[g_out._node]].drop_duplicates()
                seeds_mask = g_out._nodes[g_out._node].isin(seed_ids[g_out._node])
            missing_mask = g_out._nodes[node_hop_col].isna()
            if seeds_mask is not None:
                missing_mask = missing_mask & ~seeds_mask
            if g_out._edges is not None and edge_hop_col is not None and edge_hop_col in g_out._edges.columns:
                edge_map_df = concat(
                    [
                        g_out._edges[[g_out._source, edge_hop_col]].rename(columns={g_out._source: g_out._node}),
                        g_out._edges[[g_out._destination, edge_hop_col]].rename(columns={g_out._destination: g_out._node}),
                    ],
                    ignore_index=True,
                    sort=False,
                )
                if len(edge_map_df) > 0:
                    edge_map = edge_map_df.groupby(g_out._node)[edge_hop_col].min()
                else:
                    edge_map = pd.Series([], dtype='float64')
                mapped_edge_hops = g_out._nodes[g_out._node].map(edge_map)
                if seeds_mask is not None:
                    mapped_edge_hops = mapped_edge_hops.mask(seeds_mask)
                g_out._nodes[node_hop_col] = _combine_first_no_warn(
                    g_out._nodes[node_hop_col],
                    mapped_edge_hops
                )
            if missing_mask.any():
                g_out._nodes.loc[missing_mask, node_hop_col] = g_out._nodes.loc[missing_mask, g_out._node].map(edge_map)
            if seeds_mask is not None:
                zero_seed_mask = seeds_mask & g_out._nodes[node_hop_col].fillna(-1).eq(0)
                g_out._nodes.loc[zero_seed_mask, node_hop_col] = pd.NA
            try:
                g_out._nodes[node_hop_col] = pd.to_numeric(g_out._nodes[node_hop_col], errors='coerce')
                if pd.api.types.is_numeric_dtype(g_out._nodes[node_hop_col]):
                    g_out._nodes[node_hop_col] = g_out._nodes[node_hop_col].astype('Int64')
            except Exception:
                pass

    if (
        not label_seeds
        and seeds_provided
        and g_out._nodes is not None
        and len(g_out._nodes) > 0
        and node_hop_records is not None
        and g_out._node in g_out._nodes.columns
        and starting_nodes is not None
        and g_out._node in starting_nodes.columns
        and node_hop_col is not None
    ):
        seed_mask_all = g_out._nodes[g_out._node].isin(starting_nodes[g_out._node])
        if direction == 'undirected':
            g_out._nodes.loc[seed_mask_all, node_hop_col] = pd.NA
        else:
            seen_nodes = set(node_hop_records[g_out._node].dropna().tolist())
            seed_ids = starting_nodes[g_out._node].dropna().unique().tolist()
            unreached_seed_ids = set(seed_ids) - seen_nodes
            if unreached_seed_ids:
                mask = g_out._nodes[g_out._node].isin(unreached_seed_ids)
                g_out._nodes.loc[mask, node_hop_col] = pd.NA

    if g_out._nodes is not None and (final_output_min is not None or final_output_max is not None):
        try:
            mask = pd.Series(True, index=g_out._nodes.index)
            if node_hop_col is not None and node_hop_col in g_out._nodes.columns:
                if final_output_min is not None:
                    mask = mask & (g_out._nodes[node_hop_col] >= final_output_min)
                if final_output_max is not None:
                    mask = mask & (g_out._nodes[node_hop_col] <= final_output_max)
            endpoint_ids = None
            if g_out._edges is not None:
                endpoint_ids = pd.concat(
                    [
                        g_out._edges[[g_out._source]].rename(columns={g_out._source: g_out._node}),
                        g_out._edges[[g_out._destination]].rename(columns={g_out._destination: g_out._node}),
                    ],
                    ignore_index=True,
                    sort=False,
                ).drop_duplicates(subset=[g_out._node])
                mask = mask | g_out._nodes[g_out._node].isin(endpoint_ids[g_out._node])
            if label_seeds and seeds_provided and starting_nodes is not None and g_out._node in starting_nodes.columns:
                seed_ids = starting_nodes[[g_out._node]].drop_duplicates()
                mask = mask | g_out._nodes[g_out._node].isin(seed_ids[g_out._node])
            g_out = g_out.nodes(g_out._nodes[mask].drop_duplicates(subset=[g_out._node]))
        except Exception:
            pass

    if debugging_hop and logger.isEnabledFor(logging.DEBUG):
        logger.debug('~~~~~~~~~~ HOP OUTPUT ~~~~~~~~~~~')
        logger.debug('nodes:\n%s', g_out._nodes)
        logger.debug('edges:\n%s', g_out._edges)
        logger.debug('======== /HOP =============')
        logger.debug('==========================')

    if (
        return_as_wave_front
        and resolved_min_hops is not None
        and resolved_min_hops >= 1
        and seeds_provided
        and not label_seeds
        and g_out._nodes is not None
        and starting_nodes is not None
        and g_out._node in starting_nodes.columns
    ):
        seed_ids = starting_nodes[[g_out._node]].drop_duplicates()
        seeds_not_reached = seed_ids
        if matches_nodes is not None and g_out._node in matches_nodes.columns:
            seeds_not_reached = seed_ids[~seed_ids[g_out._node].isin(matches_nodes[g_out._node])]
        filtered_nodes = g_out._nodes[~g_out._nodes[g_out._node].isin(seeds_not_reached[g_out._node])]
        g_out = g_out.nodes(filtered_nodes)

    return g_out
