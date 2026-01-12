"""
Graph hop/traversal operations for PyGraphistry

NOTE: Excluded from pyre (.pyre_configuration) - hop() complexity causes hang. Use mypy.
"""
import logging
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
import pandas as pd

from graphistry.Engine import (
    EngineAbstract, df_concat, df_cons, df_to_engine, resolve_engine, s_series, s_to_numeric, s_na, Engine
)
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
from .filter_by_dict import filter_by_dict
from graphistry.Engine import safe_merge
from .typing import DataFrameT
from .util import generate_safe_column_name


logger = setup_logger(__name__)


def query_if_not_none(query: Optional[str], df: DataFrameT) -> DataFrameT:
    if query is None:
        return df
    return df.query(query)


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

    def _domain_unique(series):
        if engine_concrete == Engine.PANDAS:
            return pd.Index(series.dropna().unique())
        return series.dropna().unique()

    def _domain_is_empty(domain) -> bool:
        return domain is None or len(domain) == 0

    def _domain_union(left, right):
        if _domain_is_empty(left):
            return right
        if _domain_is_empty(right):
            return left
        if engine_concrete == Engine.PANDAS and isinstance(left, pd.Index):
            return left.append(right)
        return concat([left, right], ignore_index=True, sort=False).drop_duplicates()
    
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

    FROM_COL = generate_safe_column_name('__gfql_from__', edges_indexed, prefix='__gfql_', suffix='__')
    TO_COL = generate_safe_column_name('__gfql_to__', edges_indexed, prefix='__gfql_', suffix='__')

    def _build_pairs(src_col: str, dst_col: str) -> DataFrameT:
        return edges_indexed[[src_col, dst_col, EDGE_ID]].rename(
            columns={src_col: FROM_COL, dst_col: TO_COL}
        )

    if direction == 'forward':
        pairs = _build_pairs(g2._source, g2._destination)
    elif direction == 'reverse':
        pairs = _build_pairs(g2._destination, g2._source)
    else:
        pairs = concat(
            [_build_pairs(g2._source, g2._destination), _build_pairs(g2._destination, g2._source)],
            ignore_index=True,
            sort=False,
        ).drop_duplicates(subset=[FROM_COL, TO_COL, EDGE_ID])

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
    if track_node_hops:
        node_hop_col = resolve_label_col(label_node_hops, g2._nodes, '_hop')

    wave_front = starting_nodes[[g2._node]][:0]

    matches_nodes = None
    matches_edges = edges_indexed[[EDGE_ID]][:0]

    #richly-attributed subset for dest matching & return-enriching
    if target_wave_front is None:
        base_target_nodes = g2._nodes
    else:
        base_target_nodes = concat([target_wave_front, g2._nodes], ignore_index=True, sort=False).drop_duplicates(subset=[g2._node])
    #TODO precompute src/dst match subset if multihop?

    def _build_allowed_ids(
        base_nodes: DataFrameT,
        match_dict: Optional[dict],
        match_query: Optional[str],
    ) -> Optional[DataFrameT]:
        if match_dict is None and match_query is None:
            return None
        filtered = query_if_not_none(match_query, filter_by_dict(base_nodes, match_dict))
        return filtered[[g2._node]].drop_duplicates()

    allowed_source_ids: Optional[DataFrameT] = None
    if source_node_match is not None or source_node_query is not None:
        source_base_nodes = g2._nodes
        if seeds_provided and not to_fixed_point and resolved_max_hops == 1:
            source_base_nodes = starting_nodes
        allowed_source_ids = _build_allowed_ids(source_base_nodes, source_node_match, source_node_query)

    allowed_dest_ids = _build_allowed_ids(base_target_nodes, destination_node_match, destination_node_query)
    allowed_source_series = allowed_source_ids[g2._node] if allowed_source_ids is not None else None
    allowed_dest_series = allowed_dest_ids[g2._node] if allowed_dest_ids is not None else None
    allowed_target_intermediate = None
    allowed_target_final = None
    if target_wave_front is not None:
        allowed_target_intermediate = base_target_nodes[g2._node]
        allowed_target_final = target_wave_front[[g2._node]].drop_duplicates()[g2._node]

    node_hop_records = None
    edge_hop_records = None
    seen_node_ids = None
    seen_edge_ids = None

    if track_node_hops and label_seeds and node_hop_col is not None:
        seed_nodes = starting_nodes[[g2._node]].drop_duplicates()
        node_hop_records = seed_nodes.assign(**{node_hop_col: 0})
        seen_node_ids = _domain_unique(seed_nodes[g2._node])

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
            logger.debug(
                'wave_front_base:\n%s',
                starting_nodes[[g2._node]] if first_iter else wave_front,
            )

        assert len(wave_front.columns) == 1, "just indexes"
        wave_front_base = starting_nodes[[g2._node]] if first_iter else wave_front
        if allowed_source_series is None:
            wave_front_iter = wave_front_base
        else:
            wave_front_iter = wave_front_base[wave_front_base[g2._node].isin(allowed_source_series)]
        first_iter = False

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP CONTINUE ~~~~~~~~~~~')
            logger.debug('wave_front_iter:\n%s', wave_front_iter)
            
        wavefront_ids = wave_front_iter[g2._node].unique()
        hop_edges = pairs[pairs[FROM_COL].isin(wavefront_ids)]

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('hop_edges basic:\n%s', hop_edges)

        if allowed_target_intermediate is not None:
            has_more_hops_planned = to_fixed_point or resolved_max_hops is None or current_hop < resolved_max_hops
            target_ids = allowed_target_intermediate if has_more_hops_planned else allowed_target_final
            hop_edges = hop_edges[hop_edges[TO_COL].isin(target_ids)]
            if debugging_hop and logger.isEnabledFor(logging.DEBUG):
                logger.debug('hop_edges filtered by target_wave_front:\n%s', hop_edges)

        new_node_ids = hop_edges[[TO_COL]].rename(columns={TO_COL: g2._node}).drop_duplicates()

        if allowed_dest_series is not None:
            new_node_ids = new_node_ids[new_node_ids[g2._node].isin(allowed_dest_series)]
            hop_edges = hop_edges[hop_edges[TO_COL].isin(allowed_dest_series)]
            if debugging_hop and logger.isEnabledFor(logging.DEBUG):
                logger.debug('new_node_ids after precomputed filtering:\n%s', new_node_ids)
                logger.debug('hop_edges filtered by precomputed nodes:\n%s', hop_edges)

        matches_edges = concat(
            [matches_edges, hop_edges[[EDGE_ID]]],
            ignore_index=True,
            sort=False
        ).drop_duplicates(subset=[EDGE_ID])

        if len(new_node_ids) > 0:
            max_reached_hop = current_hop

        if track_edge_hops and edge_hop_col is not None:
            if len(hop_edges) > 0:
                labeled_edges = hop_edges[[EDGE_ID]].assign(**{edge_hop_col: current_hop})
                if edge_hop_records is None:
                    edge_hop_records = labeled_edges
                    seen_edge_ids = _domain_unique(labeled_edges[EDGE_ID])
                else:
                    seen_edge_ids = (
                        seen_edge_ids
                        if seen_edge_ids is not None
                        else _domain_unique(edge_hop_records[EDGE_ID])
                    )
                    if _domain_is_empty(seen_edge_ids):
                        new_edge_labels = labeled_edges
                    else:
                        new_mask = ~labeled_edges[EDGE_ID].isin(seen_edge_ids)
                        new_edge_labels = labeled_edges[new_mask]
                    if len(new_edge_labels) > 0:
                        edge_hop_records = concat(
                            [edge_hop_records, new_edge_labels],
                            ignore_index=True,
                            sort=False
                        ).drop_duplicates(subset=[EDGE_ID])
                        new_edge_ids = _domain_unique(new_edge_labels[EDGE_ID])
                        seen_edge_ids = _domain_union(seen_edge_ids, new_edge_ids)

        if track_node_hops and node_hop_col is not None:
            if node_hop_records is None:
                node_hop_records = new_node_ids.assign(**{node_hop_col: current_hop})
                seen_node_ids = _domain_unique(node_hop_records[g2._node])
            else:
                seen_node_ids = (
                    seen_node_ids
                    if seen_node_ids is not None
                    else _domain_unique(node_hop_records[g2._node])
                )
                if _domain_is_empty(seen_node_ids):
                    new_node_labels = new_node_ids
                else:
                    new_mask = ~new_node_ids[g2._node].isin(seen_node_ids)
                    new_node_labels = new_node_ids[new_mask]
                if len(new_node_labels) > 0:
                    node_hop_records = concat(
                        [node_hop_records, new_node_labels.assign(**{node_hop_col: current_hop})],
                        ignore_index=True,
                        sort=False
                    ).drop_duplicates(subset=[g2._node])
                    new_node_ids_domain = _domain_unique(new_node_labels[g2._node])
                    seen_node_ids = _domain_union(seen_node_ids, new_node_ids_domain)

        if debugging_hop and logger.isEnabledFor(logging.DEBUG):
            logger.debug('~~~~~~~~~~ LOOP STEP MERGES 1 ~~~~~~~~~~~')
            logger.debug('matches_edges:\n%s', matches_edges)
            logger.debug('matches_nodes:\n%s', matches_nodes)
            logger.debug('new_node_ids:\n%s', new_node_ids)
            logger.debug('hop_edges:\n%s', hop_edges)

        # When !return_as_wave_front, include starting nodes in returned matching node set
        # (When return_as_wave_front, skip starting nodes, just include newly reached)
        # Only need to do this in the first loop step
        if matches_nodes is None:  # first iteration
            if return_as_wave_front:
                matches_nodes = new_node_ids[:0]
            else:
                matches_nodes = hop_edges[[FROM_COL]].rename(
                    columns={FROM_COL: g2._node}
                ).drop_duplicates(subset=[g2._node])

            if debugging_hop and logger.isEnabledFor(logging.DEBUG):
                logger.debug('~~~~~~~~~~ LOOP STEP MERGES 2 ~~~~~~~~~~~')
                logger.debug('matches_edges:\n%s', matches_edges)

        if len(matches_nodes) > 0:
            combined_node_ids = concat([matches_nodes, new_node_ids], ignore_index=True, sort=False).drop_duplicates()
        else:
            combined_node_ids = new_node_ids

        if len(combined_node_ids) == len(matches_nodes):
            # fixedpoint, exit early: future will come to same spot
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
                node_labels_source.loc[~node_mask, node_hop_col] = s_na(engine_concrete)

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
                    # Engine-agnostic empty series
                    SeriesCls = s_series(engine_concrete)
                    edge_map = SeriesCls([], dtype='float64')
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
                g_out._nodes.loc[zero_seed_mask, node_hop_col] = s_na(engine_concrete)
            try:
                # Engine-agnostic numeric conversion
                to_numeric = s_to_numeric(engine_concrete)
                g_out._nodes[node_hop_col] = to_numeric(g_out._nodes[node_hop_col], errors='coerce')
                # Check if numeric and convert to nullable int
                col = g_out._nodes[node_hop_col]
                if hasattr(col, 'dtype') and hasattr(col.dtype, 'kind') and col.dtype.kind in ('i', 'f'):
                    g_out._nodes[node_hop_col] = col.astype('Int64')
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
            g_out._nodes.loc[seed_mask_all, node_hop_col] = s_na(engine_concrete)
        else:
            # Vectorized: find seed nodes not in seen nodes
            seen_nodes_series = node_hop_records[g_out._node].dropna()
            seed_ids_series = starting_nodes[g_out._node].dropna()
            # unreached = seeds that are NOT in seen_nodes
            unreached_mask = ~seed_ids_series.isin(seen_nodes_series)
            unreached_seed_ids = seed_ids_series[unreached_mask]
            if len(unreached_seed_ids) > 0:
                mask = g_out._nodes[g_out._node].isin(unreached_seed_ids)
                g_out._nodes.loc[mask, node_hop_col] = s_na(engine_concrete)

    if g_out._nodes is not None and (final_output_min is not None or final_output_max is not None):
        try:
            # Engine-agnostic constant True series - scalar broadcast, no Python list
            SeriesCls = s_series(engine_concrete)
            mask = SeriesCls(True, index=g_out._nodes.index)
            if node_hop_col is not None and node_hop_col in g_out._nodes.columns:
                if final_output_min is not None:
                    mask = mask & (g_out._nodes[node_hop_col] >= final_output_min)
                if final_output_max is not None:
                    mask = mask & (g_out._nodes[node_hop_col] <= final_output_max)
            endpoint_ids = None
            if g_out._edges is not None:
                endpoint_ids = concat(
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
