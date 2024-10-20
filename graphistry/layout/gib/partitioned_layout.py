from typing import Any, Callable, Dict, Optional, Union
import numpy as np, pandas as pd
from timeit import default_timer as timer

from graphistry.Engine import Engine, df_concat, df_to_pdf, df_cons
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger

from .layout_bulk import layout_bulk_mode
from .layout_non_bulk import layout_non_bulk_mode


logger = setup_logger(__name__)


#TODO layout engine may be diff from base engine!
def partitioned_layout(
    self: Plottable,
    partition_offsets: Dict[str, Dict[int, float]],
    layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]] = None,
    layout_params: Optional[Dict[str, Any]] = None,
    partition_key='partition',
    bulk_mode: bool = True,
    engine: Engine = Engine.PANDAS,
) -> Plottable:
    """    
    :param partition_offsets: {'dx', 'dy', 'x', 'y'} => <partition> => float
    :type partition_offsets: Dict[str, Dict[int, float]]
    :param layout_alg: Layout algorithm to be applied if partition_key column does not already exist; GPU defaults to fa2_layout, CPU defaults to igraph fr
    :type layout_alg: Optional[Union[str, Callable[[Plottable], Plottable]]]
    :param layout_params: Parameters for the layout algorithm
    :type layout_params: Optional[Dict[str, Any]]
    :param partition_key: The partition key; defaults to the layout_alg
    :type partition_key: str
    :param bulk_mode: Whether to apply layout in bulk mode
    :type bulk_mode: bool
    :param engine: The engine being used (Pandas or CUDF)
    :type engine: Engine

    :return: The resulting Plottable object with positioned nodes
    """
    start = timer()

    node_partitions = []
    # edgeless_partitions = None

    g_degrees = self.get_degrees()

    # | partition , id_count, degree_max |
    node_splits_info = (
        g_degrees._nodes[[self._node, 'degree', partition_key]]
        .groupby(partition_key).agg({
            g_degrees._node: 'count',
            'degree': 'max'
        })
        .rename(columns={g_degrees._node: 'id_count', 'degree': 'degree_max'})
        .reset_index()
    )

    singleton_nodes = (
        node_splits_info[node_splits_info['id_count'] == 1][[partition_key]]
        .merge(g_degrees._nodes, on=partition_key, how='left')
    )
    remaining = node_splits_info[node_splits_info['id_count'] > 1]

    pair_nodes = (
        node_splits_info[node_splits_info['id_count'] == 2][[partition_key]]
        .merge(g_degrees._nodes, on=partition_key, how='left')
    )
    pre_n = len(remaining)
    remaining = remaining[remaining['id_count'] > 2]
    logger.debug('pruned pairs: %s -> %s', pre_n, len(remaining))

    #incomplete: misses groups that have only out-of-group edges!
    edgeless_nodes = (
        remaining[remaining['degree_max'] == 0][[partition_key]]
        .merge(g_degrees._nodes, on=partition_key, how='left')
    )
    remaining = remaining[remaining['degree_max'] > 0]

    #nodes = g_degrees
    nodes = g_degrees._nodes.merge(
        node_splits_info[[partition_key, 'id_count', 'degree_max']],
        on=partition_key,
        how='left'
    )
    end_stats = timer()
    logger.debug('partition stats: %s s', end_stats - start)

    self_selfless = self.edges(
        self._edges[
            self._edges[self._source] != self._edges[self._destination]
        ]
    )

    if engine == Engine.CUDF and self_selfless._edge_weight is None:
        self_selfless = (
            self_selfless.edges(
                self_selfless._edges.assign(weight=1.)
            ).bind(edge_weight='weight')
        )

    if bulk_mode:
        combined_nodes = layout_bulk_mode(self, nodes, partition_key, layout_alg, layout_params, engine)
        node_partitions.append(combined_nodes)
    else:
        node_partitions, s_layout, s_keep, s_layout_by_size = layout_non_bulk_mode(
            self, node_partitions, remaining, partition_key, layout_alg, layout_params, engine, self_selfless
        )

    end_communities = timer()  # Define end_communities here to track layout time
    logger.debug('part_layout time: %s s', end_communities - start)

    if True and len(singleton_nodes) > 0:
        logger.debug('# SINGLETONS: %s', len(singleton_nodes))
        start_sing = timer()
        singletons = singleton_nodes.assign(
            x=0.5,
            y=0.5,
            type='singleton'
        )
        node_partitions.append(singletons)
        end_sing = timer()
        logger.debug('singleton groups (%s): %s s', len(singletons), end_sing - start_sing)

    if True and len(pair_nodes) > 0:
        logger.debug('# PAIRS: %s', len(pair_nodes))
        start_pair = timer()
        pairs_indexed = pair_nodes.reset_index()
        pairs = pairs_indexed.assign(
            x = 0.33 + (pairs_indexed['index'] % 2) * 0.33,
            y = 0.33 + (pairs_indexed['index'] % 2) * 0.33
        ).drop(columns=['index'])
        node_partitions.append(pairs)
        end_pair = timer()
        logger.debug('pairs groups (%s): %s s', len(pairs), end_pair - start_pair)

    #FIXME: how to make safe?
    if True and len(edgeless_nodes) > 0:
        logger.debug('# EDGELESS: %s', len(edgeless_nodes))
        start_e = timer()
        edgeless = edgeless_nodes
        # FIXME: Sorted grid vs random
        if engine == Engine.PANDAS:
            edgeless['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(edgeless)), dtype='float32')
            edgeless['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(edgeless)), dtype='float32')
        elif engine == Engine.CUDF:
            import cudf, cupy as cp
            edgeless['x'] = cudf.Series(cp.random.rand(len(edgeless), dtype=cp.float32))
            edgeless['y'] = cudf.Series(cp.random.rand(len(edgeless), dtype=cp.float32))
        else:
            raise ValueError('Unknown engine, expected Pandas or CuDF')
        edgeless['type'] = 'singleton'
        node_partitions.append(edgeless)
        end_e = timer()
        logger.debug('edgeless-community (%s): %s s', len(edgeless), end_e - start_e)

    combined_nodes = df_concat(engine)(node_partitions, ignore_index=True, sort=False)
    # FA unnconnected nodes, though circle would autoplace
    updates = {}
    if engine == Engine.PANDAS:
        if combined_nodes.x.isna().any():
            logger.debug('filling layout-returned NAs as random: %s xs', combined_nodes.x.isna().sum())
            assert combined_nodes.x.isna().sum() == 0
            updates['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(combined_nodes)), dtype='float32')
        if combined_nodes.y.isna().any():
            logger.debug('filling layout-returned NAs as random: %s ys', combined_nodes.y.isna().sum())
            assert combined_nodes.y.isna().sum() == 0
            updates['y'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(combined_nodes)), dtype='float32')
    elif engine == Engine.CUDF:
        import cudf, cupy as cp
        if combined_nodes.x.isna().any():
            logger.debug('filling layout-returned NAs as random: %s xs', combined_nodes.x.isna().sum())
            assert combined_nodes.x.isna().sum() == 0
            updates['x'] = cudf.Series(cp.random.rand(len(combined_nodes), 1, dtype=cp.float32))
        if combined_nodes.y.isna().any():
            logger.debug('filling layout-returned NAs as random: %s ys', combined_nodes.y.isna().sum())
            assert combined_nodes.y.isna().sum() == 0
            updates['y'] = cudf.Series(cp.random.rand(len(combined_nodes), 1, dtype=cp.float32))
    else:
        raise ValueError('Unknown engine, expected Pandas or CuDF')
    if len(updates.keys()) > 0:
        combined_nodes = combined_nodes.fillna(updates)

    node_stats = combined_nodes.groupby(partition_key).agg({
        'x': ['max', 'min'],
        'y': ['max', 'min']
    })
    node_stats.columns = ['x_max', 'x_min', 'y_max', 'y_min']  # type: ignore
    node_stats['dx'] = df_cons(engine)({
        'dx': node_stats['x_max'] - node_stats['x_min'],
        'min': 1
    }).max(axis=1)
    node_stats['dy'] = df_cons(engine)({
        'dy': node_stats['y_max'] - node_stats['y_min'],
        'min': 1
    }).max(axis=1)

    # {<col_name> -> {<partition> -> float}}
    partition_stats = df_to_pdf(node_stats, engine).to_dict()
    normalized_nodes = combined_nodes.copy()
    normalized_nodes['x'] = (
        combined_nodes['x']
        - combined_nodes[partition_key].map(partition_stats['x_min'])  # noqa: W503
    ) / combined_nodes[partition_key].map(partition_stats['dx'])
    normalized_nodes['y'] = (
        combined_nodes['y']
        - combined_nodes[partition_key].map(partition_stats['y_min'])  # noqa: W503
    ) / combined_nodes[partition_key].map(partition_stats['dy'])
    g_locally_positioned = self.nodes(normalized_nodes)

    global_nodes = g_locally_positioned._nodes.copy()
    global_nodes['x'] = (
        (
            g_locally_positioned._nodes['x']
            * g_locally_positioned._nodes[partition_key].map(partition_offsets['dx'])  # noqa: W503
        )
        + g_locally_positioned._nodes[partition_key].map(partition_offsets['x'])  # noqa: W503
    )
    global_nodes['y'] = (
        (
            g_locally_positioned._nodes['y']
            * g_locally_positioned._nodes[partition_key].map(partition_offsets['dy'])  # noqa: W503
        )
        + g_locally_positioned._nodes[partition_key].map(partition_offsets['y'])  # noqa: W503
    )
    global_nodes['y'] = -global_nodes['y']
    g_globally_positioned = g_locally_positioned.nodes(global_nodes)
    g_globally_positioned._edge_weight = self._edge_weight
    end = timer()
    logger.debug('part_layout postproc: %s s', end - end_communities)
    logger.debug('partitioned_layout total: %s s', end - start)
    return g_globally_positioned
