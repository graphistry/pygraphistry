import numpy as np, pandas as pd
from typing import Dict, List, Optional

from graphistry.Engine import Engine, df_concat, df_to_pdf, df_cons
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


#TODO layout engine may be diff from base engine!
def partitioned_layout(
    self: Plottable,
    partition_offsets: Dict[str, Dict[int, float]],
    layout_alg: Optional[str] = None,
    layout_params: Dict = {},
    partition_key='partition',
    engine: Engine = Engine.PANDAS
) -> 'Plottable':
    from timeit import default_timer as timer
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

    #for partition in remaining[partition_key].to_pandas().to_numpy():
    s_keep = 0.
    s_layout = 0.
    s_layout_by_size = {}
    for partition in remaining[partition_key].to_numpy():
        start_i = timer()

        node_ids = self._nodes[
            self._nodes[partition_key] == partition
        ][self._node]
        subgraph_g = self_selfless.nodes(nodes).keep_nodes({self._node: node_ids})
        start_i_mid = timer()
        s_keep += start_i_mid - start_i
        #print('node shape', subgraph_g._nodes.shape, 'edge shape', subgraph_g._edges.shape)
        #if len(subgraph_g._edges) == 0:
        #    print('EDGELESS!')
        #elif len(subgraph_g._nodes) == 1:
        #    print('SINGLETON')
        start_i_mid = timer()
        niter = min(len(subgraph_g._nodes), 300)
        if engine == Engine.PANDAS:
            layout_name = layout_alg or 'fr'
            positioned_subgraph_g = subgraph_g.layout_igraph(  # type: ignore
                layout=layout_name,
                params={
                    **({'niter': niter} if layout_name == 'fr' else {}),
                    **(layout_params or {})
                }
            )
        elif engine == Engine.CUDF:
            layout_name = layout_alg or 'force_atlas2'
            positioned_subgraph_g = subgraph_g.layout_cugraph(
                layout=layout_name,
                params={
                    **({'max_iter': niter} if layout_name == 'force_atlas2' else {}),
                    **(layout_params or {})
                }
            )
        positioned_subgraph_g = positioned_subgraph_g.nodes(
            positioned_subgraph_g._nodes.assign(
                type=layout_name,
                #max_iter=min(len(subgraph_g._nodes), 500),
                #subg_n=len(subgraph_g._nodes),
                #subg_e=len(subgraph_g._edges)
            )
        )
        #if positioned_subgraph_g._nodes.x.isna().any():
        #    # logger.debug('NA vals: cugraph fa2 is nan for unconnected nodes')
        #    #print(positioned_subgraph_g._edges[[
        #    #    positioned_subgraph_g._source,
        #    #    positioned_subgraph_g._destination,
        #    #    positioned_subgraph_g._edge_weight
        #    #]])
        #    #na_fa_graphs.append(positioned_subgraph_g)
        node_partitions.append(positioned_subgraph_g._nodes)
        end_i = timer()
        #print('start_i layout', end_i - start_i_mid, 's')
        s_layout += end_i - start_i_mid
        if len(positioned_subgraph_g._nodes) not in s_layout_by_size:
            s_layout_by_size[ len(positioned_subgraph_g._nodes) ] = (0, 0.)
        n, t = s_layout_by_size[ len(positioned_subgraph_g._nodes) ]
        s_layout_by_size[ len(positioned_subgraph_g._nodes) ] = (n + 1, t + (end_i - start_i_mid))
    end_communities = timer()
    logger.debug('s_keep: %s s', s_keep)
    logger.debug('s_layout: %s s', s_layout)
    logger.debug('s_layout_by_size: %s s', s_layout_by_size)
    #print('all sub communities', len(subgraph_g._nodes), ':', end_communities - end_stats, 's')

    if len(singleton_nodes) > 0:
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

    if len(pair_nodes) > 0:
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
    if len(edgeless_nodes) > 0:
        logger.debug('# EDGELESS: %s', len(edgeless_nodes))
        start_e = timer()
        edgeless = edgeless_nodes
        # FIXME: Sorted grid vs random
        if engine == Engine.PANDAS:
            edgeless['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(edgeless)), dtype='float32')
            edgeless['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(edgeless)), dtype='float32')
        elif engine == Engine.CUDF:
            import cudf, cupy as cp
            edgeless['x'] = cudf.Series(cp.random.rand(len(edgeless), 1, dtype=cp.float32))
            edgeless['y'] = cudf.Series(cp.random.rand(len(edgeless), 1, dtype=cp.float32))
        else:
            raise ValueError('Unknown engine, expected Pandas or CuDF')
        edgeless['type'] = 'singleton'
        node_partitions.append(edgeless)
        end_e = timer()
        logger.debug('edgeless-community (%s): %s s', len(edgeless), end_e - start_e)

    combined_nodes = df_concat(engine)(node_partitions, ignore_index=True, sort=False)
    # FA unnconnected nodes
    updates = {}
    if engine == Engine.PANDAS:
        if combined_nodes.x.isna().any():
            updates['x'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(combined_nodes)), dtype='float32')
        if combined_nodes.y.isna().any():
            updates['y'] = pd.Series(np.random.default_rng().uniform(0., 1., size=len(combined_nodes)), dtype='float32')
    elif engine == Engine.CUDF:
        import cudf, cupy as cp
        if combined_nodes.x.isna().any():
            updates['x'] = cudf.Series(cp.random.rand(len(combined_nodes), 1, dtype=cp.float32))
        if combined_nodes.y.isna().any():
            updates['y'] = cudf.Series(cp.random.rand(len(combined_nodes), 1, dtype=cp.float32))
    else:
        raise ValueError('Unknown engine, expected Pandas or CuDF')
    if len(updates.keys()) > 0:
        combined_nodes = combined_nodes.fillna(updates)

    node_stats = combined_nodes.groupby(partition_key).agg({
        'x': ['max', 'min'],
        'y': ['max', 'min']
    })
    node_stats.columns = ['x_max', 'x_min', 'y_max', 'y_min']
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
