from typing import Dict, Optional

from graphistry.Engine import Engine, df_to_engine
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)


def partition(
    self: Plottable,
    partition_alg: Optional[str] = None,
    partition_params: Optional[Dict] = None,
    partition_key='partition',
    engine: Engine = Engine.PANDAS
):
    """
    Label each node with a partition key. If partition key is already provided, preserve.

    Supports both pandas and cudf:
        Pandas (igraph): Defaults to infomap
        CuDF: Defaults to ecg
    """
    from timeit import default_timer as timer
    start = timer()

    g = self
    if g._nodes is not None and partition_key in g._nodes:
        return g

    if g._nodes is not None:
        g = g.nodes(df_to_engine(g._nodes, engine))
    if g._edges is not None:
        g = g.edges(df_to_engine(g._edges, engine))

    g = g.materialize_nodes(engine=engine)  # type: ignore
    #FIXME why is materialize returning pdf edges for cudf?
    g = g.edges(df_to_engine(g._edges, engine))
    g = g.nodes(df_to_engine(g._nodes, engine))

    if engine == Engine.PANDAS:
        if partition_alg is None:
            partition_alg = 'community_infomap'
        if partition_params is None:
            if partition_alg == 'community_infomap':
                partition_params = {'directed': False}
            else:
                partition_params = {}
    elif engine == Engine.CUDF:
        if partition_alg is None:
            partition_alg = 'ecg'
        if partition_params is None:
            if partition_alg == 'ecg':
                partition_params = {'directed': False}
            else:
                partition_params = {}
    else:
        raise ValueError('Unexpected engine')

    g2 = g
    if g._edge_weight is None:
        g2 = g.edges( g._edges.assign(weight=1.)).bind(edge_weight='weight')

    if engine == Engine.PANDAS:
        g2 = g2.compute_igraph(partition_alg, **partition_params, out_col=partition_key)  # type: ignore
    elif engine == Engine.CUDF:
        g2 = g2.compute_cugraph(partition_alg, **partition_params, out_col=partition_key)

    out = g2.edges(g._edges)
    out._edge_weight = g._edge_weight

    end = timer()
    logger.debug('partition: %s s', end - start)
    return out
