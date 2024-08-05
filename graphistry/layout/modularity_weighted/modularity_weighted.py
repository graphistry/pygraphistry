from typing import Any, Dict, Literal, Optional
import pandas as pd

from graphistry.Engine import Engine, EngineAbstract, resolve_engine
from graphistry.Plottable import Plottable


def modularity_weighted_layout(
  g: Plottable,
  community_col: Optional[str] = None,
  community_alg: Optional[str] = None,
  community_params: Optional[Dict[str, Any]] = None,
  same_community_weight: float = 2.0,
  cross_community_weight: float = 0.3,
  edge_influence: float = 2.0,
  engine: EngineAbstract = EngineAbstract.AUTO
) -> Plottable:
    """

    Compute a modularity-weighted layout, where edges are weighted based on whether they connect nodes in the same community or different communities.

    Computes the community if not provided, including with GPU acceleration, using Louvain

    :param g: input graph
    :param community_col: column in nodes with community labels
    :param community_alg: community detection algorithm, e.g., 'louvain' or 'community_multilevel'
    :param community_params: parameters for community detection algorithm
    :param same_community_weight: weight for edges connecting nodes in the same community
    :param cross_community_weight: weight for edges connecting nodes in different communities
    :param edge_influence: influence of edge weights on layout
    :param engine: graph engine, e.g., 'pandas', 'cudf', 'auto'. CPU uses igraph algorithms, and GPU, cugraph
    :return: graph with layout
 
    **Example: Basic**

        ::

            g = g.modularity_weighted_layout()
            g.plot()

    **Example: Use existing community labels**
    
            ::
    
                assert 'my_community' in g._nodes.columns
                g = g.modularity_weighted_layout(community_col='my_community')
                g.plot()

    **Example: Use GPU-accelerated Louvain algorithm**

        ::

            g = g.modularity_weighted_layout(community_alg='louvain', engine='cudf')
            g = g.modularity_weighted_layout(community_alg='community_multilevel', engine='pandas')

                
    **Example: Use custom layout settings**

        ::

            g = g.modularity_weighted_layout(
                community_col='community',
                same_community_weight=2.0,
                cross_community_weight=0.3,
                edge_influence=2.0
            )
            g.plot()

    """
    assert g._edges is not None, 'Expected edges to be set'
    if community_col is None:
        g = g.materialize_nodes()
        engine_concrete = resolve_engine(engine, g)
        if community_alg is None:
            if engine_concrete == Engine.PANDAS:
                community_alg = 'community_multilevel'
            else:
                community_alg = 'louvain'
            if community_params is None:
                community_params = {'directed': False}
        community_params = community_params or {}
        if engine_concrete == Engine.PANDAS:
            g = g.compute_igraph(community_alg, **community_params)  # type: ignore
        elif engine_concrete == Engine.CUDF:
            import cudf
            if not isinstance(g._edges, cudf.DataFrame):
                assert isinstance(g._edges, pd.DataFrame), f'Expected edges to be cudf or pandas, got: {type(g._edges)}'
                g = g.edges(cudf.DataFrame(g._edges))
            g = g.compute_cugraph(community_alg, **community_params)  # type: ignore
        else:
            raise ValueError(f'Unsupported engine: {engine}')
        community_col = community_alg
    else:
        assert community_col in g._nodes, f'Expected community column {community_col} in nodes, only available are {g._nodes.columns}'

    g = g.layout_settings(edge_influence=edge_influence)

    assert 'source_community' not in g._edges, 'Expected no source_community column in edges'
    assert 'destination_community' not in g._edges, 'Expected no destination_community column in edges'

    edges = (g._edges
        .merge(g._nodes[[community_col, g._node]], left_on=g._source, right_on=g._node).rename(columns={community_col: 'source_community'}).drop(columns=[g._node])
        .merge(g._nodes[[community_col, g._node]], left_on=g._destination, right_on=g._node).rename(columns={community_col: 'destination_community'}).drop(columns=[g._node])
    )

    same_community = (edges['source_community'] == edges['destination_community'])
    edges = edges.assign(
        weight=same_community.map({
            True: same_community_weight,
            False: cross_community_weight
        }),
        same_community=same_community
    )
    edges = edges.drop(columns=['source_community', 'destination_community'])

    return g.edges(edges).bind(edge_weight='weight')
