import pandas as pd
from typing import Any, Dict, List, Optional, Union
from typing_extensions import Literal
from graphistry.constants import NODE
from graphistry.Plottable import Plottable
from graphistry.plugins_types import CuGraphKind
from graphistry.util import setup_logger
logger = setup_logger(__name__)



#import logging
#logger.setLevel(logging.DEBUG)


# preferring igraph naming convetions over graphistry.constants
SRC_CUGRAPH = 'source'
DST_CUGRAPH = 'target'
NODE_CUGRAPH = NODE


def df_to_gdf(df: Any):
    import cudf
    if isinstance(df, cudf.DataFrame):
        return df
    elif isinstance(df, pd.DataFrame):
        return cudf.from_pandas(df)
    else:
        raise ValueError('Unsupported data type, expected pd.DataFrame or cudf.DataFrame, got: %s', type(df))


def from_cugraph(self,
    G,
    node_attributes: Optional[List[str]] = None,
    edge_attributes: Optional[List[str]] = None,
    load_nodes: bool = True, load_edges: bool = True,
    merge_if_existing: bool = True
) -> Plottable:
    """
    
    If bound IDs, use the same IDs in the returned graph.

    If non-empty nodes/edges, instead of returning G's topology, use existing topology and merge in G's attributes
    
    """

    import cudf, cugraph

    g = self

    ####

    src = self._source or SRC_CUGRAPH
    dst = self._destination or DST_CUGRAPH
    edges_gdf = G.view_edge_list()  # src, dst

    if g._nodes is not None and load_nodes:
        raise NotImplementedError('cuGraph does not support nodes tables; set g.nodes(None) or from_cugraph(load_nodes=False)')

    if src != 'src':
        edges_gdf = edges_gdf.rename(columns={'src': src})
    if dst != 'dst':
        edges_gdf = edges_gdf.rename(columns={'dst': dst})

    if g._edges is not None and g._source is not None and g._destination is not None and merge_if_existing:

        if isinstance(g._edges, cudf.DataFrame):
            g_indexed_df = g._edges
        elif isinstance(g._edges, pd.DataFrame):
            g_indexed_df = cudf.from_pandas(g._edges)

        else:
            was_dask = False
            try:
                import dask_cudf
                if isinstance(g._edges, dask_cudf.DataFrame):
                    was_dask = True
                    g_indexed_df = g._edges.compute()
            except ImportError:
                1

            if not was_dask:
                raise ValueError('Unsupported edge data type, expected pd/cudf.DataFrame, got: %s', type(g._edges))

        #TODO handle case where where 3 attrs and third is _edge        
        if len(g_indexed_df.columns) == 2 and (len(g_indexed_df) == len(edges_gdf)):
            #opt: skip merge: no old columns
            1
        elif ((len(edges_gdf.columns) == 2) or len(edges_gdf.columns) == 0) and (len(g._edges) == len(edges_gdf)):
            #opt: skip merge: no new columns
            edges_gdf = df_to_gdf(g._edges)
        else:
            if len(g_indexed_df) != len(edges_gdf):
                logger.warning('edge tables do not match in length; switch merge_if_existing to False or load_edges to False or add missing edges')
            g_edges_trimmed = df_to_gdf(g_indexed_df)[[x for x in g_indexed_df if x not in edges_gdf or x in [src, dst]]]
            if g._source != 'src':
                edges_gdf = edges_gdf.rename(columns={'src': src})
            if g._destination != 'dst':
                edges_gdf = edges_gdf.rename(columns={'dst': dst})
            edges_gdf = edges_gdf.merge(g_edges_trimmed, how='left', on=[src, dst])

    g = g.edges(edges_gdf, src, dst)

    return g

def to_cugraph(self: Plottable, 
    directed: bool = True,
    include_nodes: bool = True,
    node_attributes: Optional[List[str]] = None,
    edge_attributes: Optional[List[str]] = None,
    kind : CuGraphKind = 'Graph'
):

    import cudf, cugraph
    if kind == 'Graph':
        CG = cugraph.Graph
    elif kind == 'MultiGraph':
        CG = cugraph.MultiGraph
    elif kind == 'BiPartiteGraph':
        CG = cugraph.BiPartiteGraph

    G = CG(directed=directed)
    
    opts = {
        'source': self._source,
        'destination': self._destination,
    }
    if self._edge_weight is not None:
        opts['edge_attr'] = self._edge_weight

    if self._edges is None:
        raise ValueError('No edges loaded')
    elif isinstance(self._edges, cudf.DataFrame):
        G.from_cudf_edgelist(self._edges, **opts)
    elif isinstance(self._edges, pd.DataFrame):
        G.from_pandas_edgelist(self._edges, **opts)
    else:
        was_dask = False
        try:
            import dask_cudf
            if isinstance(self._edges, dask_cudf.DataFrame):
                was_dask = True
                G.from_dask_cudf_edgelist(self._edges, **opts)
        except ImportError:
            1
        if not was_dask:
            raise ValueError('Unsupported edge data type, expected pd/cudf.DataFrame, got: %s', type(self._edges))

    if self._node is not None and self._nodes is not None:
        nodes_gdf = self._nodes if isinstance(self._nodes, cudf.DataFrame) else cudf.from_pandas(self._nodes)
        G.add_nodes_from(nodes_gdf[self._node])

    return G


#'vertex' +
node_compute_algs_to_attr : Dict[str, Union[str, List[str]]] = {
    'betweenness_centrality': 'betweenness_centrality',
    #'degree_centrality': 'degree_centrality',
    'katz_centrality': 'katz_centrality',
    'ecg': 'partition',
    'leiden': 'partition',
    'louvain': 'partition',
    'spectralBalancedCutClustering': 'cluster',
    'spectralModularityMaximizationClustering': 'cluster',
    'connected_components': 'labels',
    'strongly_connected_components': 'labels',
    'weakly_connected_components': 'labels',
    'core_number': 'core_number',
    #'k_core': 'values',
    'hits': ['hubs', 'authorities'],
    'pagerank': 'pagerank',
    #random_walks
    'bfs': ['distance', 'predecessor'],
    'bfs_edges': ['distance', 'predecessor'],
    'sssp': ['distance', 'predecessor'],
    'shortest_path': ['distance', 'predecessor'],
    'shortest_path_length': 'distance',
}

#'src'/'dst' or 'source'/'destination' +
edge_compute_algs_to_attr = {
    'batched_ego_graphs': 'unknown',
    'edge_betweenness_centrality': 'edge_betweenness_centrality',
    'jaccard': 'jaccard_coeff',
    'jaccard_w': 'jaccard_coeff',
    'overlap': 'overlap_coeff',
    'overlap_coefficient': 'overlap_coefficient',
    'overlap_w': 'overlap_coeff',
    'sorensen': 'sorensen_coeff',
    'sorensen_coefficient': 'sorensen_coeff',
    'sorensen_w': 'sorensen_coeff',

}
graph_compute_algs = [
    'ego_graph',
    #'k_truss',  # not implemented in CUDA 11.4 for 22.04
    'k_core',
    #'ktruss_subgraph',  # not implemented in CUDA 11.4 for 22.04
    #'subgraph',  # unclear why not working
    'minimum_spanning_tree'
]

compute_algs = list(node_compute_algs_to_attr.keys()) + list(edge_compute_algs_to_attr.keys()) + graph_compute_algs

def compute_cugraph(
    self: Plottable,
    alg: str, out_col: Optional[str] = None, params: dict = {},
    kind : CuGraphKind = 'Graph', directed = True,
    G: Optional[Any] = None
) -> Plottable:

    import cugraph

    if G is None:
        G = to_cugraph(self, kind=kind, directed=directed)

    if alg in node_compute_algs_to_attr:
        out = getattr(cugraph, alg)(G, **params)
        if isinstance(out, tuple):
            out = out[0]
        g = self.materialize_nodes(engine='cudf')
        if g._node != 'vertex':
            out = out.rename(columns={'vertex': g._node})
        expected_cols = node_compute_algs_to_attr[alg]
        if not isinstance(expected_cols, list):
            expected_cols = [expected_cols]
        if out_col is not None:
            if len(expected_cols) > 1:
                raise ValueError('Multiple columns returned, but out_col (singleton) was specified')
            out = out.rename(columns={expected_cols[0]: out_col})
        nodes_gdf = g._nodes.merge(out, how='left', on=g._node)
        return g.nodes(nodes_gdf)
    elif alg in edge_compute_algs_to_attr:
        out = getattr(cugraph, alg)(G, **params)
        if isinstance(out, tuple):
            out = out[0]
        if 'source' in out.columns:
            out = out.rename(columns={'source': 'src', 'destination': 'dst'})
        g = self
        if g._source != 'src':
            out = out.rename(columns={'src': g._source})
        if g._destination != 'dst':
            out = out.rename(columns={'dst': g._destination})
        expected_cols = edge_compute_algs_to_attr[alg]
        if not isinstance(expected_cols, list):
            expected_cols = [expected_cols]
        if out_col is not None:
            if len(expected_cols) > 1:
                raise ValueError('Multiple columns returned, but out_col (singleton) was specified')
            out = out.rename(columns={expected_cols[0]: out_col})
        edges_gdf = g._edges.merge(out, how='left', on=[g._source, g._destination])
        return g.edges(edges_gdf)
    elif alg in graph_compute_algs:
        out = getattr(cugraph, alg)(G, **params)
        if isinstance(out, tuple):
            out = out[0]
        if out_col is not None:
            raise ValueError('Graph returned, but out_col was specified')
        return from_cugraph(self, out, load_nodes=False)

    raise ValueError('Unsupported algorithm: %s', alg)



layout_algs = [
    'force_atlas2'
]

def layout_cugraph(
    self: Plottable,
    layout: str, params: dict = {},
    kind : CuGraphKind = 'Graph', directed = True,
    G: Optional[Any] = None
) -> Plottable:
   
    import cugraph

    g = self.materialize_nodes(engine='cudf')

    if layout not in layout_algs:
        raise ValueError('Unsupported algorithm: %s', layout)

    if G is None:
        G = to_cugraph(self, kind=kind, directed=directed)

    out = getattr(cugraph, layout)(G, **params)
    if 'source' in out.columns:
        out = out.rename(columns={'source': 'src', 'destination': 'dst'})

    if g._node != 'vertex':
        out = out.rename(columns={'vertex': g._node})

    nodes_gdf = g._nodes.merge(out, how='left', on=g._node)
    return g.nodes(nodes_gdf)
