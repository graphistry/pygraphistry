import igraph, pandas as pd
from typing import Any, List, Optional
from graphistry.constants import NODE
from graphistry.Plottable import Plottable
from graphistry.util import setup_logger
logger = setup_logger(__name__)

#import logging
#logger.setLevel(logging.DEBUG)


# preferring igraph naming convetions over graphistry.constants
SRC_IGRAPH = 'source'
DST_IGRAPH = 'target'
NODE_IGRAPH = NODE

def from_igraph(self,
    ig,
    node_attributes: Optional[List[str]] = None,
    edge_attributes: Optional[List[str]] = None,
    load_nodes: bool = True, load_edges: bool = True,
    merge_if_existing: bool = True
) -> Plottable:
    """
    Convert igraph object into Plotter

    If base g has _node, _source, _destination definitions, use them

    When merge_if_existing with preexisting nodes/edges df and shapes match ig, combine attributes

    For merge_if_existing to work with edges, must set g._edge and have corresponding edge index attribute in igraph.Graph 
    
    :param ig: Source igraph object
    :type ig: igraph

    :param node_attributes: Subset of node attributes to load; None means all (default)
    :type node_attributes: Optional[List[str]]

    :param edge_attributes: Subset of edge attributes to load; None means all (default)
    :type edge_attributes: Optional[List[str]]

    :param load_nodes: Whether to load nodes dataframe (default True)
    :type load_nodes: bool

    :param load_edges: Whether to load edges dataframe (default True)
    :type load_edges: bool

    :param merge_if_existing: Whether to merge with existing node/edge dataframes (default True)
    :param merge_if_existing: bool

    """

    g = self.bind()

    #Compute nodes: need indexing for edges

    node_col = g._node or NODE
    
    ig_vs_df = ig.get_vertex_dataframe()
    nodes_df = ig_vs_df


    if load_nodes:

        if node_col not in nodes_df:
            #TODO if no g._nodes but 'name' in nodes_df, still use?
            if (
                ('name' in nodes_df) and  # noqa: W504
                (g._nodes is not None and g._node is not None) and  # noqa: W504
                (g._nodes[g._node].dtype.name == nodes_df['name'].dtype.name)
            ):
                nodes_df = nodes_df.rename(columns={'name': node_col})
            elif ('name' in nodes_df) and (g._nodes is None):
                nodes_df = nodes_df.rename(columns={'name': node_col})
            else:
                nodes_df = nodes_df.reset_index().rename(columns={nodes_df.index.name: node_col})
        
        if node_attributes is not None:
            nodes_df = nodes_df[ node_attributes ]

        if g._nodes is not None and merge_if_existing:
            if len(g._nodes) != len(nodes_df):
                logger.warning('node tables do not match in length; switch merge_if_existing to False or load_nodes to False or add missing nodes')

            g_nodes_trimmed = g._nodes[[x for x in g._nodes if x not in nodes_df or x == g._node]]
            nodes_df = nodes_df.merge(g_nodes_trimmed, how='left', on=g._node)

        nodes_df = nodes_df.reset_index(drop=True)
        g = g.nodes(nodes_df, node_col)

    # #####

    if load_edges:

        # #####

        #TODO: Reuse for g.nodes(nodes_df) as well?

        if len(ig_vs_df.columns) == 0:
            node_id_col = None
        elif node_col in ig_vs_df:
            node_id_col = node_col
        elif g._node is not None and g._nodes[g._node].dtype.name == ig_vs_df.reset_index()['vertex ID'].dtype.name:
            node_id_col = None
        elif 'name' in ig_vs_df:
            node_id_col = 'name'
        else:
            raise ValueError('Could not determine graphistry node ID column to match igraph nodes on')

        if node_id_col is None:
            ig_index_to_node_id_df = None
        else:
            ig_index_to_node_id_df = ig_vs_df[[node_id_col]].reset_index().rename(columns={'vertex ID': 'ig_index'})

        # #####

        src_col = g._source or SRC_IGRAPH
        dst_col = g._destination or DST_IGRAPH
        edges_df = ig.get_edge_dataframe()

        if ig_index_to_node_id_df is not None:
            edges_df['source'] = edges_df[['source']].merge(
                ig_index_to_node_id_df.rename(columns={'ig_index': 'source'}),
                how='left',
                on='source'
            )[node_id_col]
            edges_df['target'] = edges_df[['target']].merge(
                ig_index_to_node_id_df.rename(columns={'ig_index': 'target'}),
                how='left',
                on='target'
            )[node_id_col]

        edges_df = edges_df.rename(columns={
            'source': src_col,
            'target': dst_col
        })

        if edge_attributes is not None:
            edges_df = edges_df[ edge_attributes ]

        if g._edges is not None and merge_if_existing:

            g_indexed = g

            if g._edge is None:
                g_indexed = g.edges(
                    g._edges.reset_index().rename(columns={g._edges.index.name or 'index': '__edge_index__'}),
                    g._source, g._destination, '__edge_index__'
                )
                logger.warning('edge index g._edge not set so using edge index as ID; set g._edge via g.edges(), or change merge_if_existing to False')

            if g_indexed._edge not in edges_df:
                logger.warning('edge index g._edge %s missing as attribute in ig; using ig edge order for IDs', g_indexed._edge)
                edges_df = edges_df.reset_index().rename(columns={edges_df.index.name or 'index': g_indexed._edge})

            if len(g_indexed._edges.columns) == 3 and (len(g_indexed._edges) == len(edges_df)):
                #opt: skip merge: no old columns
                1
            elif ((len(edges_df.columns) == 3) or len(edges_df.columns) == 0) and (len(g._edges) == len(edges_df)):
                #opt: skip merge: no new columns
                edges_df = g._edges
            else:
                if len(g._edges) != len(edges_df):
                    logger.warning('edge tables do not match in length; switch merge_if_existing to False or load_edges to False or add missing edges')
                g_edges_trimmed = g_indexed._edges[[x for x in g_indexed._edges if x not in edges_df or x == g_indexed._edge]]
                edges_df = edges_df.merge(g_edges_trimmed, how='left', on=g_indexed._edge)

            if g._edge is None:
                edges_df = edges_df[[x for x in edges_df if x != g_indexed._edge]]

        g = g.edges(edges_df, src_col, dst_col)

    return g


def to_igraph(self: Plottable, 
    directed: bool = True,
    include_nodes: bool = True,
    node_attributes: Optional[List[str]] = None,
    edge_attributes: Optional[List[str]] = None
):
    """Convert current item to igraph Graph

    :param directed: Whether to create a directed graph (default True)
    :type directed: bool

    :param include_nodes: Whether to ingest the nodes table, if it exists (default True)
    :type include_nodes: bool

    :param node_attributes: Which node attributes to load, None means all (default None)
    :type node_attributes: Optional[List[str]]

    :param edge_attributes: Which edge attributes to load, None means all (default None)
    :type edge_attributes: Optional[List[str]]

    """
    import igraph

    g = self.bind()
    if not include_nodes and self._nodes is not None:
        g._node = None
        g._nodes = None

    #otherwise, if no nodes, ig adds extra column 'name' to vertices
    g = g.materialize_nodes()

    #igraph expects src/dst first
    edge_attrs = g._edges.columns if edge_attributes is None else edge_attributes
    edge_attrs = [x for x in edge_attrs if x not in [g._source, g._destination]]
    edges_df = g._edges[[g._source, g._destination] + edge_attrs]

    #igraph expects node first
    node_attrs = g._nodes if node_attributes is None else node_attributes
    node_attrs = [x for x in node_attrs if x != g._node]
    nodes_df = g._nodes[[g._node] + node_attrs]
    return igraph.Graph.DataFrame(edges_df, directed=directed, vertices=nodes_df)


compute_algs = [
    #'bipartite_projection',
    'clusters',
    'community_edge_betweenness',
    'community_fastgreedy',
    'community_infomap',
    'community_label_propagation',
    #'community_leading_eigenvector_naive',  # in docs but not in code?
    'community_leading_eigenvector',
    'community_leiden',
    'community_multilevel',
    'community_spinglass',
    'community_walktrap',
    'gomory_hu_tree',
    'k_core',
    #'modularity',
    'pagerank',
    'spanning_tree'
]

def compute_igraph(
    self: Plottable, alg: str, alg_as: Optional[str] = None, directed: Optional[bool] = None, params: dict = {}
) -> Plottable:
    """Enrich or replace graph using igraph methods

    :param alg: Name of an igraph.Graph method like `pagerank`
    :type alg: str

    :param alg_as: For algorithms that compute values for nodes, which attribute to write to. If None, use the algorithm's name. (default None)
    :type alg_as: Optional[str]

    :param directed: During the to_igraph conversion, whether to be directed. If None, try undirected and then directed. (default None)
    :type directed: Optional[bool]

    :param params: Any named parameters to pass to the underlying igraph method
    :type params: dict
    """

    if alg not in compute_algs:
        raise ValueError(f'Unexpected parameter alg "{alg}" does not correspond to a known igraph graph.*() algorithm like "pagerank"')

    if alg_as is None:
        alg_as = alg

    try:
        ig = self.to_igraph(directed=directed or False)        
        out = getattr(ig, alg)(**params)
    except NotImplementedError as e:
        if directed is None:
            ig = self.to_igraph(directed=True)        
            out = out = getattr(ig, alg)(**params)
        else:
            raise e

    if isinstance(out, igraph.clustering.VertexClustering):
        clustering = out.membership
    elif isinstance(out, igraph.clustering.VertexDendrogram):
        clustering = out.as_clustering().membership
    elif isinstance(out, igraph.Graph):
        return from_igraph(self, out)
    elif isinstance(out, list) and self._nodes is None:
        raise ValueError("No g._nodes table found; use .bind(), .nodes(), .materialize_nodes()")
    elif len(out) == len(self._nodes):
        clustering = out
    else:
        raise RuntimeError(f'Unexpected output type "{type(out)}"; should be VertexClustering, VertexDendrogram, Graph, or list_<|V|>')    

    ig.vs[alg_as] = clustering

    return self.from_igraph(ig)


layout_algs = [
    'auto', 'automatic',
    'bipartite',
    'circle', 'circular',
    'dh', 'davidson_harel',
    'drl',
    'drl_3d',
    'fr', 'fruchterman_reingold',
    'fr_3d', 'fr3d', 'fruchterman_reingold_3d',
    'grid',
    'grid_3d',
    'graphopt',
    'kk', 'kamada_kawai',
    'kk_3d', 'kk3d', 'kamada_kawai_3d',
    'lgl', 'large', 'large_graph',
    'mds',
    'random', 'random_3d',
    'rt', 'tree', 'reingold_tilford',
    'rt_circular', 'reingold_tilford_circular',
    'sphere', 'spherical', 'circle_3d', 'circular_3d',
    'star',
    'sugiyama'
]

def layout_igraph(
    self: Plottable,
    layout: str,
    directed: Optional[bool] = None,
    bind_position: bool = True,
    x_as: str = 'x',
    y_as: str = 'y',
    play: Optional[int] = 0,
    params: dict = {}
) -> Plottable:
    """Compute graph layout using igraph algorithm. For a list of layouts, see layout_algs or igraph documentation.

    :param layout: Name of an igraph.Graph.layout method like `sugiyama`
    :type layout: str

    :param directed: During the to_igraph conversion, whether to be directed. If None, try undirected and then directed. (default None)
    :type directed: Optional[bool]

    :param bind_position: Whether to call bind(point_x=, point_y=) (default True)
    :type bind_position: bool

    :param x_as: Attribute to write x position to. (default 'x')
    :type x_as: str

    :param y_as: Attribute to write x position to. (default 'y')
    :type y_as: str

    :param play: If defined, set settings(url_params={'play': play}). (default 0)
    :type play: Optional[str]

    :param params: Any named parameters to pass to the underlying igraph method
    :type params: dict
    """

    try:
        ig = self.to_igraph(directed=directed or False)
        layout_df = pd.DataFrame([x for x in ig.layout(layout, **params)])
    except NotImplementedError as e:
        if directed is None:
            ig = self.to_igraph(directed=True)
            layout_df = pd.DataFrame([x for x in ig.layout(layout, **params)])
        else:
            raise e

    g2 = self.from_igraph(ig)
    g2 = g2.nodes(g2._nodes.assign(**{x_as: layout_df[0], y_as: layout_df[1]}))
    if bind_position:
        g2 = g2.bind(point_x=x_as, point_y=y_as)
    if play is not None:
        g2 = g2.settings(url_params={'play': play})
    return g2
