import pandas as pd
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

    :returns: Plotter
    :rtype: Plotter

    **Example: Convert from igraph, including all node/edge properties**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a', 'b', 'c', 'd'], 'd': ['b', 'c', 'd', 'e'], 'v': [101, 102, 103, 104]})
            g = graphistry.edges(edges, 's', 'd').materialize_nodes().get_degrees()
            assert 'degree' in g._nodes.columns
            g2 = g.from_igraph(g.to_igraph())
            assert len(g2._nodes.columns) == len(g._nodes.columns)

    **Example: Enrich from igraph, but only load in 1 node attribute**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a', 'b', 'c', 'd'], 'd': ['b', 'c', 'd', 'e'], 'v': [101, 102, 103, 104]})
            g = graphistry.edges(edges, 's', 'd').materialize_nodes().get_degree()
            assert 'degree' in g._nodes
            ig = g.to_igraph(include_nodes=False)
            assert 'degree' not in ig.vs
            ig.vs['pagerank'] = ig.pagerank()
            g2 = g.from_igraph(ig, load_edges=False, node_attributes=[g._node, 'pagerank'])
            assert 'pagerank' in g2._nodes
            asssert 'degree' in g2._nodes

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
    edge_attributes: Optional[List[str]] = None,
    use_vids: bool = False
):
    """Convert current item to igraph Graph . See examples in from_igraph.

    :param directed: Whether to create a directed graph (default True)
    :type directed: bool

    :param include_nodes: Whether to ingest the nodes table, if it exists (default True)
    :type include_nodes: bool

    :param node_attributes: Which node attributes to load, None means all (default None)
    :type node_attributes: Optional[List[str]]

    :param edge_attributes: Which edge attributes to load, None means all (default None)
    :type edge_attributes: Optional[List[str]]

    :param use_vids: Whether to interpret IDs as igraph vertex IDs, which must be non-negative integers (default False)
    :type use_vids: bool

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
    return igraph.Graph.DataFrame(
        edges_df, directed=directed, vertices=nodes_df, use_vids=use_vids
    )


compute_algs = [
    'authority_score',
    'betweenness',
    'bibcoupling',
    #'biconnected_components',
    #'bipartite_projection',
    'harmonic_centrality',
    'closeness',
    'clusters',
    'cocitation',
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
    'constraint',
    'coreness',
    'gomory_hu_tree',
    'harmonic_centrality',
    'hub_score',
    'eccentricity',
    'eigenvector_centrality',
    'k_core',
    #'modularity',
    'pagerank',
    'spanning_tree'
]

def compute_igraph(
    self: Plottable,
    alg: str,
    out_col: Optional[str] = None,
    directed: Optional[bool] = None,
    use_vids=False,
    params: dict = {}
) -> Plottable:
    """Enrich or replace graph using igraph methods

    :param alg: Name of an igraph.Graph method like `pagerank`
    :type alg: str

    :param out_col: For algorithms that generate a node attribute column, `out_col` is the desired output column name. When `None`, use the algorithm's name. (default None)
    :type out_col: Optional[str]

    :param directed: During the to_igraph conversion, whether to be directed. If None, try directed and then undirected. (default None)
    :type directed: Optional[bool]

    :param use_vids: During the to_igraph conversion, whether to interpret IDs as igraph vertex IDs (non-negative integers) or arbitrary values (False, default)
    :type use_vids: bool

    :param params: Any named parameters to pass to the underlying igraph method
    :type params: dict

    :returns: Plotter
    :rtype: Plotter

    **Example: Pagerank**

        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.compute_igraph('pagerank')
            assert 'pagerank' in g2._nodes.columns

    **Example: Pagerank with custom name**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.compute_igraph('pagerank', out_col='my_pr')
            assert 'my_pr' in g2._nodes.columns

    **Example: Pagerank on an undirected**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.compute_igraph('pagerank', directed=False)
            assert 'pagerank' in g2._nodes.columns

    **Example: Pagerank with custom parameters**
            ::
                import graphistry, pandas as pd
                edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['c','c','e','e']})
                g = graphistry.edges(edges, 's', 'd')
                g2 = g.compute_igraph('pagerank', params={'damping': 0.85})
                assert 'pagerank' in g2._nodes.columns

    """

    import igraph

    if alg not in compute_algs:
        raise ValueError(f'Unexpected parameter alg "{alg}" does not correspond to a known igraph graph.*() algorithm like "pagerank"')

    if out_col is None:
        out_col = alg

    try:
        ig = self.to_igraph(directed=True if directed is None else directed, use_vids=use_vids)
        out = getattr(ig, alg)(**params)
    except NotImplementedError as e:
        if directed is None:
            ig = self.to_igraph(directed=False, use_vids=use_vids)
            out = getattr(ig, alg)(**params)
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

    ig.vs[out_col] = clustering

    return self.from_igraph(ig)


layout_algs = [
    'auto', 'automatic',
    'bipartite',
    'circle', 'circular',
    #'connected_components',
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
    'sugiyama',
    #'umap'
]

def layout_igraph(
    self: Plottable,
    layout: str,
    directed: Optional[bool] = None,
    use_vids: bool = False,
    bind_position: bool = True,
    x_out_col: str = 'x',
    y_out_col: str = 'y',
    play: Optional[int] = 0,
    params: dict = {}
) -> Plottable:
    """Compute graph layout using igraph algorithm. For a list of layouts, see layout_algs or igraph documentation.

    :param layout: Name of an igraph.Graph.layout method like `sugiyama`
    :type layout: str

    :param directed: During the to_igraph conversion, whether to be directed. If None, try directed and then undirected. (default None)
    :type directed: Optional[bool]

    :param use_vids: Whether to use igraph vertex ids (non-negative integers) or arbitary node ids (False, default)
    :type use_vids: bool

    :param bind_position: Whether to call bind(point_x=, point_y=) (default True)
    :type bind_position: bool

    :param x_out_col: Attribute to write x position to. (default 'x')
    :type x_out_col: str

    :param y_out_col: Attribute to write x position to. (default 'y')
    :type y_out_col: str

    :param play: If defined, set settings(url_params={'play': play}). (default 0)
    :type play: Optional[str]

    :param params: Any named parameters to pass to the underlying igraph method
    :type params: dict

    :returns: Plotter
    :rtype: Plotter

    **Example: Sugiyama layout**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.layout_igraph('sugiyama')
            assert 'x' in g2._nodes
            g2.plot()

    **Example: Change which column names are generated**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.layout_igraph('sugiyama', x_out_col='my_x', y_out_col='my_y')
            assert 'my_x' in g2._nodes
            assert g2._point_x == 'my_x'
            g2.plot()

    **Example: Pass parameters to layout methods - Sort nodes by degree**
        ::
            import graphistry, pandas as pd
            edges = pd.DataFrame({'s': ['a','b','c','d'], 'd': ['b','c','d','e']})
            g = graphistry.edges(edges, 's', 'd')
            g2 = g.get_degrees()
            assert 'degree' in g._nodes.columns
            g3 = g.layout_igraph('sugiyama', params={'layers': 'degree'})
            g3.plot()
    """

    try:
        ig = self.to_igraph(directed=True if directed is None else directed, use_vids=use_vids)
        layout_df = pd.DataFrame([x for x in ig.layout(layout, **params)])
    except NotImplementedError as e:
        if directed is None:
            ig = self.to_igraph(directed=False, use_vids=use_vids)
            layout_df = pd.DataFrame([x for x in ig.layout(layout, **params)])
        else:
            raise e

    g2 = self.from_igraph(ig)
    g2 = g2.nodes(g2._nodes.assign(**{x_out_col: layout_df[0], y_out_col: layout_df[1]}))
    if bind_position:
        g2 = g2.bind(point_x=x_out_col, point_y=y_out_col)
    if play is not None:
        g2 = g2.layout_settings(play=play)
    return g2
