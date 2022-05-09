from typing import List, Optional
from graphistry.constants import NODE
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
):
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
    
    nodes_df = ig.get_vertex_dataframe()

    if len(nodes_df.columns) == 0:
        node_id_col = None
    elif node_col in nodes_df:
        node_id_col = node_col
    elif g._node is not None and g._nodes[g._node].dtype.name == nodes_df.reset_index()['vertex ID'].dtype.name:
        node_id_col = None
    elif 'name' in nodes_df:
        node_id_col = 'name'
    else:
        raise ValueError('Could not determine graphistry node ID column to match igraph nodes on')

    if node_id_col is None:
        ig_index_to_node_id_df = None
    else:
        ig_index_to_node_id_df = nodes_df[[node_id_col]].reset_index().rename(columns={'vertex ID': 'ig_index'})
    
    if node_col not in nodes_df:
        #TODO if no g._nodes but 'name' in nodes_df, still use?
        if (
            ('name' in nodes_df) and  # noqa: W504
            (g._nodes is not None and g._node is not None) and  # noqa: W504
            (g._nodes[g._node].dtype.name == nodes_df['name'].dtype.name)
        ):
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

    # #####

    if load_nodes:
        g = g.nodes(nodes_df, node_col)

    # #####

    if load_edges:

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


def to_igraph(self, 
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
